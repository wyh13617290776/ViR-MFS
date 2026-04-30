import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast
# from torch.nn.utils.stateless import functional_call # torch 1.x
from torch.func import functional_call # torch 2.x


def select_named_params_by_prefix(model, prefixes):
    """Select trainable parameters by name prefix.

    Args:
        model: Model or DDP-wrapped model.
        prefixes: Iterable of accepted parameter-name prefixes.

    Returns:
        Tuple ``(names, params)`` containing selected parameter names and
        parameter tensors.
    """
    pairs = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not pairs:
        return tuple(), tuple()
    names, params = zip(*pairs)
    return tuple(names), tuple(params)


def select_named_params_containing(model, token: str):
    """Select trainable parameters whose names contain a token.

    Args:
        model: Model or DDP-wrapped model.
        token: Name fragment used for selection.

    Returns:
        Tuple ``(names, params)`` containing selected parameter names and
        parameter tensors.
    """
    pairs = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and token in name
    ]
    if not pairs:
        return tuple(), tuple()
    names, params = zip(*pairs)
    return tuple(names), tuple(params)


def build_meta_parameter_groups(model, distributed: bool):
    """Build fusion and segmentation parameter groups for meta-learning.

    Args:
        model: Model or DDP-wrapped model.
        distributed: Whether the model is wrapped by DistributedDataParallel.

    Returns:
        Tuple ``(fusion_names, fusion_params, seg_names, seg_params)``.
    """
    fusion_prefixes = (
        "module.f0",
        "module.f1",
        "module.f2",
        "module.f3",
        "module.fusion_head",
    ) if distributed else (
        "f0",
        "f1",
        "f2",
        "f3",
        "fusion_head",
    )
    fusion_names, fusion_params = select_named_params_by_prefix(model, fusion_prefixes)
    seg_names, seg_params = select_named_params_containing(model, "decode_head")
    return fusion_names, fusion_params, seg_names, seg_params


def resolve_meta_target(epoch: int, inner_warmup: int, inner_every: int, meta_epoch_index: int):
    """Resolve whether an epoch should run a meta step and which branch to use.

    Args:
        epoch: Zero-based training epoch index.
        inner_warmup: Number of warmup epochs before meta-learning starts.
        inner_every: Epoch interval for meta-learning after warmup.
        meta_epoch_index: Number of previous meta epochs used for alternating
            fusion and segmentation branches.

    Returns:
        Tuple ``(meta_target, next_meta_epoch_index)``. ``meta_target`` is
        ``None`` when the epoch should not run meta-learning; otherwise it is
        ``fusion`` or ``seg``.
    """
    safe_inner_every = max(1, int(inner_every))
    should_meta_epoch = (
        epoch >= int(inner_warmup)
        and ((epoch - int(inner_warmup)) % safe_inner_every == 0)
    )
    if not should_meta_epoch:
        return None, meta_epoch_index
    meta_target = "fusion" if meta_epoch_index % 2 == 0 else "seg"
    return meta_target, meta_epoch_index + 1


def make_params_dict(m: nn.Module):
    """Convert model parameters to a name-parameter dictionary.

    Args:
        m: Model to inspect.

    Returns:
        Dictionary mapping parameter names to tensors.
    """
    return {name: p for name, p in m.named_parameters()}


def has_invalid_tensor(value: torch.Tensor) -> bool:
    """Check whether a tensor contains NaN or Inf values.

    Args:
        value: Tensor to inspect.

    Returns:
        True if the tensor has invalid numeric values.
    """
    return bool(torch.isnan(value).any() or torch.isinf(value).any())


def has_invalid_grads(grads) -> bool:
    """Check whether any gradient tensor contains NaN or Inf.

    Args:
        grads: Iterable of gradient tensors.

    Returns:
        True if any gradient is numerically invalid.
    """
    return any(has_invalid_tensor(grad) for grad in grads)


def merge_updated_params(params_all, names_to_update, updates, inner_lr):
    """Build a virtual parameter dictionary for a meta-learning inner step.

    Args:
        params_all: Full model parameter dictionary.
        names_to_update: Names of parameters updated in the virtual inner loop.
        updates: Gradient tensors corresponding to ``names_to_update``.
        inner_lr: Inner-loop learning rate used for the virtual update.

    Returns:
        New parameter dictionary with only the selected parameters updated.
    """
    updated_params = dict(params_all)
    for name, grad in zip(names_to_update, updates):
        # Keep untouched parameters shared and replace only the inner-loop subset.
        updated_params[name] = params_all[name] - inner_lr * grad
    return updated_params


def assign_meta_grads(params, grads, context) -> None:
    """Assign and synchronize manually computed meta gradients.

    Args:
        params: Target model parameters.
        grads: Gradient tensors computed by ``torch.autograd.grad``.
        context: Runtime context containing distributed-training state.

    Returns:
        None.
    """
    for param, grad in zip(params, grads):
        param.grad = grad.detach().clone()
    if context.distributed:
        for param in params:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()


def maybe_clip_grad_norm(params, max_norm) -> None:
    """Clip gradients when a maximum norm is configured.

    Args:
        params: Parameters whose gradients should be clipped.
        max_norm: Maximum norm or ``None`` to disable clipping.

    Returns:
        None.
    """
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_norm))


def run_meta_step(
    model,
    params,
    names,
    optimizer,
    support_loss_fn,
    query_loss_fn,
    support_args,
    query_args,
    inner_lr: float,
    use_amp: bool,
    context,
    grad_clip_norm,
) -> bool:
    """Run one MAML-style second-order update for a parameter subset.

    Args:
        model: Model or DDP-wrapped model.
        params: Parameters updated by the meta step.
        names: Names corresponding to ``params``.
        optimizer: Optimizer for the selected parameter subset.
        support_loss_fn: Callable that receives support model outputs and
            returns a scalar loss.
        query_loss_fn: Callable that receives query model outputs and returns a
            scalar loss.
        support_args: Positional model inputs for the inner-loop batch.
        query_args: Positional model inputs for the outer-loop batch.
        inner_lr: Inner-loop virtual update learning rate.
        use_amp: Whether AMP autocast is enabled.
        context: Runtime context containing distributed-training state.
        grad_clip_norm: Optional gradient clipping norm.

    Returns:
        True if an optimizer step was applied, otherwise False.
    """
    if optimizer is None or not params:
        return False

    with autocast(enabled=use_amp):
        support_outputs = model(*support_args, return_lists=True)
        support_loss = support_loss_fn(support_outputs)

    grads = torch.autograd.grad(
        support_loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    grads = [grad if grad is not None else torch.zeros_like(param) for grad, param in zip(grads, params)]
    if has_invalid_grads(grads):
        return False

    updated = merge_updated_params(make_params_dict(model), names, grads, inner_lr)

    with autocast(enabled=use_amp):
        query_outputs = functional_call(
            model,
            updated,
            query_args,
            {"return_lists": True},
        )
        query_loss = query_loss_fn(query_outputs)

    meta_grads = torch.autograd.grad(query_loss, params, retain_graph=False, allow_unused=True)
    meta_grads = [grad if grad is not None else torch.zeros_like(param) for grad, param in zip(meta_grads, params)]
    if has_invalid_grads(meta_grads):
        return False

    optimizer.zero_grad(set_to_none=True)
    assign_meta_grads(params, meta_grads, context)
    maybe_clip_grad_norm(params, grad_clip_norm)
    optimizer.step()
    return True


def split_mtr_mts(x: torch.Tensor):
    """Split a batch into meta-train and meta-test halves.

    Args:
        x: Input batch tensor.

    Returns:
        Tuple ``(meta_train, meta_test)``. A batch of size one is duplicated.
    """
    b = x.shape[0]
    mid = b // 2
    if mid == 0:
        return x, x
    return x[:mid], x[mid:]
