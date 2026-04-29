# utils_meta.py
import os
import torch
import torch.nn as nn

# Checkpoint loading helpers.
def _strip_module(sd: dict):
    """Remove DDP ``module.`` prefixes from state-dict keys.

    Args:
        sd: Source state dictionary.

    Returns:
        State dictionary without ``module.`` prefixes.
    """
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _add_module(sd: dict):
    """Add DDP ``module.`` prefixes to state-dict keys.

    Args:
        sd: Source state dictionary.

    Returns:
        State dictionary with ``module.`` prefixes.
    """
    return { (k if k.startswith("module.") else f"module.{k}"): v for k, v in sd.items() }

def load_partial_weights(model, weight_path, device="cuda"):
    """Load matching checkpoint weights into a model.

    Args:
        model: Target model.
        weight_path: Checkpoint path.
        device: Device used for checkpoint loading.

    Returns:
        Model with compatible weights loaded.
    """
    if not os.path.exists(weight_path):
        print(f"[WARN] Pretrained checkpoint not found: {weight_path}")
        return model

    # Prefer safe loading when the installed PyTorch version supports it.
    try:
        ckpt = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Unexpected checkpoint format")

    model_dict = model.state_dict()

    # Try raw keys, stripped DDP keys, and added DDP keys.
    cands = [("raw", sd), ("strip", _strip_module(sd)), ("add", _add_module(sd))]

    best_name, best_matched, best_dict = None, -1, None
    for name, cand in cands:
        matched = {k: v for k, v in cand.items() if k in model_dict and v.size() == model_dict[k].size()}
        if len(matched) > best_matched:
            best_matched = len(matched)
            best_name, best_dict = name, matched

    print(f"[INFO] Pretrained checkpoint: {weight_path}")
    print(f"[INFO] Key format: {best_name}, matched parameters: {best_matched}/{len(model_dict)}")

    model_dict.update(best_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

# Parameter helpers.
# def get_fusion_param_names(m: nn.Module):
#     keep_prefix = ("shallow1", "shallow2", "seg1", "seg2", "seg3", "fusion_task_head")
#     names = [n for n, p in m.named_parameters() if any(n.startswith(k) for k in keep_prefix)]
#     return names
def get_fusion_param_names(m: nn.Module):
    """Return parameter names belonging to the fusion branch.

    Args:
        m: Model to inspect.

    Returns:
        List of fusion parameter names.
    """
    keep_prefix = ("f0", "f1", "f2", "f3", "fusion_head")
    names = [n for n, p in m.named_parameters() if any(n.startswith(k) for k in keep_prefix)]
    return names

def make_params_dict(m: nn.Module):
    """Convert model parameters to a name-parameter dictionary.

    Args:
        m: Model to inspect.

    Returns:
        Dictionary mapping parameter names to tensors.
    """
    return {name: p for name, p in m.named_parameters()}

def merge_updated_params(params_all, names_to_update, updates, inner_lr):
    """Build a virtual parameter dictionary for a meta-learning inner step.

    Args:
        params_all: Full parameter dictionary.
        names_to_update: Parameter names updated in the inner loop.
        updates: Gradient tensors for selected parameters.
        inner_lr: Inner-loop learning rate.

    Returns:
        New parameter dictionary with selected parameters updated.
    """
    new_params = dict(params_all)
    for name, g in zip(names_to_update, updates):
        new_params[name] = params_all[name] - inner_lr * g
    return new_params

# Miscellaneous helpers.
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
