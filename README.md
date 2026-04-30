# ViR-MFS: Wavelet-Driven Meta-Learning for Infrared-Visible Fusion and Segmentation

[![DOI](https://zenodo.org/badge/1164573018.svg)](https://doi.org/10.5281/zenodo.18740397)

Official PyTorch implementation of the paper: **"Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception"** (Currently under review / Submitted to *The Visual Computer*).

> **Important Note for Readers:** If you find this code, our dataset processing, or our methodology useful in your research, please kindly consider citing our manuscript submitted to *The Visual Computer*. Citation details will be updated immediately upon publication.

---

## Abstract

Infrared and visible image fusion is a pivotal task in computer vision, aiming to integrate complementary modal characteristics to generate fused images suitable for both human visual observation and machine perception. However, most state-of-the-art fusion algorithms prioritise visual quality at the expense of utility for downstream high-level vision tasks, and existing semantic-driven methods face critical limitations: noise aliasing from coarse frequency-domain processing and gradient conflicts in multi-task joint training of fusion and segmentation. To address these challenges, this paper proposes ViR-MFS (Visible and Infrared Image Meta-learning Framework for Fusion and Segmentation), an end-to-end joint network for infrared-visible image fusion and segmentation based on alternating meta-learning optimisation and a multi-scale wavelet fusion module. First, a MixVisionTransformer is adopted as a shared backbone to construct a multi-scale feature pyramid that balances local spatial continuity and global semantic dependencies. Second, a Multi-scale Wavelet Fusion Module (MWFM) is designed to explicitly decouple high- and low-frequency components in the frequency domain; with learnable scale factors for adaptive recalibration, MWFM enables precise injection of visible texture details while effectively suppressing infrared thermal noise. To resolve objective conflicts in multi-task learning, an alternating meta-learning optimisation strategy is introduced, which dynamically coordinates the training of fusion and segmentation tasks to guide the backbone in learning robust features with both fine texture details and strong semantic discriminability, mitigating inter-task performance trade-offs. Here we show that extensive experiments on the MSRS and FMB datasets demonstrate ViR-MFS outperforms all state-of-the-art methods across key evaluation metrics: on the MSRS dataset, it achieves a mutual information (MI) of 5.212, edge fidelity ($Q^{AB/F}$) of 0.721, and a mean Intersection over Union (mIoU) of 58.96% for semantic segmentation; on the FMB dataset, it attains an MI of 4.956, $Q^{AB/F}$ of 0.730, and an mIoU of 64.15%. These results validate ViR-MFS’s superior performance in both visual fusion quality and semantic segmentation accuracy, highlighting its significant potential for all-weather complex scene perception applications in critical domains such as autonomous driving and video surveillance.

---

## Algorithm Core

ViR-MFS is an end-to-end infrared-visible fusion and semantic segmentation framework. Its key components are:

* A shared MixVisionTransformer / SegFormer backbone extracts multi-scale visible and infrared features.
* The Multi-scale Wavelet Fusion Module (MWFM) fuses low-frequency structures and, by default, reconstructs high-frequency details from the visible branch to keep the original asymmetric infrared-visible design.
* A fusion head predicts adaptive visible/infrared fusion weights, while a segmentation head predicts semantic masks from the fused feature pyramid.
* Alternating meta-learning updates the fusion branch and segmentation branch to reduce multi-task gradient conflict.

## Directory Structure

The recommended workspace keeps datasets outside the code repository and at the same level as `ViR-MFS/`. The code root remains the execution root. Configuration, environment scripts, requirements, README, and `.gitignore` stay at the repository root.

```text
workspace/
├── datasets/
│   ├── FMB/
│   │   ├── ir/
│   │   │   ├── train/
│   │   │   └── test/
│   │   ├── label/
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── vi/
│   │       ├── train/
│   │       └── test/
│   ├── MSRS/
│   │   ├── ir/
│   │   │   ├── train/
│   │   │   └── test/
│   │   ├── label/
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── vi/
│   │       ├── train/
│   │       └── test/
│   └── ...
└── ViR-MFS/
    ├── config/
    │   ├── config.yaml         # Dataset, backbone, output, and checkpoint paths.
    │   ├── config_fmb_legacy.yaml # Legacy FMB dataset/checkpoint paths.
    │   ├── params.yaml         # Strict train/test/meta-learning/wavelet parameters.
    │   └── params_fmb_legacy.yaml # Legacy FMB 17-class checkpoint reproduction.
    ├── config_loader.py        # YAML loader and ConfigInjector.
    ├── data_pipeline/
    │   └── dataloader.py
    ├── engine/
    │   ├── training.py         # Training/fine-tuning orchestration.
    │   └── testing.py          # Testing and output orchestration.
    ├── nets/                   # Network definitions. Backbone topology is unchanged.
    │   ├── backbone.py
    │   ├── segformer.py
    │   ├── wavelet.py
    │   └── wtconv2d.py
    ├── utils/
    │   ├── checkpoint.py
    │   ├── common.py
    │   ├── evaluation.py
    │   ├── experiment.py
    │   ├── losses.py
    │   ├── metrics.py
    │   ├── runtime.py
    │   ├── seg_visualization.py
    │   ├── utils_logger.py
    │   └── utils_meta.py
    ├── train.py                # Thin training entrypoint.
    ├── test.py                 # Thin testing entrypoint.
    ├── run_experiment.sh
    ├── build_ViR_MFS_env.sh
    ├── requirements.txt
    ├── .gitignore
    └── README.md
```

**Note:** The following directories are generated during runtime and are ignored by version control:
* `model_data/`: Directory for pre-trained backbone weights.
* `runs/`: Checkpoints, training logs, and timestamped traceability history.
* `test_results/`: Output directories for fused images and segmentation masks.

---

## Environment Setup

Please ensure you have Python 3.8+ and PyTorch 2.x (e.g., PyTorch 2.2.2+cu118 or PyTorch 2.3.0+cu118) installed matching your CUDA environment.

Run the provided shell script from the repository root to build the environment and install dependencies:

```bash
bash build_ViR_MFS_env.sh
# Alternatively, install manually: pip install -r requirements.txt
```

---

## Dataset Preparation

We evaluate our method on the **MSRS** and **FMB** datasets. 

1. Download the datasets from the following links:
   - [MSRS Dataset](https://github.com/Linfeng-Tang/MSRS)
   - [FMB Dataset](https://github.com/JinyuanLiu-CV/SegMiF)
   
   *(Note: The original copyrights of the datasets belong to their respective authors. We provide these links solely to facilitate reproducibility.)*

2. Organize the downloaded datasets strictly matching the `workspace/datasets/` structure shown in the Directory Structure section above.
3. Set `config/config.yaml -> dataset.root_dir` to the absolute or relative path of the `datasets/` directory. For the recommended sibling layout, this is typically `../datasets` when running from `ViR-MFS/`.

---

## Quick Start

Hardcoded training/testing paths have been removed from the active entrypoints. Runtime dependencies are resolved through `ConfigInjector` from YAML configuration files.

### 1. Configuration
* **`config.yaml`**: Ensure `dataset.root_dir` points to your `datasets/` folder and `backbone.pretrained_dir` points to `model_data/` or another pretrained-weight directory.
* **`params.yaml`**: Configure training hyperparameters, testing parameters, meta-learning controls, resume/fine-tuning settings, and MWFM wavelet parameters.
* **Dataset class counts**: MSRS uses valid label ids `0-8` and should use `num_classes: 9`. FMB uses valid label ids `0-14` and should normally use `num_classes: 15`. The FMB visualization palette defines 15 entries: background, road, sidewalk, building, lamp, sign, vegetation, sky, person, car, truck, bus, motorcycle, bicycle, and pole.
* **Label resize policy**: `nearest` preserves integer semantic labels and is the strict segmentation setting. `default` reproduces the legacy `PIL.Image.resize(size)` behavior from the old test script and may create interpolated pseudo label ids, such as FMB ids `15` or `16`, which are not valid FMB palette classes.
* **Legacy FMB checkpoints**: FMB checkpoints trained with `num_classes: 17` have a 17-channel segmentation head and cannot be loaded into a 15-class model because `decode_head.linear_pred` shapes differ. Use `config/params_fmb_legacy.yaml` to reproduce those old checkpoints.
* **Segmentation visualization**: `test.visualization.palette` selects the semantic palette (`auto`, `MSRS`, or `FMB`). `save_pred_color` saves colorized predicted masks, while `save_label_color` saves colorized ground-truth labels for side-by-side inspection.
* **MWFM high-frequency control**: `params.yaml -> wavelet.high_frequency_source` defaults to `visible`, preserving the asymmetric design. Supported values are `visible`, `infrared`, `mean`, and `sum`.
* **Learnable high-frequency injection**: `params.yaml -> wavelet.high_frequency_injection` can be `learnable` or `static`. The learnable mode initializes from `high_frequency_source` and then optimizes visible/infrared high-frequency injection weights during training.
* **Checkpoint compatibility**: old checkpoints trained before `HighFrequencyInjectionController` do not contain `f0/f1/f2.high_frequency_controller.logits`. Set `test.checkpoint_strict: false` for evaluation or `train.resume.strict: false` for fine-tuning old checkpoints. The new high-frequency controller is still created; missing logits are initialized from the current wavelet config instead of being loaded from the checkpoint.
* **Traceability**: each training or testing run writes a timestamped `history/<run_id>/` folder under its output directory, including configs, resolved configs, manifest, git status, git diff, requirements, and pip freeze.

### 2. Training
Use the unified shell script to start training. The script automatically handles environment variables and memory optimization:

```bash
# Usage: bash run_experiment.sh [GPU_ID] [MODE] [PARAMS_PATH] [CONFIG_PATH]
bash run_experiment.sh 0 train
```

### 3. Fine-tuning
Fine-tuning is merged into `train.py`; there is no separate fine-tuning script. Enable it in `config/params.yaml`:

```yaml
train:
  resume:
    enabled: true
    checkpoint: "/path/to/checkpoint.pth"
    strict: false  # Use false when fine-tuning older checkpoints without new modules.
```

### 4. Evaluation
To evaluate the model and generate fused images alongside segmentation masks, simply change the mode to `test`:

```bash
bash run_experiment.sh 0 test
```
The outputs (fused images and predicted masks) will be automatically saved to the `test_results/` directory as specified in your `config.yaml`.

To reproduce an old FMB checkpoint trained with a 17-class segmentation head, run:

```bash
bash run_experiment.sh 0 test config/params_fmb_legacy.yaml config/config_fmb_legacy.yaml
```

For strict FMB evaluation and new training, keep `num_classes: 15` and `label_resize_interpolation: nearest` in `config/params.yaml`.

Semantic visualization is controlled in `config/params.yaml`:

```yaml
test:
  visualization:
    palette: "auto"        # auto uses config.yaml -> dataset.name.
    save_pred_color: true  # Save colorized predicted segmentation masks.
    save_label_color: true # Save colorized ground-truth label masks.
```

Raw prediction id masks are still saved in the existing `_seg` directory. Colorized prediction masks are saved to `_seg_color`, and colorized ground-truth labels are saved to `_seg_label_color`.

When evaluating older checkpoints, keep this setting in `config/params.yaml`:

```yaml
test:
  checkpoint_strict: false
```

With non-strict loading, all matching checkpoint parameters are loaded, while newly added parameters such as `high_frequency_controller.logits` keep their config-based initialization.

---

## Pre-trained Models

For quick inference and reproduction of the results reported in our paper, you can download our pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/11dXQ-pkYgPVe9qD4AXCpv-XIn5JZIMGh?usp=sharing).

Please place the downloaded `.pth` files into the root-level `model_data/` directory (create it if it does not exist) or specify the exact checkpoint path in `config.yaml` / `params.yaml`.

---

## Citation

If you find this code, our dataset processing, or our methodology useful in your research, please kindly consider citing our manuscript:

```bibtex
@article{ViRMFS2026,
  title={Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception},
  author={Shichao Liu and Dengshi Li and Yihui Wang and Shiwei Hu and Zhiming Zhan},
  journal={The Visual Computer},
  year={2026}
}
```

---

## Contact

If you have any questions about the code or paper, please feel free to open an issue or contact `wyh37133@gmail.com`.
