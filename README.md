# ViR-MFS

[![DOI](https://zenodo.org/badge/1164573018.svg)](https://doi.org/10.5281/zenodo.18740397)

Official PyTorch implementation of **"Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception"**.

**Authors:** Shichao Liu<sup>1</sup> · Chun Sun<sup>2</sup> · Dengshi Li<sup>1</sup> · Yihui Wang<sup>1</sup> · Shiwei Hu<sup>1</sup> · Zhiming Zhan

Shichao Liu and Chun Sun contributed equally to this work.

ViR-MFS is an end-to-end infrared-visible image fusion and semantic segmentation framework. It uses a shared SegFormer-style backbone, a multi-scale wavelet fusion module, and alternating meta-learning optimization to improve both fused-image quality and downstream semantic segmentation.

## Repository Layout

```text
ViR-MFS/
|-- config/
|   |-- config.yaml        # Dataset, checkpoint, backbone, and output paths.
|   `-- params.yaml        # Training, testing, wavelet, and visualization settings.
|-- data_pipeline/
|   `-- dataloader.py
|-- engine/
|   |-- training.py
|   `-- testing.py
|-- nets/
|   |-- backbone.py
|   |-- segformer.py
|   |-- wavelet.py
|   `-- wtconv2d.py
|-- utils/
|-- config_loader.py
|-- train.py
|-- test.py
|-- run_experiment.sh
|-- environment.yml
`-- README.md
```

Runtime outputs are intentionally excluded from version control:

```text
model_data/      # Pretrained backbone weights and ViR-MFS checkpoints.
runs/            # Training checkpoints, logs, and run history.
test_results/    # Fused images, segmentation masks, and evaluation outputs.
```

## Environment

Create the conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate vir-mfs
```

The environment file targets Python 3.10 and PyTorch 2.3.0 with CUDA 11.8 wheels. If your CUDA driver or hardware requires a different PyTorch build, install the matching PyTorch/Torchvision packages first and then install the remaining packages listed in `environment.yml`.

## Dataset Preparation

Download the datasets from their official sources:

- [MSRS Dataset](https://github.com/Linfeng-Tang/MSRS)
- [FMB Dataset](https://github.com/JinyuanLiu-CV/SegMiF)

Place datasets outside the repository, preferably as a sibling directory:

```text
workspace/
|-- datasets/
|   |-- MSRS/
|   |   |-- ir/
|   |   |   |-- train/
|   |   |   `-- test/
|   |   |-- vi/
|   |   |   |-- train/
|   |   |   `-- test/
|   |   `-- label/
|   |       |-- train/
|   |       `-- test/
|   `-- FMB/
|       |-- ir/
|       |-- vi/
|       `-- label/
`-- ViR-MFS/
```

The default dataset path is `../datasets`, configured in `config/config.yaml`:

```yaml
dataset:
  name: "MSRS"
  root_dir: "../datasets"
```

For FMB evaluation or training, set `dataset.name` to `"FMB"` and update `train.num_classes` / `test.num_classes` in `config/params.yaml` to match the dataset setting.

## Pretrained Models

Download pretrained ViR-MFS weights from [Google Drive](https://drive.google.com/drive/folders/11dXQ-pkYgPVe9qD4AXCpv-XIn5JZIMGh?usp=sharing).

Place the downloaded `.pth` files under `model_data/`, or set an explicit checkpoint path in `config/config.yaml`:

```yaml
test:
  checkpoint_name: "best_mIoU"
```

Short checkpoint names such as `best_mIoU` are resolved by the code according to the configured dataset, backbone, and training output directory. Absolute paths and explicit `.pth` filenames are also supported.

## Quick Start

Run evaluation first after preparing the dataset and checkpoint:

```bash
bash run_experiment.sh 0 test
```

Arguments are:

```bash
bash run_experiment.sh [GPU_ID] [MODE] [PARAMS_PATH] [CONFIG_PATH]
```

Examples:

```bash
bash run_experiment.sh 0 test
bash run_experiment.sh 0 train
bash run_experiment.sh 0,1,2,3 train
```

Testing saves fused images, raw segmentation masks, colorized predictions, and run metadata under `test_results/`.

## Configuration

Main settings live in two YAML files:

- `config/config.yaml`: dataset root, dataset name, backbone variant, pretrained-weight directory, checkpoint name, and output directories.
- `config/params.yaml`: batch size, resize size, class count, learning rates, meta-learning settings, wavelet settings, checkpoint strictness, and visualization options.

Useful defaults:

```yaml
wavelet:
  high_frequency_source: "visible"
  high_frequency_injection: "static"

test:
  label_resize_interpolation: "nearest"
  visualization:
    palette: "auto"
    save_pred_color: true
    save_label_color: true
```

Use nearest-neighbor label resizing for standard semantic segmentation evaluation so integer class IDs are preserved.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{ViRMFS2026,
  title={Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception},
  author={Shichao Liu and Chun Sun and Dengshi Li and Yihui Wang and Shiwei Hu and Zhiming Zhan},
  journal={The Visual Computer},
  year={2026}
}
```

## Contact

For questions about the code or paper, please contact `wyh37133@gmail.com`.
