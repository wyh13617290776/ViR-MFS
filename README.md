# ViR-MFS: Wavelet-Driven Meta-Learning for Infrared-Visible Fusion and Segmentation

[![DOI](https://zenodo.org/badge/1164573018.svg)](https://doi.org/10.5281/zenodo.18740397)

Official PyTorch implementation of the paper: **"Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception"** (Currently under review / Submitted to *The Visual Computer*).

> **Important Note for Readers:** > If you find this code, our dataset processing, or our methodology useful in your research, please kindly consider citing our manuscript submitted to *The Visual Computer*. (Citation details will be updated immediately upon publication).

---

## Abstract

Infrared and visible image fusion is a pivotal task in computer vision, aiming to integrate complementary modal characteristics to generate fused images suitable for both human visual observation and machine perception. However, most state-of-the-art fusion algorithms prioritise visual quality at the expense of utility for downstream high-level vision tasks, and existing semantic-driven methods face critical limitations: noise aliasing from coarse frequency-domain processing and gradient conflicts in multi-task joint training of fusion and segmentation. To address these challenges, this paper proposes ViR-MFS (Visible and Infrared Image Meta-learning Framework for Fusion and Segmentation), an end-to-end joint network for infrared-visible image fusion and segmentation based on alternating meta-learning optimisation and a multi-scale wavelet fusion module. First, a MixVisionTransformer is adopted as a shared backbone to construct a multi-scale feature pyramid that balances local spatial continuity and global semantic dependencies. Second, a Multi-scale Wavelet Fusion Module (MWFM) is designed to explicitly decouple high- and low-frequency components in the frequency domain; with learnable scale factors for adaptive recalibration, MWFM enables precise injection of visible texture details while effectively suppressing infrared thermal noise. To resolve objective conflicts in multi-task learning, an alternating meta-learning optimisation strategy is introduced, which dynamically coordinates the training of fusion and segmentation tasks to guide the backbone in learning robust features with both fine texture details and strong semantic discriminability, mitigating inter-task performance trade-offs. Here we show that extensive experiments on the MSRS and FMB datasets demonstrate ViR-MFS outperforms all state-of-the-art methods across key evaluation metrics: on the MSRS dataset, it achieves a mutual information (MI) of 5.212, edge fidelity ($Q^{AB/F}$) of 0.721, and a mean Intersection over Union (mIoU) of 58.96% for semantic segmentation; on the FMB dataset, it attains an MI of 4.956, $Q^{AB/F}$ of 0.730, and an mIoU of 64.15%. These results validate ViR-MFS’s superior performance in both visual fusion quality and semantic segmentation accuracy, highlighting its significant potential for all-weather complex scene perception applications in critical domains such as autonomous driving and video surveillance.

---

## Directory Structure

The project has been refactored for robust engineering and reproducibility. The core organization is as follows:

```text
Project_Root/
├── datasets/                   # Datasets directory (MSRS, FMB)
│   ├── FMB/
│   │   ├── ir/ (test/, train/)
│   │   ├── label/ (test/, train/)
│   │   └── vi/ (test/, train/)
│   └── MSRS/
│       ├── ir/ (test/, train/)
│       ├── label/ (test/, train/)
│       └── vi/ (test/, train/)
└── codes/                      # Main codebase
    ├── config/                 # Configuration files (YAML)
    │   ├── config.yaml         # Core paths and environment configurations
    │   └── params.yaml         # Hyperparameters for training and testing
    ├── data_pipeline/          # Data loading and processing
    │   └── dataloader.py
    ├── nets/                   # Network architectures
    │   ├── backbone.py         
    │   ├── segformer.py        
    │   ├── wavelet.py          
    │   └── wtconv2d.py         
    ├── utils/                  # Utility functions and loss metrics
    │   ├── common.py           
    │   ├── losses.py           
    │   ├── metrics.py          
    │   ├── utils_meta.py       
    │   └── utils_logger.py     
    ├── config_loader.py        # YAML configuration loader
    ├── train.py                # Main training script
    ├── test.py                 # Main testing and evaluation script
    ├── run_experiment.sh       # Unified execution script for training/testing
    ├── build_ViR_MFS_env.sh    # Environment setup script
    └── requirements.txt        # Python dependencies
```

**Note:** The following directories are generated during runtime and are ignored by version control:
* `model_data/`: Directory for pre-trained backbone weights.
* `runs/` & `runs_meta/`: Checkpoints and training logs.
* `test_results/`: Output directories for fused images and segmentation masks.

---

## Environment Setup

Please ensure you have Python 3.8+ and PyTorch 2.x (e.g., PyTorch 2.2.2+cu118 or PyTorch 2.3.0+cu118) installed matching your CUDA environment. 

Run the provided shell script to build the environment and install dependencies:

```bash
cd codes
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

2. Organize the downloaded datasets strictly matching the `datasets/` structure shown in the Directory Structure section above.

---

## Quick Start

We have eliminated hardcoded paths. All parameters and dataset paths are now centrally managed via YAML configuration files.

### 1. Configuration
Navigate to the `codes/config/` directory:
* **`config.yaml`**: Ensure the absolute or relative paths point correctly to your `datasets/` folder and `model_data/`.
* **`params.yaml`**: Configure training hyperparameters such as `batch_size`, `epochs`, `learning_rates`, and AMP settings (`use_amp`).

### 2. Training
Use the unified shell script to start training. The script automatically handles environment variables and memory optimization:

```bash
cd codes
# Usage: bash run_experiment.sh [GPU_ID] [MODE]
bash run_experiment.sh 0 train
```

### 3. Evaluation
To evaluate the model and generate fused images alongside segmentation masks, simply change the mode to `test`:

```bash
cd codes
bash run_experiment.sh 0 test
```
The outputs (fused images and predicted masks) will be automatically saved to the `test_results/` directory as specified in your `config.yaml`.

---

## Pre-trained Models

For quick inference and reproduction of the results reported in our paper, you can download our pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/11dXQ-pkYgPVe9qD4AXCpv-XIn5JZIMGh?usp=sharing).

Please place the downloaded `.pth` files into the `codes/model_data/` directory (create it if it does not exist) or specify the exact path in your `config.yaml`.

---

## Citation

If you find this code, our dataset processing, or our methodology useful in your research, please kindly consider citing our manuscript:

```bibtex
@article{ViRMFS2026,
  title={Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception},
  author={Yihui Wang and Dengshi Li and Shichao Liu and Shiwei Hu and Zhiming Zhan},
  journal={The Visual Computer},
  year={2026}
}
```

---

## Contact

If you have any questions about the code or paper, please feel free to open an issue or contact `wyh37133@gmail.com`.