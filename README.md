# ViR-MFS: Wavelet-Driven Meta-Learning for Infrared-Visible Fusion and Segmentation

[![DOI](https://zenodo.org/badge/1164573018.svg)](https://doi.org/10.5281/zenodo.18740397)

Official PyTorch implementation of the paper: **"Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception"** (Currently under review / Submitted to *The Visual Computer*).

> **📌 Important Note for Readers:** > If you find this code, our dataset processing, or our methodology useful in your research, please kindly consider citing our manuscript submitted to *The Visual Computer*. (Citation details will be updated immediately upon publication).

---

## 📖 Abstract

Infrared and visible image fusion is a pivotal task in computer vision, aiming to integrate complementary modal characteristics to generate fused images suitable for both human visual observation and machine perception. However, most state-of-the-art fusion algorithms prioritise visual quality at the expense of utility for downstream high-level vision tasks, and existing semantic-driven methods face critical limitations: noise aliasing from coarse frequency-domain processing and gradient conflicts in multi-task joint training of fusion and segmentation. To address these challenges, this paper proposes ViR-MFS (Visible and Infrared Image Meta-learning Framework for Fusion and Segmentation), an end-to-end joint network for infrared-visible image fusion and segmentation based on alternating meta-learning optimisation and a multi-scale wavelet fusion module. First, a MixVisionTransformer is adopted as a shared backbone to construct a multi-scale feature pyramid that balances local spatial continuity and global semantic dependencies. Second, a Multi-scale Wavelet Fusion Module (MWFM) is designed to explicitly decouple high- and low-frequency components in the frequency domain; with learnable scale factors for adaptive recalibration, MWFM enables precise injection of visible texture details while effectively suppressing infrared thermal noise. To resolve objective conflicts in multi-task learning, an alternating meta-learning optimisation strategy is introduced, which dynamically coordinates the training of fusion and segmentation tasks to guide the backbone in learning robust features with both fine texture details and strong semantic discriminability, mitigating inter-task performance trade-offs. Here we show that extensive experiments on the MSRS and FMB datasets demonstrate ViR-MFS outperforms all state-of-the-art methods across key evaluation metrics: on the MSRS dataset, it achieves a mutual information (MI) of 5.212, edge fidelity ($Q^{AB/F}$) of 0.721, and a mean Intersection over Union (mIoU) of 58.96% for semantic segmentation; on the FMB dataset, it attains an MI of 4.956, $Q^{AB/F}$ of 0.730, and an mIoU of 64.15%. These results validate ViR-MFS’s superior performance in both visual fusion quality and semantic segmentation accuracy, highlighting its significant potential for all-weather complex scene perception applications in critical domains such as autonomous driving and video surveillance.

---

## 🚀 Environment Setup

Please ensure you have Python 3.8+ and PyTorch installed. We recommend using Conda to manage your environment.

```bash
# Clone the repository
git clone https://github.com/[你的GitHub用户名]/ViR-MFS.git
cd ViR-MFS

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

We evaluate our method on the **MSRS** and **FMB** datasets. 

1. Download the datasets from the following links:
   - [MSRS Dataset Link]([在这里填入你准备好的 MSRS 网盘链接])
   - [FMB Dataset Link]([在这里填入你准备好的 FMB 网盘链接])
   
   *(Note: The original copyrights of the datasets belong to their respective authors. We provide these links solely to facilitate reproducibility.)*

2. Organize the downloaded datasets in the `datasets/` directory as follows:

```text
ViR-MFS/
├── datasets/
│   ├── MSRS/
│   │   ├── Infrared/
│   │   ├── Visible/
│   │   └── Labels/
│   └── FMB/
│       ├── Infrared/
│       ├── Visible/
│       └── Labels/
```

---

## ⚙️ Quick Start

### Training
To train the ViR-MFS model from scratch using the proposed alternating meta-learning optimization strategy, please run:

```bash
python train.py --dataset MSRS --batch_size 8 --epochs 300
```
*(Modify the `--dataset` and other hyper-parameters as needed according to your local environment.)*

### Evaluation
To evaluate the fusion and segmentation performance using our pre-trained weights, please run:

```bash
python test.py --dataset MSRS --checkpoint_path ./weights/best_model.pth
```

---

## 🔗 Pre-trained Models

For quick inference and reproduction of the results reported in our paper, you can download our pre-trained weights from [Google Drive]([在这里填入你按照流程生成的 Google Drive 分享链接]) and place them in the `./weights` folder.

---

## 📝 Citation

If you find this code, our dataset processing, or our methodology useful in your research, please kindly consider citing our manuscript submitted to *The Visual Computer*:

```bibtex
@article{ViRMFS2026,
  title={Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception},
  author={[你的全拼姓名, 例如 San Zhang] and [合著者1全拼] and [合著者2全拼]},
  journal={The Visual Computer},
  year={2026}
}
```

---

## 📧 Contact

If you have any questions about the code or paper, please feel free to open an issue or contact `[你的电子邮箱地址]`.
