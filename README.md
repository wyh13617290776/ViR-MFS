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
git clone https://github.com/wyh13617290776/ViR-MFS.git
cd ViR-MFS

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

We evaluate our method on the **MSRS** and **FMB** datasets. 

1. Download the datasets from the following links:
   - [MSRS Dataset Link](https://github.com/Linfeng-Tang/MSRS)
   - [FMB Dataset Link](https://github.com/JinyuanLiu-CV/SegMiF)
   
   *(Note: The original copyrights of the datasets belong to their respective authors. We provide these links solely to facilitate reproducibility.)*

2. Organize the downloaded datasets in the `datasets/` directory strictly as follows to match the data loading logic:

```text
ViR-MFS/
├── datasets/
│   ├── MSRS/
│   │   ├── Infrared/
│   │   │   ├── train/MSRS/
│   │   │   └── test/MSRS/
│   │   ├── Visible/
│   │   │   ├── train/MSRS/
│   │   │   └── test/MSRS/
│   │   └── Label/
│   │       ├── train/MSRS/
│   │       └── test/MSRS/
│   └── FMB/
│       ├── ir/
│       │   ├── train/
│       │   └── test/
│       ├── vi/
│       │   ├── train/
│       │   └── test/
│       └── Label/
│           ├── train/
│           └── test/
```
---

## ⚙️ Quick Start

### 1. Configure Dataset Paths
Since our data loading strategy currently uses absolute paths, please open `dataloder.py` and modify the paths to match the location of your downloaded datasets before running the code.

For example, update the `vi_dir`, `ir_dir`, and `label_dir` in both the `vifs_dataloder` (for training) and `vifs_dataloder_test` (for testing) classes:
```python
# In dataloder.py
self.vi_dir = os.path.join(rf'/your_local_path/MSRS/Visible/{task}/MSRS')
self.ir_dir = os.path.join(rf'/your_local_path/MSRS/Infrared/{task}/MSRS')
self.label_dir = os.path.join(rf'/your_local_path/MSRS/Label/{task}/MSRS')
```

### 2. Training
Once the dataset paths are correctly configured, you can start training the ViR-MFS model from scratch by running:

```bash
python train.py --batch_size 8 --epochs 300
```
*(Note: You can adjust the batch size and epochs in the command line or directly within `train.py` depending on your GPU memory capacity.)*

### 3. Evaluation
Since the evaluation parameters are currently hardcoded, please open `test.py` and navigate to the `if __name__ == '__main__':` block at the bottom of the file. Update the parameters such as `model_path`, `save_dir`, and `num_classes` according to the dataset you want to test:

```python
# In test.py
if __name__ == '__main__':
    test_model(
        model_path='./weights/FMB_pth/fmb_b0.pth',  # Path to your downloaded/trained weights
        batch_size=1,
        save_dir='test_results',                # Output directory for fused images and seg maps
        num_classes=17,                         # 9 for MSRS, 17 for FMB
        use_dataparallel=True                   # Set to False if using a single GPU
    )
```

After modifying the parameters and ensuring the test data paths in `dataloder.py` are correct, run the evaluation script:

```bash
python test.py
```

---

## 🔗 Pre-trained Models

For quick inference and reproduction of the results reported in our paper, you can download our pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/11dXQ-pkYgPVe9qD4AXCpv-XIn5JZIMGh?usp=sharing) and place them in the `./weights` folder.

---

## 📝 Citation

If you find this code, our dataset processing, or our methodology useful in your research, please kindly consider citing our manuscript submitted to *The Visual Computer*:

```bibtex
@article{ViRMFS2026,
  title={Wavelet-Driven Meta-Learning: Unifying Infrared-Visible Fusion and Semantic Segmentation for Robust Scene Perception},
  author={Yihui Wang and Dengshi Li and Shichao Liu and Shiwei Hu and Zhiming Zhan},
  journal={The Visual Computer},
  year={2026}
}
```

---

## 📧 Contact

If you have any questions about the code or paper, please feel free to open an issue or contact `wyh37133@gmail.com`.
