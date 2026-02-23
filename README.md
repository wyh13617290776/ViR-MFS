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
git clone [https://github.com/](https://github.com/)[你的GitHub用户名]/ViR-MFS.git
cd ViR-MFS

# Install dependencies
pip install -r requirements.txt

---

