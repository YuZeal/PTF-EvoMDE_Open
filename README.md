<div align="center">


<h1>Efficient Evolutionary Architecture Search for Monocular Depth Estimation via Hierarchical Parameter Mapping<h1>

<p>
  <a href="https://ieeexplore.ieee.org/abstract/document/11178251"><img alt="IEEE TEVC" src="https://img.shields.io/badge/IEEE%20TEVC-2025-b31b1b.svg"></a>
  <a href="https://www.python.org/downloads/release/python-380/"><img alt="Python 3.8" src="https://img.shields.io/badge/Python-3.8-blue.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.1.2" src="https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg"></a>
  <a href="https://github.com/open-mmlab/mmdetection"><img alt="mmdetection 0.6.0" src="https://img.shields.io/badge/mmdetection-0.6.0-green.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p>
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

</div>

---

## Introduction

PTF-EvoMDE is an efficient NAS framework for monocular depth estimation that eliminates per-candidate ImageNet pretraining through **Hierarchical Parameter Mapping (HPM)** — dynamically transferring weights from a single MobileNetV2 template to candidate architectures across depth, channel, and kernel dimensions. Combined with a **Feature-Aligned CRFs (Fa-CRFs)** decoder that uses deformable convolutions to correct spatial misalignments, the resulting PTF-EvoMDENet achieves performance comparable to large-scale models with only ~6M parameters, while demonstrating strong transferability to detection, segmentation, and medical depth estimation tasks.

<div align="center">
  <img src="assets/framework.png" width="100%" />
</div>

---

## Key Features

- 🧬 **Template-Based Inverted Residual Search Space**: Modular search space built on inverted residual blocks, allowing the evolutionary algorithm to dynamically explore varying depths, expansion ratios, and kernel sizes.
- ⚡ **Hierarchical Parameter Mapping (HPM)**: Transfers weights from a single pretrained MobileNetV2 template (`seed_mbv2.pt`) to candidate architectures, reducing NAS overhead by over 75% — no per-candidate pretraining required.
- 🎯 **Spatial Feature Extraction Module (SFEM)**: Coordinate attention that selectively weights encoder features along vertical and horizontal dimensions for precise depth prediction.
- 🧠 **Feature-Aligned CRFs (Fa-CRFs) Decoder**: Deformable convolutions + multi-head window attention to align and refine upsampled features, preserving sharp object boundaries.
- 🌐 **Cross-Task Generalizability**: Transfers effectively from KITTI/NYU Depth v2 to object detection, semantic segmentation, and colonoscopy depth estimation.

---

## Installation

> [!WARNING]
> This codebase depends on a **locally installed fork** of MMDetection (`mmdetection-0.6.0`) and a compiled `DCNv2_latest` extension. These are **not** available via PyPI and must be built from source in the correct order. Ensure CUDA 11.8 toolkit headers are available before compiling custom CUDA ops.

```bash
# 1. Create and activate environment
conda create -n PTF-EvoMDE python=3.8 -y && conda activate PTF-EvoMDE

# 2. Install PyTorch (CUDA 11.8)
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install Python dependencies
pip install cython tqdm einops fvcore mmcv==0.2.10 timm==0.4.12

# 4. Build and install the bundled mmdetection-0.6.0
sh ./mmdet_build.sh

# 5. Build and install DCNv2 (deformable convolution kernel)
cd DCNv2_latest && python3 setup.py build develop && cd ..
```

---

## Data Preparation

Download the datasets and organize them as follows. Pre-defined train/test split files are provided in `data_splits/`.

```
data/
├── NYU_Depth_V2/
│   ├── sync/                      ← training RGB-D frames
│   └── test/                      ← test images & ground-truth
├── KITTI/
│   ├── raw/                       ← raw KITTI sequences (RGB images)
│   └── data_depth_annotated/      ← ground-truth velodyne depth maps
└── ColonoscopyDepth/              ← synthetic colonoscopy depth dataset
```

Update `--data_path` and `--gt_path` in the scripts under `scripts/` to match your local paths before running.

**Dataset Links:** [NYU Depth v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) · [KITTI Depth](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) · [Colonoscopy Depth](http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/)

---

## Quickstart

> [!NOTE]
> The search stage requires the MobileNetV2 HPM seed weights (`seed_mbv2.pt`) placed in the project root.

**NYU Depth v2 (example)**

```bash
# Stage 1 — Evolutionary Search (4 GPUs)
sh scripts/search/search_nyu.sh

# Stage 2 — Subnet Retraining (2 GPUs, arch string embedded in script)
sh scripts/retrain/retrain_nyu.sh

# Stage 3 — Evaluation
sh scripts/test/nyu_test.sh
```

<details>
<summary>Commands for KITTI and Colonoscopy</summary>

**KITTI (Eigen split)**
```bash
sh scripts/search/search_kitti.sh
sh scripts/retrain/retrain_kitti.sh
sh scripts/test/kitti_test.sh
```

**Colonoscopy Depth (Synthetic)**
```bash
sh scripts/search/search_colon.sh
sh scripts/retrain/retrain_med.sh
sh scripts/test/colon_test.sh
```

</details>

---

## Results

PTF-EvoMDENet achieves competitive depth estimation accuracy with only ~6M parameters, while retaining strong cross-task transferability.

**NYU Depth v2**

| Method | Params | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[PTF-EvoMDENet](https://github.com/sakura-yzh/PTF-EvoMDE_Open/releases/download/weight_files/nyu_best.pth)** | **6.13M** | **0.1108** | **0.0675** | **0.4003** | **0.1448** | **0.8769** | **0.9775** | **0.9947** |

**KITTI (Eigen split)**

| Method | Params | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[PTF-EvoMDENet](https://github.com/sakura-yzh/PTF-EvoMDE_Open/releases/download/weight_files/kitti_best.pth)** | **6.26M** | **0.0621** | **0.2173** | **2.5566** | **0.0954** | **0.9578** | **0.9943** | **0.9988** |

**Colonoscopy Depth (Synthetic)**

| Method | Mean L1-error ↓ | Mean Rel. L1-error ↓ | Mean RMSE ↓ | δ<0.5 ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **[PTF-EvoMDENet](https://github.com/sakura-yzh/PTF-EvoMDE_Open/releases/download/weight_files/colon_best.pth)** | **0.1397** | **0.0543** | **0.2306** | **0.8919** |

**NYU Depth v2 — Qualitative Comparison**

<div align="center">
  <img src="assets/nyu_compare.png" width="95%" />
</div>

---

## Citation
If you find this code useful in your research, please cite:
```bibtex
@article{zhang2025efficient,
  title={Efficient Evolutionary Neural Architecture Search With Hierarchical Parameter Mapping for Monocular Depth Estimation},
  author={Zhang, Haoyu and Yu, Zhihao and Jin, Yaochu and Liu, Xiufeng and Sheng, Weiguo and Liu, Ruyu and Li, Xiumei and Liu, Qiqi and Cheng, Ran},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2025},
  publisher={IEEE}
}
```

---

## Acknowledgements

PTF-EvoMDE builds on a strong ecosystem of open-source tools. We are grateful to the teams behind [PyTorch](https://pytorch.org/), [MMDetection](https://github.com/open-mmlab/mmdetection), [timm](https://github.com/huggingface/pytorch-image-models), [NewCRFs](https://github.com/aliyun/NeWCRFs), [FaPN](https://github.com/ShihuaHuang95/FaPN), and [DCNv2](https://github.com/CharlesShang/DCNv2) for making this work possible.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
