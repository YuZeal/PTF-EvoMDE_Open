<div align="center">

<h1>基于层次参数映射的单目深度估计高效进化神经架构搜索<h1>

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

## 简介

PTF-EvoMDE 是一个面向单目深度估计的高效神经架构搜索框架，其核心创新**层次参数映射（HPM）**通过在深度、通道、卷积核三个维度上将单一 MobileNetV2 模板网络的预训练权重动态迁移至候选架构，彻底消除了逐候选网络独立预训练的需求。配合以可形变卷积修正空间偏移的**特征对齐条件随机场（Fa-CRFs）解码器**，最终搜索得到的 PTF-EvoMDENet 仅凭约 6M 参数即可达到大型模型的精度水平，并在目标检测、语义分割和医学深度估计任务上展现出强大的迁移能力。

<div align="center">
  <img src="assets/framework.png" width="100%" />
</div>

---

## 核心特性

- 🧬 **基于模板的倒置残差搜索空间**：以倒置残差块为基本单元，进化算法可动态探索不同深度、扩展比和卷积核尺寸的组合。
- ⚡ **层次参数映射（HPM）**：从单一 MobileNetV2 模板（`seed_mbv2.pt`）跨维度迁移权重至候选架构，NAS 计算开销降低 75% 以上，无需逐候选网络预训练。
- 🎯 **空间特征提取模块（SFEM）**：利用坐标注意力机制沿水平/垂直方向选择性加权编码器特征，有效抑制冗余信息。
- 🧠 **特征对齐条件随机场（Fa-CRFs）解码器**：可形变卷积 + 多头窗口注意力，动态对齐上采样特征与高分辨率编码器输出，保留清晰的物体边界。
- 🌐 **跨任务泛化能力**：从 KITTI/NYU Depth v2 有效迁移至目标检测、语义分割及结肠镜合成深度估计任务。

---

## 安装

> [!WARNING]
> 本代码库依赖**本地安装**的 MMDetection 定制版（`mmdetection-0.6.0`）和编译好的 `DCNv2_latest` 扩展，两者**均无法通过 PyPI 获取**，必须按顺序从源码构建。编译自定义 CUDA 算子前，请确保已安装 CUDA 11.8 工具链头文件。

```bash
# 1. 创建并激活虚拟环境
conda create -n PTF-EvoMDE python=3.8 -y && conda activate PTF-EvoMDE

# 2. 安装 PyTorch（CUDA 11.8）
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装 Python 依赖
pip install cython tqdm einops fvcore mmcv==0.2.10 timm==0.4.12

# 4. 构建并安装 mmdetection-0.6.0（本地版本）
sh ./mmdet_build.sh

# 5. 构建并安装 DCNv2（可形变卷积 CUDA 核）
cd DCNv2_latest && python3 setup.py build develop && cd ..
```

---

## 数据准备

下载数据集并按如下结构组织，`data_splits/` 目录下已提供官方训练/测试划分文件。

```
data/
├── NYU_Depth_V2/
│   ├── sync/                      ← 训练 RGB-D 帧
│   └── test/                      ← 测试图像及真值深度图
├── KITTI/
│   ├── raw/                       ← 原始 KITTI 序列（RGB 图像）
│   └── data_depth_annotated/      ← 激光雷达真值深度图
└── ColonoscopyDepth/              ← 合成结肠镜深度数据集
```

运行脚本前，请将 `scripts/` 下对应 shell 文件中的 `--data_path` 和 `--gt_path` 修改为本地数据集路径。

**数据集链接：** [NYU Depth v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) · [KITTI 深度预测](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) · [结肠镜深度（合成）](http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/)

---

## 快速开始

> [!NOTE]
> 搜索阶段需将 MobileNetV2 HPM 种子权重文件 `seed_mbv2.pt` 放置于项目根目录。

**NYU Depth v2（示例）**

```bash
# 阶段一：进化搜索（4 GPU）
sh scripts/search/search_nyu.sh

# 阶段二：子网重训练（2 GPU，架构字符串已内嵌于脚本）
sh scripts/retrain/retrain_nyu.sh

# 阶段三：评估
sh scripts/test/nyu_test.sh
```

<details>
<summary>KITTI 与结肠镜数据集运行命令</summary>

**KITTI（Eigen 划分）**
```bash
sh scripts/search/search_kitti.sh
sh scripts/retrain/retrain_kitti.sh
sh scripts/test/kitti_test.sh
```

**结肠镜深度（合成）**
```bash
sh scripts/search/search_colon.sh
sh scripts/retrain/retrain_med.sh
sh scripts/test/colon_test.sh
```

</details>

---

## 实验结果

PTF-EvoMDENet 仅凭约 6M 参数，实现了具有竞争力的深度估计精度，同时保持了强大的跨任务迁移能力。

**NYU Depth v2**

| 方法 | 参数量 | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[PTF-EvoMDENet](https://github.com/sakura-yzh/PTF-EvoMDE_Open/releases/download/weight_files/nyu_best.pth)** | **6.13M** | **0.1108** | **0.0675** | **0.4003** | **0.1448** | **0.8769** | **0.9775** | **0.9947** |

**KITTI（Eigen 划分）**

| 方法 | 参数量 | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[PTF-EvoMDENet](https://github.com/sakura-yzh/PTF-EvoMDE_Open/releases/download/weight_files/kitti_best.pth)** | **6.26M** | **0.0621** | **0.2173** | **2.5566** | **0.0954** | **0.9578** | **0.9943** | **0.9988** |

**结肠镜深度（合成）**

| 方法 | Mean L1-error ↓ | Mean Rel. L1-error ↓ | Mean RMSE ↓ | δ<0.5 ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **[PTF-EvoMDENet](https://github.com/sakura-yzh/PTF-EvoMDE_Open/releases/download/weight_files/colon_best.pth)** | **0.1397** | **0.0543** | **0.2306** | **0.8919** |

**NYU Depth v2 — 定性对比**

<div align="center">
  <img src="assets/nyu_compare.png" width="95%" />
</div>

---

## 引用
如果你认为这个代码对你的研究有帮助，请引用：
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

## 致谢

PTF-EvoMDE 的实现受益于以下优秀的开源工作。向 [PyTorch](https://pytorch.org/)、[MMDetection](https://github.com/open-mmlab/mmdetection)、[timm](https://github.com/huggingface/pytorch-image-models)、[NewCRFs](https://github.com/aliyun/NeWCRFs)、[FaPN](https://github.com/ShihuaHuang95/FaPN) 和 [DCNv2](https://github.com/CharlesShang/DCNv2) 的贡献者们致以诚挚感谢。

---

## 开源协议

本项目遵循 **MIT License** — 详见 [LICENSE](LICENSE) 文件。
