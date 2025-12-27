# MASR-Net: An Asymmetric Mixture-of-Attention based Sparse Restoration Network for Rectifying Visual Imbalance Defects

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📋 概述

MASR-Net (Asymmetric Mixture-of-Attention based Sparse Restoration Network) 是一个用于解决低光图像增强中视觉不平衡缺陷的深度学习网络。针对现有方法难以平衡全局一致性与局部细节恢复的问题，MASR-Net 通过以下创新设计实现了突破：

1. **区域自适应处理**：受大语言模型(LLMs)稀疏计算范式启发，将混合注意力机制(MoA)创新性地应用于低层视觉任务。MoA的非对称架构通过共享Key-Value投影结合稀疏Query-Output专家选择，在保持全局空间一致性的同时关注局部变化。

2. **物理属性解耦**：提出ISP引导编码器(ISP-Guided Encoder)，通过注意力路由将特征隐式解耦到基于物理的分支(Gamma、Color、Saturation)，使专用专家能够协同处理空间非均匀降质。

3. **层次化MASR模块**：构建结合Mixture-of-Attention和Mixture-of-Experts的层次化MASR Block，堆叠多层以处理特征级差异以及空间变化和相关性。

4. **参数图生成**：Map Generator通过双路径融合生成像素级ISP参数图，消除棋盘伪影。

![MASR-Net Architecture](MASR-Net.pdf)

在多个基准数据集(LOL-v1/v2、LSRW)和地下矿山数据集(CMUPD)上的广泛实验表明，MASR-Net达到了最先进的性能，PSNR高达28.71 dB，SSIM高达0.860，同时保持了具有竞争力的效率(7.18G FLOPs)。

## 🌟 主要特性

- **ISP语义-内容解耦机制**: 创新的"指引-内容"解耦编码，实现选择与处理的深度整合
- **分离式MoE**: 为不同ISP方法创建专用信息通道，避免信息混淆
- **高效架构**: 仅6.58M参数即可达到PSNR=23.2, SSIM=0.86的优秀性能
- **端到端训练**: 支持完整的训练和推理流程
- **多数据集支持**: 支持LOL-v1、LOL-v2、LSRW等主流低光增强数据集

## 📊 性能表现

| 数据集 | PSNR | SSIM | FLOPs |
|--------|------|------|-------|
| LOL-v1      | 28.71 dB | 0.860 | 7.18G |
| LOL-v2-real | 23.2 dB  | 0.86  | 7.18G |
| LSRW        | 22.8 dB  | 0.69  | 7.18G |
| CMUPD       | -        | -     | 7.18G |

MASR-Net在保持高效计算(7.18G FLOPs)的同时达到了最先进的性能指标。



## 🚀 快速开始

### 环境要求

**硬件要求**：
- GPU: NVIDIA RTX 5090 (推荐) 或其他支持CUDA的GPU
- 显存: 建议16GB+

**软件要求**：
- Python版本: Python 3.9+
- CUDA: 12.8 (与PyTorch版本匹配)

**安装依赖**：

推荐使用 `requirements.txt` 安装所有依赖：
```bash
pip install -r requirements.txt
```

或手动安装核心依赖：
```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128
pip install numpy matplotlib scikit-image tqdm pillow kornia optuna rich pandas
```

### 数据集准备

项目支持以下数据集结构：

```
datasets/
├── LOL_V1/
│   └── lol_dataset/
│       ├── Train/
│       ├── Test/
│       └── Val/
├── LOL_v2/
│   ├── Train/
│   ├── Test/
│   └── Val/
└── OpenDataLab___LSRW/
    └── raw/LSRW/
```

将数据集下载后放置在对应目录即可。

### ⚙️ 重要配置说明

**针对不同数据集的关键配置参数：**

| 数据集 | patch_size | win_size | 说明 |
|--------|------------|----------|------|
| LOL-v1 | 4          | 2        | 适用于LOL-v1数据集的配置 |
| LSRW   | 4          | 2        | 适用于LSRW数据集的配置   |
| LOL-v2 | 2          | 4        | 适用于LOL-v2数据集的配置 |

**配置修改方法：**
1. 在 `config.py` 文件中修改对应参数：
   ```python
   # LOL-v1和LSRW数据集
   'patch_size': 4,
   'win_size': 2,

   # LOL-v2数据集
   'patch_size': 2,
   'win_size': 4,
   ```

2. **如果需要更改输入输出图像尺寸**，需要同时修改：
   - `config.py` 中的 `input_size` 和 `output_size` 参数
   - `data_loader.py` 中对应的图像预处理尺寸设置

⚠️ **注意**: 不同的patch_size和win_size组合会影响模型的窗口分割和特征提取策略，务必根据所使用的数据集选择正确的配置。

### 预训练模型

将预训练权重文件放置在 `checkpoints/` 目录下：
- `LOLv1_checkpoints.pth` - LOL-v1数据集训练的模型
- `LOLv2_real_checkpoints.pth` - LOL-v2-real数据集训练的模型
- `LSRW_checkpoints.pth` - LSRW数据集训练的模型

## 📖 使用说明

### 训练模型

#### 基础训练
```bash
# 在LOL-v2数据集上训练
python train.py --data_dir ./datasets/LOL_v2 --epochs 1200 --batch_size 4

# 自定义参数训练
python train.py \
    --data_dir ./datasets/LOL_V1/lol_dataset \
    --epochs 800 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_dir ./checkpoints
```


### 测试模型

#### 在测试集上评估
```bash
# 使用LOL-v1模型测试
python test.py \
    --data_dir ./datasets/LOL_V1/lol_dataset \
    --weights_path ./checkpoints/LOLv1_checkpoints.pth \
    --dataset_split test

# 使用LOL-v2模型测试
python test.py \
    --data_dir ./datasets/LOL_v2 \
    --weights_path ./checkpoints/LOLv2_real_checkpoints.pth \
    --dataset_split test
```

测试结果会自动保存在 `./result/{dataset_type}/` 目录下，包括：
- 三图对比效果图 (`comparison_XXXX.png`)
- 增强后的单独图像 (`enhanced_XXXX.png`)
- 测试指标报告 (`test_results.txt`, `test_results.json`)

## 🖼️ 效果展示

`demo/` 目录下的对比图展示了MASR-Net在不同场景下的增强效果对比：

- **comparison_1.png**: 室内场景低光增强效果
- **comparison_2.png**: 户外夜景增强对比
- **comparison_3.png**: 复杂光照条件处理
- **comparison_4.png**: 细节保持和噪声抑制
- **comparison_5.png**: 颜色还原准确性
- **comparison_6.png**: 高对比度场景处理

每张对比图包含三部分：原始低光图像、MASR-Net增强结果、参考真值图像，并显示PSNR和SSIM指标。

## 📁 项目结构

```
masenet/
├── README.md              # 项目说明文档
├── config.py              # 模型配置文件
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── search.py              # 架构搜索脚本
├── models.py              # 主模型定义
├── MoA.py                 # MoA和MoE模块实现
├── feature_extractor.py   # 特征提取器
├── ISP.py                 # ISP操作模块
├── decoder.py             # ISP参数解码器
├── data_loader.py         # 数据加载器
├── data_augmentation.py   # 数据增强工具
├── losses.py              # 损失函数定义
├── utils.py               # 工具函数
├── emb_gen.py             # 嵌入生成器
├── checkpoints/           # 预训练模型目录
├── datasets/              # 数据集目录
└── demo/                  # 效果展示图片目录
    └── comparison_*.png   # 效果对比图
```
### 损失函数

组合损失包含：
- L1重建损失
- 感知损失(VGG特征)
- SSIM结构相似性损失
- PSNR优化损失
- LAB色彩空间损失
- 辅助正则化损失(MoE负载均衡)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📚 引用

如果您在研究中使用了MASR-Net，请考虑引用：

```bibtex
@misc{masrnet2025,
  title={MASR-Net: An Asymmetric Mixture-of-Attention based Sparse Restoration Network for Rectifying Visual Imbalance Defects},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/Britark/MASR-Net}}
}
```

## 🙏 致谢

感谢以下开源项目和数据集：
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/)
- [LSRW Dataset](https://github.com/JianghaiSCU/R2RNet)
- [Optuna](https://optuna.org/)

---

**联系方式**: 如有问题请提交Issue或发送邮件至 britarklxt@gmail.com
