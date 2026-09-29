# Fast3R 复现记录

这份文档和 [`fast3r_reproduction.ipynb`](fast3r_reproduction.ipynb) 记录了本分支的可重复实验流程，适合第一次接触多视图 3D 重建的同学。

## 复现范围

本分支完成的是官方模型的功能性复现：

```text
示例视频 → 抽帧 → 图片归一化 → Fast3R 一次多视图前向
         → 点图/置信度 → fast-PnP 相机位姿 → PLY 点云与可视化
```

论文中的 DTU、7-Scenes、Neural-RGBD 等数值指标属于另一层工作，需要对应的预处理数据和 Lightning `last.ckpt`。Hugging Face 的 `model.safetensors` 可以直接做 Demo/推理，但不能直接传给 `fast3r/eval.py` 作为 Lightning checkpoint；Notebook 会显式检查这一点。

## 环境准备

建议使用 Python 3.11 和显存至少 12 GB 的 CUDA GPU。模型目录约 2.6 GB，运行 12 个 512 长边视角时还需要额外显存。

```bash
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r

# 根据本机 CUDA 版本安装 PyTorch；官方示例使用 CUDA 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 \
  nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia

pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name fast3r --display-name "fast3r"
```

不要安装 DUSt3R 的 cuROPE 扩展；Fast3R 的 README 明确说明它可能导致预测错误。网络不稳定时，先把 Hugging Face checkpoint 下载到本地，再让 Notebook 使用 `checkpoints/Fast3R_ViT_Large_512/`。

## 运行顺序

1. 在仓库根目录打开 `fast3r_reproduction.ipynb`，选择 `fast3r` kernel。
2. 从上到下运行环境检查和图片准备单元。没有现成图片时，Notebook 会从 `demo_examples/family/Family.mp4` 自动抽帧。
3. 运行主线推理单元，生成点图、置信度和相机位姿。
4. 运行模块级验证、视角规模实验和可视化单元。
5. 查看导出的 `demo_outputs/notebook_inference/`：

   - `reconstruction_results.npz`：点图、置信度、相机位姿、估计焦距和输入路径；
   - `reconstruction_pointcloud.ply`：经过置信度百分位筛选的彩色点云；
   - `reproducibility_card.json`：设备、精度、视角数、耗时和产物路径。

6. DTU 单元默认只检查数据和 checkpoint 是否存在，不会自动下载数 GB 数据。需要下载时，显式把 `ALLOW_DATA_DOWNLOAD` 改成 `True`。

## 已记录的本地结果

在 `fast3r` 环境、NVIDIA GeForce RTX 5070 Ti Laptop GPU（11.5 GB）上，12 个视频视角的历史运行记录为：

| 项目 | 结果 |
| --- | ---: |
| 输入视角数 | 12 |
| 预处理分辨率 | 512 × 288 |
| Fast3R 前向时间 | 约 1.20 s |
| 12 视角峰值显存 | 约 6.13 GB |
| PLY 点数（置信度筛选后） | 973,464 |

这些数字是本机功能性复现记录，不是论文 benchmark。当前机器没有 CUDA 时，Notebook 会在模型加载前给出明确提示；环境检查、数据准备和结果可视化仍可运行。

## 简历表述

> **Fast3R 多视图 3D 重建复现（PyTorch / CUDA）**：基于官方 ViT-L/512 checkpoint，搭建从视频抽帧、图像归一化、多视图一次前向到 fast-PnP 相机估计的端到端推理流程；实现置信度热力图、相机轨迹和 RGB 点云可视化，并导出 `.ply` / `.npz` 复现产物。通过 forward hook 检查 patch embedding、encoder、decoder、global/local head 的输出形状与数值有效性，记录不同视角规模下的耗时和显存。

面试时应明确说“完成官方模型的功能性复现”；只有准备好 Lightning checkpoint 和官方数据后，才应进一步声称复现论文指标。

