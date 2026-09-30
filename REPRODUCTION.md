# Fast3R 复现记录

这份文档和 [`fast3r_reproduction.ipynb`](fast3r_reproduction.ipynb) 记录了本分支的可重复实验流程，适合第一次接触多视图 3D 重建的同学。

## 复现范围

本分支完成了公开权重的端到端推理、DTU 22 场景评测，并继续开展视角数与 local/global 点图消融。整篇论文尚未复现完成，逐项状态与实验条件见 [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md)。推理链路为：

```text
示例视频 → 抽帧 → 图片归一化 → Fast3R 一次多视图前向
         → 点图/置信度 → fast-PnP 相机位姿 → PLY 点云与可视化
```

数据集评测需要对应预处理数据。公开 HF 权重可以通过官方 `load_for_inference` 与指标实现评测；原始 `fast3r/eval.py` 则读取 Lightning checkpoint。当前尚未核实各论文实验权重与公开权重的对应关系，不能仅用格式差别解释数值差距。

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

## 论文级 DTU 重建指标

公开 Hugging Face 权重不是 Lightning `last.ckpt`，因此不能直接传给官方 `fast3r/eval.py`。本分支新增的 `scripts/fast3r_hf_dtu_eval.py` 保留 Hugging Face 权重格式，但直接复用 `MultiViewDUSt3RLitModule.evaluate_reconstruction` 及仓库中的 accuracy、completion、normal consistency 实现。

先做不加载模型的数据检查：

```bash
python scripts/fast3r_hf_dtu_eval.py \
  --device cpu \
  --dry-run \
  --max-scenes 2 \
  --output-json /tmp/fast3r_dtu_dry_run.json
```

在有 CUDA GPU 的机器上先跑一个场景确认显存和耗时：

```bash
python scripts/fast3r_hf_dtu_eval.py \
  --device cuda \
  --scenes scan12 \
  --output-json demo_outputs/paper_eval/dtu_scan12.json
```

确认单场景成功后运行全部 22 个 DTU 场景：

```bash
python scripts/fast3r_hf_dtu_eval.py \
  --device cuda \
  --output-json demo_outputs/paper_eval/dtu_all.json
```

脚本默认遵循仓库评测配置：512 分辨率、`kf_every=5`、局部头对齐置信度 85%、指标阶段置信度 0%。输出 JSON 包含每个场景和全体场景均值。它是“公开 HF checkpoint 的官方指标实现复现”，与原始训练 Lightning checkpoint 的严格复现实验应分别标注。

### 已完成的全量 DTU 运行

2026-09-30，在 NVIDIA GeForce RTX 5070 Ti Laptop GPU 上完成了公开 Hugging Face checkpoint 的全量 22 场景运行。运行配置为 CUDA、16-mixed、512 分辨率、每场景 10 个视角；原始逐场景结果见 [`demo_outputs/paper_eval/dtu_all.json`](demo_outputs/paper_eval/dtu_all.json)。下表是该 JSON 中 `aggregate_mean` 的 22 场景均值：

| 指标 | 22 场景均值 |
| --- | ---: |
| Accuracy | 5.2460 |
| Accuracy (median) | 2.9306 |
| Completion | 3.6110 |
| Completion (median) | 1.8200 |
| Normal consistency 1 | 0.6720 |
| Normal consistency 1 (median) | 0.7526 |
| Normal consistency 2 | 0.6364 |
| Normal consistency 2 (median) | 0.7041 |

这些是本分支使用公开 HF 推理 checkpoint、官方仓库指标实现得到的可复现实验结果；它们不等同于论文中使用原始 Lightning `last.ckpt` 的官方表格数值，也没有用论文参考值替代本地结果。

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

面试时可以说“完成公开权重推理、DTU 全量评测及推理消融”。应同时说明：Table 4 的 median 数值尚未对齐，其他数据集与重新训练实验未完成。见 Notebook 末尾与论文逐项对应表。

## 逐项论文实验与新结果

Notebook末尾已经添加独立可运行的“论文逐项核对”单元，包含损失数值/梯度检查（§3.2）、22场景固定种子DTU（§4.3/Table 4）、3/5/10/20视角与head消融（§5.1/Figure 5、§5.4/Table 5）和预热后的性能图（§4.1/Table 2的本机适配）。

- [`results/dtu_seed42_stride5.json`](results/dtu_seed42_stride5.json)：22场景，seed42、stride5，Accuracy/Completion median为2.0827/1.0311。
- [`results/dtu_paper_experiments.json`](results/dtu_paper_experiments.json)：88组视角实验，额外22组local/global配对指标。
- [`results/loss_checks.json`](results/loss_checks.json)：合成点图归一化、扰动和有限梯度检查。
- [`results/checkpoint_manifest.json`](results/checkpoint_manifest.json)：实际配置与权重的SHA256；原始下载revision未记录。
- 科学图：[`results/figures/`](results/figures/)。

Neural RGB-D官方9个序列已解压。`bash scripts/queue_nrgbd_reproduction.sh` 会先检查实际目录中的全部RGB/深度帧、非空文件与位姿数量；检查通过即跳过下载和解压，即使ZIP或manifest不存在。只有数据不完整时才恢复下载/解压；已有完整ZIP则复用，无需重下。可运行 `python scripts/check_nrgbd_data.py` 单独检查（这是文件布局/数量检查，不是重新解码PNG或CRC校验）。可选PID参数仅用于数据不完整时等待已有下载进程。后续评测入口支持 `python scripts/fast3r_hf_dtu_eval.py --dataset nrgbd --device cuda --output-json results/nrgbd_seed42_stride40.json`。默认stride40；Table 3比较时将native距离乘100。完整实验状态、限制与余下前提请查阅 [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md)。
