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

面试时可以说“完成公开权重推理、DTU 22场景、Neural RGB-D 9场景和7-Scenes全部18测试轨迹评测及推理消融”。应同时说明：论文指标尚未匹配，CO3D/RE10K位姿评测与重新训练实验未完成。见 Notebook 末尾与论文逐项对应表。

## 逐项论文实验与新结果

步骤7（2026-10-02，§4.2/Table1）：扩容后恢复数据准备，官方RE10K1832相机TXT全部核验；
test-only RGB归档55.6GB后台下载中。首chunk17个clip全量图像解码360×640，8个规定ID相机/时间戳
与官方GT吻合，仅是format probe。新HF单卡入口用真实1个scene/10视角完成接线测试，PnP0失败，
RRA@15=1.0、RTA@15=0.0889、mAA30=0.1627；这是smoke而不是正式Table1成绩。
断点再运行的JSON与首次结果完全相同，47项离线测试通过。Notebook已记录数据准备与smoke分析，
不覆盖DTU/NRGBD/7-Scenes历史结果。下一步核验RGB全量覆盖、GT与裁剪协议后运行完整1832集合。
续接任务每30分钟回到本聊天，配合[磁盘检查点](CONTINUATION.md)和独立后台下载；
额度不足也会影响定时任务，不能保证精确重置时刻自动恢复，需要开机与应用运行。

最新新增步骤6（§3.2、§4.2–4.3、§5.4）：[`PROTOCOL_AUDIT.md`](PROTOCOL_AUDIT.md)与[`results/protocol_audit_20261001.json`](results/protocol_audit_20261001.json)核实本地权重/配置匹配当前HF公开revision，但未证明各论文实验的checkpoint映射。记录RoMa对应点注册/ICP文字与置信度项符号差异，不武断归因。Notebook新增独立审计分析单元并保存真实输出。

`python scripts/check_pose_data.py`只读检查规定测试清单及RGB/相机GT，10项合成测试通过；缺失或损坏返回incomplete，CLI退出码2。2026-10-01的CO3D清单缺失、RE10K RGB未准备、空闲2.38GiB是历史检查点。2026-10-02扩容后初查空闲122G，官方1832个测试相机TXT已齐备；作者test-only RGB归档正在独立后台下载，覆盖/GT等价性仍待核验。数据完整性不等于论文协议一致，预检查不计作Table 1指标。新单卡HF位姿入口具备固定采样与断点恢复，但尚无正式Table1分数；参见下述步骤7与[续接检查点](CONTINUATION.md)。

最新完成阶段：7-Scenes全部7类场景、18条TestSplit轨迹、850个stride20视角的同次前向local/global评测已正常结束（2026-10-01 12:44:47，Asia/Shanghai）。`python scripts/validate_7scenes_report.py`核验完整集合、视角/seed、全部有限aggregate与配对标记。Table3逐轨迹median平均×100为3.3035/3.1058，论文参考1.58/0.93，误差高109.08%/233.95%；Table5 mean×100为local6.3613/7.0516、global6.9567/6.3510。local降低Accuracy但未改善Completion，不能宣称达到论文成绩或全面local优势。NRGBD配对结果与历史JSON保留不变。

后续执行入口（2026-10-01）：`--head both` 在同一次预测上报告local/global配对指标；NRGBD输出 `results/nrgbd_paired_seed42_stride40.json`。`bash scripts/queue_7scenes_reproduction.sh` 准备官方7-Scenes TestSplit的stride20采样帧、注册深度，等待NRGBD消融结束及GPU空闲后输出 `results/7scenes_paired_seed42_stride20.json`。只有完整结果产生并通过Notebook校验才报告成绩。空间节省仅改变存储方式，不改变该stride的实际输入/GT；不能用该稀疏目录训练或评测其他stride。

Notebook末尾已经添加独立可运行的“论文逐项核对”单元，包含损失数值/梯度检查（§3.2）、22场景固定种子DTU（§4.3/Table 4）、3/5/10/20视角与head消融（§5.1/Figure 5、§5.4/Table 5）和预热后的性能图（§4.1/Table 2的本机适配）。

- [`results/dtu_seed42_stride5.json`](results/dtu_seed42_stride5.json)：22场景，seed42、stride5，Accuracy/Completion median为2.0827/1.0311。
- [`results/dtu_paper_experiments.json`](results/dtu_paper_experiments.json)：88组视角实验，额外22组local/global配对指标。
- [`results/loss_checks.json`](results/loss_checks.json)：合成点图归一化、扰动和有限梯度检查。
- [`results/checkpoint_manifest.json`](results/checkpoint_manifest.json)：实际配置与权重的SHA256；原始下载revision未记录。
- [`results/nrgbd_seed42_stride40.json`](results/nrgbd_seed42_stride40.json)：Neural RGB-D完整9场景、278视角，seed42/stride40/chunk2；Table 3口径（逐场景median平均×100）Accuracy/Completion为4.0165/1.2001，论文参考3.40/1.01，尚未匹配。Notebook已核验场景、aggregate并保存真实分析输出；公开权重与论文具体权重对应关系仍未验证。
- [`results/7scenes_paired_seed42_stride20.json`](results/7scenes_paired_seed42_stride20.json)：官方全部18条测试轨迹、850视角，完整同次前向双分支结果；Table3 median与Table5 mean分开分析。数据深度由固定SimpleRecon depth-to-RGB投影注册，采样保留原始编号。Notebook保存真实结果表与逐轨迹科学图，不覆盖历史NRGBD/DTU报告。
- 科学图：[`results/figures/`](results/figures/)。

Neural RGB-D官方9个序列已解压。`bash scripts/queue_nrgbd_reproduction.sh` 会先检查实际目录中的全部RGB/深度帧、非空文件与位姿数量；检查通过即跳过下载和解压，即使ZIP或manifest不存在。只有数据不完整时才恢复下载/解压；已有完整ZIP则复用，无需重下。可运行 `python scripts/check_nrgbd_data.py` 单独检查（这是文件布局/数量检查，不是重新解码PNG或CRC校验）。可选PID参数仅用于数据不完整时等待已有下载进程。后续评测入口支持 `python scripts/fast3r_hf_dtu_eval.py --dataset nrgbd --device cuda --output-json results/nrgbd_seed42_stride40.json`。默认stride40；Table 3比较时将native距离乘100。完整实验状态、限制与余下前提请查阅 [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md)。
