# Fast3R：逐项复现与论文对应表

核对版本：[arXiv v2，2025-03-19](https://arxiv.org/html/2501.13928v2)。本记录更新于 2026-10-01。

## 完成标准

当前完成了公开权重的推理链路、DTU 22 场景和 Neural RGB-D 9 场景指标运行；整篇论文复现尚未完成。运行结束只说明实验产生了结果，还需要核对指标、采样协议、模型版本与论文数值的差距。以下区分实测、代码核对、缺数据与缺训练算力。

| 论文位置 | 要验证的问题 | 本项目证据 | 状态 |
| --- | --- | --- | --- |
| §3.1；§3.3；Figure 2 | 多张图是否一次输出 global/local 点图与置信度？ | Notebook 的推理与模块 hook；`config.json` | 推理验证已执行 |
| §3.2，Eq. (1)–(3) | 归一化点图回归、置信度加权损失 | `results/loss_checks.json` | 数值/梯度检查通过；尚未重新训练 |
| §3.4；§4.1；Table 2 | 视角数增加时耗时与显存如何变化？ | `results/dtu_paper_experiments.json` 的 performance | 本机适配实验；单卡不覆盖论文 A100/多卡设置 |
| §4.2；Table 1 | CO3D / RealEstate10K 的 RRA、RTA、mAA | 官方 pose 配置与脚本 | 缺对应测试集；PnP Demo 不能代替这些指标 |
| §4.3；Table 3 | 7-Scenes / NRGBD 重建 | `results/nrgbd_seed42_stride40.json`；NRGBD完整9场景；7-Scenes准备/排队服务 | NRGBD已运行，未对齐论文数值；7-Scenes准备与全量评测尚在进行 |
| §4.3；Table 4 | DTU 完整 22 场景重建 | `demo_outputs/paper_eval/dtu_all.json` | 已运行；论文数值尚未对齐 |
| §5.1；Figure 5 | 测试视角数对重建质量的影响 | 新脚本 3/5/10/20 视角 | 本地均匀采样适配实验 |
| §5.1；Figures 6–7 | 不同训练视角数的模型比较 | 官方训练配置 | 缺各组训练权重；未执行 |
| §5.2；附录 A/B | 模型规模和训练数据量的影响 | 官方 model/data scaling 配置 | 缺各组训练权重；未执行 |
| §5.3；Figure 8 | 移除训练位置插值后的性能 | 代码中的 image-index embedding | 已核对机制；缺独立训练模型 |
| §5.4；Table 5 | 使用 aligned local 或 global 点图的差别 | 10 视角同次预测、双分支指标；新增跨数据集 `--head both` | DTU配对已完成；NRGBD配对执行中；7-Scenes排队 |
| 附录 C/D/E/F | Gaussian splatting、BA、深度 benchmark 与可视化 | 已有点云；官方 robustmvd 接口 | 点云可视化部分完成；其余未执行 |

## 步骤 1：纠正比较口径

论文 Table 4 的 Fast3R 行为 Accuracy median **1.706**、Completion median **0.857**。历史 DTU JSON 中对应的是 `aggregate_mean.accuracy_median=2.930606`、`completion_median=1.819960`，不是 `accuracy=5.246037` 与 `completion=3.610983`。这里是先算每场景所有点距离的中位数，再在 22 场景间求均值；不能与全体点距离均值混用。指标越低越好。

历史结果相对论文分别约高 71.8% 与 112.4%，因此不能宣称论文数值已经复现成功。需要继续核查权重来源、数据预处理、置信度、随机 image-index embedding 与对齐设置；当前证据不足以认定某个因素就是差距原因。

## 步骤 2：让后续结果可追踪

§3.2 损失额外运行了 `python scripts/verify_fast3r_loss.py`。使用3个合成点图，确认精确预测与统一放大7倍的预测具有相同归一化损失，空间扰动会增加误差，并检查 global/local 点图与置信度的梯度均有限。修复了 `losses.py` 残留旧包名的导入。结果记录在 `results/loss_checks.json`；这是损失数值验证，不计作训练完成。

历史报告没有 seed，且模型在推理时也会随机采样 image-index embedding。新实验记录 Python/NumPy/PyTorch seed、实际图片标签、GPU、软件版本、checkpoint 配置 SHA256。每场景、每视角数使用稳定 seed，10 视角的 local/global 分支复用同一组预测。CUDA 算子未强制 deterministic，因此记录种子不代表跨硬件逐 bit 一致。

```bash
python scripts/fast3r_paper_experiments.py \
  --view-counts 3 5 10 20 \
  --seed 42 --head-chunk-size 2 \
  --output-json results/dtu_paper_experiments.json
```

默认 22 场景、每场景 4 组输入，共 88 次评测前向；10 视角同时报告 local/global。首场景额外预热一次，再测量三次前向。DPT head 的 batch 分成 2 视角小块，以控制显存，所有视角仍一起经过融合 Transformer。

视角实验从加载器完整帧序列中均匀选择 3/5/10/20 张，保留同一首帧。它与官方 `kf_every=50//N` 的采样存在差别，各组 GT 点的并集也随输入变化；应标记为 Figure 5 的本地适配实验。不能把这些数值直接代入 Table 4。local/global 对比只改变用于指标的点图，并非省掉 local head 的模型计算，因此不证明移除 local head 能加速。

## 步骤 3：读取结果与画图

Notebook 末尾的“论文逐项核对”单元独立读取结果，验证 22 场景、逐场景均值、论文 median 对照，绘制视角数、耗时、峰值显存与 local/global 差异图。阅读这些单元无需重新加载模型或 GPU。

### 2026-10-01 新实测结果

官方 stride=5、seed=42、DPT chunk=2 的独立运行完成22场景，见 [`results/dtu_seed42_stride5.json`](results/dtu_seed42_stride5.json)。Accuracy/Completion median 为 **2.0827 / 1.0311**，仍高于 Table 4 的 **1.706 / 0.857**。历史运行与新运行同时存在seed、场景遍历顺序与head分块差异，尚未做单变量实验，不能将数值差异全归因于某一因素。

视角数适配实验共88次评测前向全部完成，每组覆盖22场景，见 [`results/dtu_paper_experiments.json`](results/dtu_paper_experiments.json)：

| 视角数/点图 | Accuracy median | Completion median |
| --- | ---: | ---: |
| 3 / aligned local | 3.3735 | 2.9197 |
| 5 / aligned local | 2.6129 | 1.8621 |
| 10 / aligned local | 1.9145 | 0.9070 |
| 10 / global | 2.0541 | 1.0239 |
| 20 / aligned local | 1.7020 | 0.9653 |

10视角配对结果支持在这组DTU设置下aligned local降低距离误差。增加视角改善了Accuracy，但Completion在20视角时回升，normal consistency也未随视角持续改善，因此没有验证论文所有指标均持续改善的完整结论。20视角的数值不能与Table 4的10视角直接对齐。

### 步骤 4：Neural RGB-D 全量评测与核验（§4.3 / Table 3）

2026-10-01 01:55（Asia/Shanghai），宿主服务正常退出（退出码0），日志记录 `PIPELINE COMPLETE`。结果 [`results/nrgbd_seed42_stride40.json`](results/nrgbd_seed42_stride40.json) 覆盖官方9个场景、共278个采样视角；场景集合与数据清单一致，所有逐场景指标有限，aggregate逐项核对等于9场景的算术平均。

| Table 3 指标（距离×100） | 本地公开权重 | 论文参考 | 相对误差增加 |
| --- | ---: | ---: | ---: |
| Accuracy median | 4.0165 | 3.40 | 18.13% |
| Completion median | 1.2001 | 1.01 | 18.82% |

本地native median为0.0401652412 / 0.0120010220米；这里先算每场景median再平均，最后乘100，不能用mean距离替代。数据最初通过ZIP解压CRC检查（`results/nrgbd_data_manifest.json`），ZIP现已移除但解压数据保留。Notebook分析单元读取实际JSON并保存对照表和逐场景结果。

协议使用官方NRGBD加载器、stride40完整有效轨迹、512分辨率、aligned local点图与官方指标实现，seed42，alignment置信度percentile85、metric percentile0。为适配12GiB显存，DPT按2视角分块；这是硬件适配，不是论文训练配置。公开HF权重与论文Table 3所用具体训练权重的对应关系、原始下载revision未验证；不把数值差距归因于未经单变量实验核实的因素，也不将论文251.1 FPS参考冒充本机性能。当前完成了该数据集的运行与报告，尚未匹配论文数值。

## 剩余工作的确切前提

### 步骤 5（执行中）：跨数据集配对消融与7-Scenes准备

- §5.4/Table 5：`scripts/fast3r_hf_dtu_eval.py --head both` 保证两种metric使用同一次前向、同一输入与GT。NRGBD后台服务 `fast3r-nrgbd-paired.service` 正执行全部9场景，独立输出 `results/nrgbd_paired_seed42_stride40.json`，不覆盖Table 3历史报告。局部/全局选择不省略head计算，不作为加速证据。配对runner的CPU mock测试通过。
- §4.3/Table 3、§5.4/Table 5：`scripts/queue_7scenes_reproduction.sh` 先准备官方7-Scenes全部TestSplit，再排队执行stride20的local/global评测；服务 `fast3r-7scenes-reproduction.service`，日志 `results/7scenes_pipeline.log`。全量JSON生成前不填写成绩。
- 7-Scenes磁盘适配：通过官方HTTP Range只获取测试序列ZIP，逐序列CRC/SHA256核验。仅保存原始stride20选中帧，不改变对应输入/GT点集；原始帧总数、实际帧编号写入逐序列inventory。加载器拒绝其他stride和训练随机采样，避免把稀疏目录误认为连续完整视频。重跑复用已完整准备的序列，不覆盖其他数据；删除的只是脚本自身创建的临时ZIP。
- 深度遵循[固定版本SimpleRecon预处理](https://github.com/nianticlabs/simplerecon/blob/477aa5b32aa1b93f53abc72828f86023b6e46ce7/data_scripts/7scenes_preprocessing.py)，执行depth-to-RGB投影和z-buffer，输出`.depth.proj.png`；该参考不是TSDF raycasting，不以原始depth直接代替。向量化版本与标量逐点实现在离线测试上逐像素一致，8项测试通过。参考SHA256为`ac8dee029c28f600fc0e72ac79f8be19b83b7edfe9a2cdc88381abe5afe4e808`。数据遵循[Microsoft官方许可](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)，不提交数据集到GitHub。
- 本机空间由约8GiB降至2.4GiB，第一次准备在下载前因预算检查停止。后改成实际测试序列ZIP+1GiB预留，并在下载和预处理期间持续检查；若外部写入消耗空间则安全报错，不清理其他项目。Notebook新增完整结果的验证/分析单元；运行中仅显示pending。

### 后续推进顺序与真实边界

1. 完成以上数据准备/评测，核验全部轨迹集合与每项aggregate，执行Notebook分析并逐步提交GitHub。
2. 对已有DTU与NRGBD结果做协议审计，区分权重来源、随机seed、DPT chunk和数据掩码；不能调参至偶然接近参考值便宣称原协议复现。
3. §4.2/Table 1与Figure 4：准备CO3Dv2/RealEstate10K规定测试划分，修复官方脚本的作者绝对路径与checkpoint加载适配，报告真实RRA/RTA/mAA。缺GT或子集实验不会冒充全量benchmark。
4. §4.1/Table 2及附录C/D/E：需要合适的DUSt3R基线、Gaussian splatting/BA代码和规定测试数据，再执行本机可支持实验。不能用低分辨率/少视角结果冒充原A100多卡实验。
5. §5.1训练视角、§5.2/附录A/B、§5.3/Figure 8：必须取得各组独立训练权重或额外训练资源。仅公开主模型权重不能完成这些对照。未经用户明确预算授权不租用算力、付费购买资源，也不把短程训练称为论文完整训练。

官方入口 `fast3r/eval.py` 接受 Lightning checkpoint；公开 HF 权重可以通过官方 `load_for_inference` 接口评测，因此“只能 Demo，不能评测”是不准确的。checkpoint 文件格式本身不是不能复现指标的证明；当前无法验证公开权重与各论文实验训练权重的对应关系。

仍缺完整数据：`data/co3d_50_seqs_per_category_subset_processed` 与RealEstate10K测试样本。`data/7_scenes_processed`正在逐序列准备，不能把已存在的部分目录当作全量完成。`data/neural_rgbd`已准备。请遵循 [官方 README](https://github.com/facebookresearch/fast3r#datasets) 与 [Spann3R 预处理说明](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md)，注意7-Scenes要求预处理深度，视频Demo没有相应GT。

论文 §4 的完整训练使用 128 张 A100-80GB、174K steps；当前单张约 12 GiB 显卡不能完成同规格训练。训练视角、模型规模、数据规模和无位置插值消融需要对应独立模型。Table 2 的 1000–1500 视角实验也超出当前硬件的原始设置。需要这些数据与权重/算力后，才能逐项将未完成状态改成真实实验完成。

简历可以表述为“基于官方公开权重复现多视图重建流程，完成 DTU 22 场景与 Neural RGB-D 9 场景评测，开展视角数与点图头消融并分析论文指标差距”；目前不能写“完整复现所有论文实验并达到论文指标”。
