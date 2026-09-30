# Fast3R：逐项复现与论文对应表

核对版本：[arXiv v2，2025-03-19](https://arxiv.org/html/2501.13928v2)。本记录更新于 2026-10-01。

## 完成标准

当前完成了公开权重的推理链路与 DTU 22 场景指标运行；整篇论文复现尚未完成。运行结束只说明实验产生了结果，还需要核对指标、采样协议、模型版本与论文数值的差距。以下区分实测、代码核对、缺数据与缺训练算力。

| 论文位置 | 要验证的问题 | 本项目证据 | 状态 |
| --- | --- | --- | --- |
| §3.1；§3.3；Figure 2 | 多张图是否一次输出 global/local 点图与置信度？ | Notebook 的推理与模块 hook；`config.json` | 推理验证已执行 |
| §3.2，Eq. (1)–(3) | 归一化点图回归、置信度加权损失 | `results/loss_checks.json` | 数值/梯度检查通过；尚未重新训练 |
| §3.4；§4.1；Table 2 | 视角数增加时耗时与显存如何变化？ | `results/dtu_paper_experiments.json` 的 performance | 本机适配实验；单卡不覆盖论文 A100/多卡设置 |
| §4.2；Table 1 | CO3D / RealEstate10K 的 RRA、RTA、mAA | 官方 pose 配置与脚本 | 缺对应测试集；PnP Demo 不能代替这些指标 |
| §4.3；Table 3 | 7-Scenes / NRGBD 重建 | NRGBD官方9序列已准备；全量评测进行中 | 7-Scenes仍缺预处理数据 |
| §4.3；Table 4 | DTU 完整 22 场景重建 | `demo_outputs/paper_eval/dtu_all.json` | 已运行；论文数值尚未对齐 |
| §5.1；Figure 5 | 测试视角数对重建质量的影响 | 新脚本 3/5/10/20 视角 | 本地均匀采样适配实验 |
| §5.1；Figures 6–7 | 不同训练视角数的模型比较 | 官方训练配置 | 缺各组训练权重；未执行 |
| §5.2；附录 A/B | 模型规模和训练数据量的影响 | 官方 model/data scaling 配置 | 缺各组训练权重；未执行 |
| §5.3；Figure 8 | 移除训练位置插值后的性能 | 代码中的 image-index embedding | 已核对机制；缺独立训练模型 |
| §5.4；Table 5 | 使用 aligned local 或 global 点图的差别 | 10 视角同次预测、双分支指标 | DTU 配对实验；其余两数据集未执行 |
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

Neural RGB-D 官方数据包已下载并通过ZIP解压CRC检查，9个序列已准备（`results/nrgbd_data_manifest.json`）。单序列数据检查通过，首个序列有30视角，图像512×512，有效深度比例约0.994。宿主服务 `fast3r-nrgbd-reproduction.service` 正执行stride=40全量评测；日志为 `results/nrgbd_pipeline.log`。得到完整JSON之前，该项仍标记为未完成。

## 剩余工作的确切前提

官方入口 `fast3r/eval.py` 接受 Lightning checkpoint；公开 HF 权重可以通过官方 `load_for_inference` 接口评测，因此“只能 Demo，不能评测”是不准确的。checkpoint 文件格式本身不是不能复现指标的证明；当前无法验证公开权重与各论文实验训练权重的对应关系。

仍缺数据目录：`data/co3d_50_seqs_per_category_subset_processed`、`data/7_scenes_processed`，以及RealEstate10K测试样本。`data/neural_rgbd`现已准备。请遵循 [官方 README](https://github.com/facebookresearch/fast3r#datasets) 与 [Spann3R 预处理说明](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md)，注意7-Scenes要求预处理深度，视频Demo没有相应GT。

论文 §4 的完整训练使用 128 张 A100-80GB、174K steps；当前单张约 12 GiB 显卡不能完成同规格训练。训练视角、模型规模、数据规模和无位置插值消融需要对应独立模型。Table 2 的 1000–1500 视角实验也超出当前硬件的原始设置。需要这些数据与权重/算力后，才能逐项将未完成状态改成真实实验完成。

简历可以表述为“基于官方公开权重复现多视图重建流程，完成 DTU 22 场景评测并开展视角数与点图头消融”；目前不能写“完整复现所有论文实验并达到论文指标”。
