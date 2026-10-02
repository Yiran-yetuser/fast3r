# Fast3R：逐项复现与论文对应表

核对版本：[arXiv v2，2025-03-19](https://arxiv.org/html/2501.13928v2)。本记录更新于 2026-10-02。

## 完成标准

当前完成了公开权重的推理链路、DTU 22 场景、Neural RGB-D 9 场景和7-Scenes全部18条测试轨迹指标运行；整篇论文复现尚未完成。运行结束只说明实验产生了结果，还需要核对指标、采样协议、模型版本与论文数值的差距。以下区分实测、代码核对、缺数据与缺训练算力。

| 论文位置 | 要验证的问题 | 本项目证据 | 状态 |
| --- | --- | --- | --- |
| §3.1；§3.3；Figure 2 | 多张图是否一次输出 global/local 点图与置信度？ | Notebook 的推理与模块 hook；`config.json` | 推理验证已执行 |
| §3.2，Eq. (1)–(3) | 归一化点图回归、置信度加权损失 | `results/loss_checks.json` | 数值/梯度检查通过；尚未重新训练 |
| §3.4；§4.1；Table 2 | 视角数增加时耗时与显存如何变化？ | `results/dtu_paper_experiments.json` 的 performance | 本机适配实验；单卡不覆盖论文 A100/多卡设置 |
| §4.2；Table 1 | CO3D / RealEstate10K 的 RRA、RTA、mAA | 单卡HF位姿入口；RE10K1832相机记录及265447帧核验 | RGB/GT已核验1756/1832场景，缺76，尚无正式位姿指标 |
| §4.3；Table 3 | 7-Scenes / NRGBD 重建 | `results/nrgbd_seed42_stride40.json`；`results/7scenes_paired_seed42_stride20.json` | NRGBD完整9场景、7-Scenes全部18测试轨迹已运行核验；未对齐论文数值 |
| §4.3；Table 4 | DTU 完整 22 场景重建 | `demo_outputs/paper_eval/dtu_all.json` | 已运行；论文数值尚未对齐 |
| §5.1；Figure 5 | 测试视角数对重建质量的影响 | 新脚本 3/5/10/20 视角 | 本地均匀采样适配实验 |
| §5.1；Figures 6–7 | 不同训练视角数的模型比较 | 官方训练配置 | 缺各组训练权重；未执行 |
| §5.2；附录 A/B | 模型规模和训练数据量的影响 | 官方 model/data scaling 配置 | 缺各组训练权重；未执行 |
| §5.3；Figure 8 | 移除训练位置插值后的性能 | 代码中的 image-index embedding | 已核对机制；缺独立训练模型 |
| §5.4；Table 5 | 使用 aligned local 或 global 点图的差别 | DTU10视角；NRGBD/7-Scenes全量同次预测、双分支指标 | 三个数据集配对已完成；local优势并非所有距离指标均成立 |
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

### 步骤 5：NRGBD配对与7-Scenes全量配对评测已完成

- §5.4/Table 5：`scripts/fast3r_hf_dtu_eval.py --head both` 保证两种metric使用同一次前向、同一输入与GT。NRGBD后台服务 `fast3r-nrgbd-paired.service` 已正常退出（退出码0），独立输出 `results/nrgbd_paired_seed42_stride40.json`，不覆盖Table 3历史报告。9场景唯一集合、有限逐场景指标及全部aggregate_by_head核验通过；所有local指标与历史seed42/stride40运行逐项完全相等。局部/全局选择不省略head计算，不作为加速证据。配对runner的CPU mock测试通过。
- §4.3/Table 3、§5.4/Table 5：`scripts/queue_7scenes_reproduction.sh` 准备官方7-Scenes全部TestSplit后执行stride20的local/global评测；2026-10-01 12:44:47（Asia/Shanghai）日志记录 `PIPELINE COMPLETE`，宿主服务正常退出（退出码0），无残留评测进程。完整结果为 `results/7scenes_paired_seed42_stride20.json`。
- 7-Scenes磁盘适配：通过官方HTTP Range只获取测试序列ZIP，逐序列CRC/SHA256核验。仅保存原始stride20选中帧，不改变对应输入/GT点集；原始帧总数、实际帧编号写入逐序列inventory。加载器拒绝其他stride和训练随机采样，避免把稀疏目录误认为连续完整视频。重跑复用已完整准备的序列，不覆盖其他数据；删除的只是脚本自身创建的临时ZIP。
- 深度遵循[固定版本SimpleRecon预处理](https://github.com/nianticlabs/simplerecon/blob/477aa5b32aa1b93f53abc72828f86023b6e46ce7/data_scripts/7scenes_preprocessing.py)，执行depth-to-RGB投影和z-buffer，输出`.depth.proj.png`；该参考不是TSDF raycasting，不以原始depth直接代替。向量化版本与标量逐点实现在离线测试上逐像素一致，8项测试通过。参考SHA256为`ac8dee029c28f600fc0e72ac79f8be19b83b7edfe9a2cdc88381abe5afe4e808`。数据遵循[Microsoft官方许可](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)，不提交数据集到GitHub。
- 本机空间由约8GiB降至2.4GiB，第一次准备在下载前因预算检查停止。后改成实际测试序列ZIP+1GiB预留，并在下载和预处理期间持续检查；若外部写入消耗空间则安全报错，不清理其他项目。Notebook新增完整结果的验证/分析单元；运行中仅显示pending。

NRGBD Table 5口径（9场景mean距离平均×100，越低越好）：

| 点图 | 本地Accuracy mean | 本地Completion mean | 论文Accuracy参考 | 论文Completion参考 |
| --- | ---: | ---: | ---: | ---: |
| aligned local | 9.7108 | 3.1292 | 4.39 | 1.28 |
| global | 9.2594 | 3.1789 | 4.85 | 1.32 |

当前公开权重/本地协议下，aligned local降低Completion且提高normal consistency，但Accuracy更高，未完整复现论文local两项距离都更好的结论。Table 5与Table 3使用不同统计口径，不能混比mean和median；checkpoint对应关系及协议差异未核实，不据此断言论文结论错误。[逐场景差值图](results/figures/nrgbd_paired_heads.png)来自实际配对结果。

7-Scenes清单 [`results/7scenes_data_manifest.json`](results/7scenes_data_manifest.json) 为`prepared`：覆盖7类场景、官方18条TestSplit轨迹、原始17,000帧与850个stride20视角。每条轨迹帧编号与原始帧总数匹配，测试序列CRC/SHA256记录齐全，全量集合与保存的官方TestSplit一致。准备阶段850组文件全部解码检查通过：RGB为480×640×3，注册深度为480×640 uint16，位姿为有限4×4矩阵。首条chess测试轨迹dry-run为50视角，有效深度比例约0.799。

新增只读校验入口 `python scripts/validate_7scenes_report.py`：完整报告的唯一轨迹集合与prepared清单及保存的TestSplit严格一致，每条轨迹视角数/稳定seed匹配；每个head的8项aggregate均等于18条轨迹有限指标的算术平均，所有`paired_same_forward`为真，兼容的`aggregate_mean`明确属于local。同时复核NRGBD local与历史报告逐项相等，不覆盖历史JSON。

| 7-Scenes Table 3（逐轨迹median平均×100） | 本地公开权重 | 论文参考 | 相对误差增加 |
| --- | ---: | ---: | ---: |
| Accuracy median | 3.3035 | 1.58 | 109.08% |
| Completion median | 3.1058 | 0.93 | 233.95% |

| 7-Scenes Table 5（逐轨迹mean平均×100） | 本地Accuracy mean | 本地Completion mean | 论文Accuracy参考 | 论文Completion参考 |
| --- | ---: | ---: | ---: | ---: |
| aligned local | 6.3613 | 7.0516 | 2.84 | 1.37 |
| global | 6.9567 | 6.3510 | 4.81 | 1.64 |

当前7-Scenes配对设置下local降低Accuracy且改善汇总normal consistency，但Completion更高，不能宣称local全面优于global。Table 3数值与论文差距明显；完成完整测试划分不等于匹配论文结果。协议为公开HF权重、512×512加载器输出、seed42、stride20、16-mixed、DPT chunk2、alignment percentile85、metric percentile0。公开权重对应关系、原始revision、深度/裁剪/对齐协议差异仍需审计；未通过单变量实验定位差距原因，也未通过调参选择最好结果。[逐轨迹双分支差值图](results/figures/7scenes_paired_heads.png)来自真实结果。

逐轨迹检查发现：local的Accuracy在12/18条轨迹更低，Completion在14/18条更低，但`office/seq-06`的Completion mean×100为local56.1612、global29.5266，较大的反向差值影响了完整18轨迹平均。这只是定位值得核查的轨迹，不是证明根因；保留该轨迹参与全部汇总，不剔除难例或改用有利的子集成绩。

### 步骤6：公开权重/协议审计与Table 1数据前提（2026-10-01）

已新增[`PROTOCOL_AUDIT.md`](PROTOCOL_AUDIT.md)和小型JSON证据快照。重新计算本地weight/config SHA256，与当前HF revision `a2c770b768ceb3a53c36c4f7a3619db0413dc3a1`的LFS哈希及配置原文逐项相同；原始下载revision仍未知，公开权重与论文各实验权重的对应关系仍未证明。这部分来源审计已经完成，不再笼统记成“权重来源全未核实”。

已核实公开代码的对应像素RoMa注册（含scale）并不是迭代最近邻ICP；§3.2公式印出的log-confidence正号与实现负号不同。整数resolution512产生512×512重建输入。保留公开代码，不将这些事项直接断言为误差根因。CO3D组合采样恒读`combinations[0]`再jitter的实现风险也已记录，尚未改采样协议。

新增只读`check_pose_data.py`：按全量规定清单检查RGB/GT/相机元数据，缺损返回incomplete，不静默跳过。10项合成离线测试通过。当前CO3D processed测试清单缺失；RE10K指定文件实际含1,832个唯一ID，当前目标目录0个数据对准备好。该预检查不是位姿评测，原作者绝对路径/HF加载适配尚未完成，不填报Table 1指标。

宿主项目盘仅余约2.38GiB，仍需留1GiB，未有已挂载的额外数据存储。官方CO3D的8.9GB挑战single-sequence子集不是规定测试划分；不能用它替换，也不假称必须下载全部5.5TB。稀疏下载的正确选择/峰值预算尚未验证。下一步需要提供更大的可写目录或已有规定数据路径，先算预算再下载必要部分；不会擅自挂载或写入未挂载NTFS分区。独立训练消融另外需要权重来源或训练资源授权。

### 步骤7：扩容后恢复Table 1数据阶段（2026-10-02）

项目分区已扩为约251G，初查可用122G；步骤6的2.38GiB限制是历史状态，不再作为当前阻塞。
规定RE10K测试文件仍为1832个唯一ID，未因文件名“1800”或镜像可用性改变集合。
官方相机元数据下载、长度/MD5/SHA256与逐TXT解析已完成；服务正常退出0，
01:55:16日志记录完成。清单见[`results/re10k_metadata_manifest.json`](results/re10k_metadata_manifest.json)。
prepared清单及全部文件哈希匹配时重跑跳过网络，不重复下载。

新入口[`scripts/fast3r_hf_re10k_pose_eval.py`](scripts/fast3r_hf_re10k_pose_eval.py)使用公开HF权重，
沿用原RE10K principal-point crop/resize到512×288、first-view global focal、官方相对位姿指标。
单卡16-mixed、DPT chunk2、seed42；10视角按scene稳定抽取并记录timestamp和输入SHA。
GT相机不输入网络和PnP。顺序OpenCV RANSAC固定seed是可追踪适配，不冒充原线程执行和未公布frame draws。
PnP实际mask为conf>1，公开函数的percentile参数并未用于该mask；失败identity fallback保留并单独计数。
逐scene JSON支持恢复并重新计算已保存位姿的指标，协议/输入哈希不一致拒绝复用，最终报告不覆盖。
13项初版离线合成测试通过，随后补充3项RGB下载恢复测试；它们不是Table 1实测。
真实全量dry-run因1832个RGB目录缺失而拒绝运行，没有生成正式位姿成绩。
随后用下载中的首个完整chunk做CRC与safe weights-only格式probe：17个clip全部RGB解码为360×640，
其中8个规定ID的camera/timestamp与官方TXT核对通过，最大绝对差约2.38e-7。
独立保存其中一个真实clip的全部105候选帧并随机抽10视角，HF单卡前向→focal→PnP→45对指标成功，
PnP失败0/10。该接线测试的RRA@15为1.0、RTA@15约0.0889、mAA30约0.1627；
这些是1个scene的smoke值，不是Table1全量成绩，不能拿来声称论文效果复现。
原始probe和smoke输出分别保存，不与1832-scene正式输出混用。
断点再运行没有重新加载模型，独立输出与首次smoke JSON逐字节一致；全部47项离线测试通过。

扩容后预算允许下载pixelSplat作者公开test-only归档（55,604,889,849bytes，约51.79GiB），
无需下载完整训练集。`fast3r-re10k-rgb-archive.service`于02:02:47启动；
`results/re10k_rgb_archive.log`和`data/re10k_test_only.zip.part`记录进度。
镜像从[作者README](https://github.com/dcharatan/pixelsplat#acquiring-datasets)确认，
已读取少量ZIP头；作者loader预期JPEG为360×640，但本机尚未解码整个归档核验。
下载仅说明获得候选数据：后续必须核验1832 ID完整覆盖、timestamp/官方相机GT、RGB裁剪与分辨率，
逐chunk安全解析/CRC检查后只保留规定测试scene。不同项目的evaluation_index不替换Fast3R清单。
该服务器不支持HTTP Range；恢复时重新传输并校验已有前缀再追加，网络流量可能增加，磁盘数据不抹掉。
每次下载预算留至少1GiB并持续检查空间；没有无预算整包解压，也没有使用不受限制的torch pickle。

Notebook新增数据准备与真实smoke分析，完整续接顺序见[`CONTINUATION.md`](CONTINUATION.md)。
当前聊天已创建每30分钟的heartbeat接续，不保证额度重置瞬间恢复；需要开机和应用运行。
确定性后台下载与Codex额度独立，额度可用后的定时触发从检查点继续；不购买额度、不兑换重置权益。
CO3D规定测试选择、RE10K正式指标与独立训练实验仍未完成。

### 步骤8：CO3D41类协议来源与空间可行性（2026-10-02 06:21续接）

对应§4.2/Table1的数据前提，不是新的位姿成绩。固定PoseDiffusion官方代码revision
`b138198e2891a0f1a1c3435614b9490adb7fd4d6`的41类`TRAINING_CATEGORIES`与seen/10-view配置
给出有来源的类别映射；它与论文“41类中的未见轨迹”相符，但仍是协议推断，未取得Fast3R作者processed清单确认。
新`audit_co3d_41_categories.py`只AST读取源常量、固定源码/配置SHA，不执行远程代码。
在上一已核验51类候选上过滤10个PoseDiffusion unseen类别，保留原seed+类别索引/帧顺序，
实际得到41类、2011轨迹、399204候选帧；不是41×50猜出的2050，也不是全部RGB准备完成。

apple官方6个图像ZIP完成真实有界Range目录探测：83,941,069bytes目录传输，
9563组RGB/depth/mask，目录标示大小共6,073,570,411bytes。大ZIP不落盘、图像成员未下载，
未验证成员CRC/图像解码/大ZIP SHA。它验证“可以只读中央目录”的可行性，不代表全41类预算。
`fast3r-co3d-seen41-storage.service`06:35:46真实启动、确认running，
逐类校验完整候选成员路径集合及跨ZIP无重复，保存小JSON断点，最终输出
`results/co3d_seen41_storage_budget_20261002.json`。服务不使用GPU或修改采样划分。

Notebook `paper-co3d41`已真实执行，保存来源、计数、apple探测及限制；84项离线测试通过。
下一步先核验全41类预算，再设计候选清单不变的存储与相机预处理；
Fast3R原始100@范围、逐轨迹/帧选择、权重和GT几何仍需审计，未产生正式Table1 RRA/RTA/mAA。
RE10K76缺失候选的完整来源扫描仍健康运行，尚未完成全SHA和集合/几何核验，不重复启动。

06:51续接：CO3D目录预算服务因banana实际229525成员超过初版保护上限而退出。
诊断确认目录29.04MB仍低于32MiB字节上限、宿主内存充足；有界调整成员上限后06:54:25恢复。
apple/backpack已完成检查点经固定旧代码SHA、输入/路径SHA及逐包汇总核验，导入新v2目录，
不覆盖历史或重做两类网络索引。恢复越过原失败位置，88项离线测试通过，Notebook记录真实快照。
仍仅对应§4.2/Table1的数据预算阶段，没有新的模型成绩或完整41类数据就绪声明。

07:21续接：v2在bowl目录37,498,281bytes处触发32MiB保护阈值。新增全41类、235个官方ZIP
尾部预检完成，只读15.41MB尾部，测得最大目录37.50MB、最大300491成员；不读图像或完整大包。
v3采用64MiB硬上限且按每包实际目录大小+尾部余量收紧读取；原400000成员上限与安全检查保留。
07:29:59恢复，8类旧记录经固定v2代码/输入/路径/新footer身份核验复用，不重复中央目录网络读取。
bowl已完成10098组成员目录核验。95项测试通过，Notebook保存真实尾部预检和恢复快照；
全41类空间预算、图像/GT就绪和正式Table1成绩仍未完成，当前v3文件名见CONTINUATION。

### 后续推进顺序与真实边界

08:21续接完成§4.2/Table1的数据目录预算与独立核验：41类、2011轨迹、399204组每种
RGB/depth/mask路径及235包汇总一致。原始成员标示297.20GB（276.79GiB），其中深度246.35GB；
约46GiB可用空间不支持原始全量落盘，继续按需预处理，不将目录覆盖称为图片CRC/GT就绪。
源采样名义追踪采用组合seed42、dataset seed777、epoch0且全部有效：100@仅99轨迹、38类、
918唯一帧，58次重复；全部2011轨迹的另一个适配追踪为18588唯一帧、1149次重复。
这不是实际图片输入或论文规定draws，真实有效深度、补采/重试未执行；不作为Table1成绩。
Notebook保存真实预算/采样输出，旧输出保留。

随后41类/2011轨迹/399204帧的本地相机metadata审计完整结束：按官方set-list连接编号/路径，
核验原始ndc_isotropic、depth scale、R/T与K/c2w。38869个filename编号不等于annotation编号，
不能按文件名猜GT。car有12项平移量>1e6，最大约8.296e16；记录绝对/尺度逆残差，不改GT、不删场景。
这是数学与元数据核验，不证明float32指标稳定、RGB/depth/crop等价或已有camera NPZ就绪。
结果与当前后续顺序见CONTINUATION；正式RRA/RTA/mAA仍未产生。
111项离线测试通过；Notebook共16个paper分析单元均保存真实执行输出，合成测试不计为论文成绩。

09:21续接完成§4.2/Table1相对位姿的CPU数值诊断：全部候选41类/2011轨迹/399204帧，
共39624443个同轨迹相机对；没有读取RGB或模型预测。遵循参考先构造float32 w2c再求逆，
再调用公开relative-pose函数，比较float64对应结果，不修改源GT/指标或删场景。
可定义方向的对中3301对精度间角差>1°；基线>1e-6×max(1,相机中心范数)的独立诊断分组
仍有340对>1°、最大5.9845°。该阈值不是论文pair过滤规则，所有pair均计数保留。
直接中心相同1181对，而float32相对计算零平移1375对；GT自比较这1375对的平移角为90°。
这些属于零基线/数值行为，不是网络预测错误，不宣称float32整体稳定或论文效果复现。
新JSON记录运行版本、代码/源SHA、逐类汇总和风险轨迹；Notebook新增数学解释及实际输出。
117项离线测试通过；17个paper分析单元均有真实输出，旧模型实验与历史报告不覆盖。
下一步从有界真实Range成员CRC/解码推进按需预处理与原候选池延迟加载，不继续重复metadata审计。

### 09:51续接：真实CO3D十帧预处理接线（仍是§4.2/Table1前提）

apple一条轨迹的前十个原候选帧完成30个RGB/depth/mask成员的有界Range、CRC与解码，
没有下载整个大ZIP。原始成员7.74MB，首次传输91.66MB含完整apple中央目录；复核重跑传输0bytes。
参照固定DUSt3R预处理实际运行，适配后的十帧JPEG/PNG逐文件字节一致，K/float32 pose/max-depth
逐NPZ数组一致。保留FP16位模式深度、主点crop与半像素K缩放、NEAREST、max-depth量化。
这是同运行库、十帧范围的接线证明，不外推到399204帧或作者processed清单。

独立核验调用真实公开单视图loader，仍传入原202帧候选pool，不将十帧稀疏目录替换为采样范围。
十帧均有有效深度；公开近方形随机方向逻辑保留，两个尺寸朝向均记录，没有拉伸图像。
Notebook数学说明和真实报告均已保存，128项离线测试通过，18个paper分析单元有实际输出。
下一步实现严格按需加载并记录_get_views真实有效性/补采/scene retries；尚无Table1全量模型成绩。

### 10:21续接：一个真实100@映射样本与严格按需加载

完整候选池lazy adapter已接到公开_get_views和base.__getitem__，只移除文件异常吞掉逻辑，
不改jitter、重复视图、真实零深度失效/补采/scene retry。138项离线测试包含合成零深度重试。
实际只执行epoch0 wrapper index0：bicycle轨迹10views/9unique，原133帧pool保持。
10次读取均有效，无真实补采/换场景；完整返回值与原loader精确相等。独立固定参考预处理
核验9帧JPEG/PNG字节和NPZ数组一致。这是单样本接线，不是全部100@或正式Table1。
首次NumPy int64 JSON保存失败已安全修复，保留未完整文件，新v2报告及Notebook保存真实输出。
HF位姿单样本smoke已执行，见10:51续接；连续100@共享失效状态/补采RNG还需断点设计，作者采样/权重身份未确认。

2026-10-02继续§4.2/Table1数据可行性：新增精确HTTP Range/gzip流式镜像审计，
真实8MiB前缀核验通过，但只见1个场景，未确认补齐76。后台仅保留缺失场景候选PNG，
不覆盖已有JPEG、不将未完整来源SHA/裁剪验证的候选当正式数据。
另启动CO3D51类官方元数据ZIP/SHA/CRC准备，按固定DUSt3R公开规则推导test序列，
还未准备RGB/depth/mask/NPZ，也未验证作者选择等价性；不会用challenge单序列子集替换。
论文41类指“这些类别中的未见轨迹”，不是41个未见类别；原eval的100 @是采样长度包装。
73项离线测试通过，新Notebook单元记录来源审计与待验证条件，不产生新的RRA/RTA/mAA。

CO3D元数据准备随后正常结束0，51类ZIP共1.316GB通过官方SHA/CRC，独立选序列核验通过。
实际默认公开规则得到51非空类别、2511序列、498757候选帧，不是论文所述41类。
不能任意删10类凑数量；下一步先核对原41类协议，再预算必要RGB/depth/mask及相机转换。
全帧索引留本地，仅提交35KB汇总和独立核验；Notebook保存真实计数及“不等价”的说明。

2026-10-02 05:10:00，恢复下载正常结束0，55,604,889,849bytes完整归档及SHA已记录。
进一步目录/index核验发现：7286个镜像ID只覆盖规定1832中的1756，缺76个，
详见`results/re10k_rgb_index_audit.json`。缺失集合保留，不产生1756交集的“全量”成绩。
58项离线测试通过，宿主`fast3r-re10k-rgb-prepare.service`已逐块验证CRC、安全tensor、
官方全部timestamp/camera与RGB解码，并保存covered clip的全部候选帧，不改变采样范围。
真实原图存在640×338，不一律640×360；保留实际尺寸和编码字节，按原入口实际宽高换算K，
记录尺寸差异，不能根据首chunk固定尺寸推测全数据。05:36:54服务正常结束0，
1756场景共265447帧完成准备；独立检查全部已保存RGB字节/SHA、原官方GT SHA与候选集合通过。
1753场景360×640、2场景338×640、1场景272×640（H×W），最大camera绝对差9.51156e-7。
见`results/re10k_rgb_prepared_manifest.json`与`results/re10k_rgb_verification_20261002.json`。
这是已覆盖部分的数据就绪，不是完整1832数据或位姿成绩。
另一个公开HF test索引同样缺76；两个原URL匿名探测遇429/登录错误，不作为永久缺失证明。
其他公开归档的有界流式选择/完整GT可行性仍待核实；不盲目下载205GB归档，不静默换测试划分。

2026-10-02 03:25:41，RE10K候选RGB归档HTTP响应在54,818,855,664bytes提前结束，
服务退出1；预期55,604,889,849bytes，未生成完整归档清单，不能继续正式位姿评测。
重新核对镜像长度/修改时间未变，03:52:39安全重启，保留已有数据与日志。
因服务器无Range支持，先重新传输并逐字节核对已有前缀，再追加786,034,185bytes；
这期间本地大小不增长不代表进程失败。新增恢复进度日志与截断保留测试，48项测试通过。
证据见`results/re10k_rgb_recovery_20261002.json`；对应§4.2/Table1数据准备，
不是下载完成或新的RRA/RTA/mAA实测，具体网络/服务端根因仍未确定。

1. 已完成三个重建数据集的公开权重运行、NRGBD/7-Scenes完整配对集合与aggregate核验；保存Notebook真实分析并逐步提交GitHub，不视为整篇论文完成。
2. 第一轮权重与公开代码协议审计已完成，见上述步骤6；误差根因及论文各实验checkpoint对应关系仍未定位。不能调参至偶然接近参考值便宣称原协议复现。
3. §4.2/Table 1与Figure 4：准备CO3Dv2/RealEstate10K规定测试划分，修复官方脚本的作者绝对路径与checkpoint加载适配，报告真实RRA/RTA/mAA。缺GT或子集实验不会冒充全量benchmark。
4. §4.1/Table 2及附录C/D/E：需要合适的DUSt3R基线、Gaussian splatting/BA代码和规定测试数据，再执行本机可支持实验。不能用低分辨率/少视角结果冒充原A100多卡实验。
5. §5.1训练视角、§5.2/附录A/B、§5.3/Figure 8：必须取得各组独立训练权重或额外训练资源。仅公开主模型权重不能完成这些对照。未经用户明确预算授权不租用算力、付费购买资源，也不把短程训练称为论文完整训练。

官方入口 `fast3r/eval.py` 接受 Lightning checkpoint；公开 HF 权重可以通过官方 `load_for_inference` 接口评测，因此“只能 Demo，不能评测”是不准确的。checkpoint 文件格式本身不是不能复现指标的证明；当前无法验证公开权重与各论文实验训练权重的对应关系。

仍缺完整数据：`data/co3d_50_seqs_per_category_subset_processed` 与RealEstate10K测试样本。`data/7_scenes_processed`已按官方全部TestSplit准备stride20评测帧。`data/neural_rgbd`已准备。请遵循 [官方 README](https://github.com/facebookresearch/fast3r#datasets) 与 [Spann3R 预处理说明](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md)，注意7-Scenes要求预处理深度，视频Demo没有相应GT。

论文 §4 的完整训练使用 128 张 A100-80GB、174K steps；当前单张约 12 GiB 显卡不能完成同规格训练。训练视角、模型规模、数据规模和无位置插值消融需要对应独立模型。Table 2 的 1000–1500 视角实验也超出当前硬件的原始设置。需要这些数据与权重/算力后，才能逐项将未完成状态改成真实实验完成。

简历可以表述为“基于官方公开权重复现多视图重建流程，完成DTU 22场景、Neural RGB-D 9场景与7-Scenes全部18测试轨迹评测，开展视角数与点图头消融并分析论文指标差距”；目前不能写“完整复现所有论文实验并达到论文指标”。

### 10:51续接：公开HF的CO3D真实单样本位姿输出

§4.2/Table 1接线阶段：新入口复核已归档draw0、完整候选池、raw/processed输入SHA、
相机/RNG/公开权重。模型只接收RGB与true_shape；GT K仍用于公开crop，GT深度/点图/K/位姿
不进入网络或focal/PnP。初始化后重设seed12303675，16-mixed、DPT chunk2，10views/9unique、45pairs。

首次11:01:12在新入口形状断言失败：HF landscape_only=False输出portrait，发布版
correct_preds_orientation转回loader landscape，误断言要求portrait。只修断言，不改发布算法；
真实head wrapper回归测试通过。旧预检查/失败代码/日志保留，另写v2预检查安全重试。
fast3r-co3d-pose-smoke.service在GPU≥10240MiB空闲后执行，11:05:17正常退出0。
结果results/co3d_pose_draw0_seed42_20261002.json及恢复核验JSON已归档；verify-only重算
保存c2w指标，无模型重跑。raw confidence [1,512,384]，校正后[1,384,512]。

| 单样本真实指标（比例×100） | 5° | 15° | 30° |
| --- | ---: | ---: | ---: |
| RRA | 0.0000% | 0.0000% | 0.0000% |
| RTA | 0.0000% | 8.8889% | 13.3333% |

mAA@30=0%，PnP失败0/10；旋转误差50.1264°–171.6512°，首视图focal约27.9829px。
GT-self RTA@30=97.7778%是保留重复零基线pair的90°方向角行为，不是预测表现。
实际前向1.08593s、峰值allocated4,970,486,784bytes，仅此样本，不与Table 2硬件性能混比。
144项离线测试通过；Notebook新增2个真实报告单元，旧19个未改。
运行/验证通过不代表指标好，更不是正式Table 1成绩；低分不删除、不静默改算法。
下一步固定同输入/同次前向比较方向校正与focal路径，保留发布流程基线，再设计连续100@断点。
RE10K全源SHA/76集合/图像几何仍待核验；未训练实验保持未完成。
