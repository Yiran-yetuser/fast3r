# Fast3R：逐项复现与论文对应表

核对版本：[arXiv v2，2025-03-19](https://arxiv.org/html/2501.13928v2)。本记录更新于 2026-10-03。

## 完成标准

当前完成了公开权重的推理链路、DTU 22 场景、Neural RGB-D 9 场景和7-Scenes全部18条测试轨迹指标运行；整篇论文复现尚未完成。运行结束只说明实验产生了结果，还需要核对指标、采样协议、模型版本与论文数值的差距。以下区分实测、代码核对、缺数据与缺训练算力。

| 论文位置 | 要验证的问题 | 本项目证据 | 状态 |
| --- | --- | --- | --- |
| §3.1；§3.3；Figure 2 | 多张图是否一次输出 global/local 点图与置信度？ | Notebook 的推理与模块 hook；`config.json` | 推理验证已执行 |
| §3.2，Eq. (1)–(3) | 归一化点图回归、置信度加权损失 | `results/loss_checks.json` | 数值/梯度检查通过；尚未重新训练 |
| §3.4；§4.1；Table 2 | 视角数增加时耗时与显存如何变化？ | `results/dtu_paper_experiments.json` 的 performance | 本机适配实验；单卡不覆盖论文 A100/多卡设置 |
| §4.2；Table 1 | CO3D / RealEstate10K 的 RRA、RTA、mAA | 旧CO3D100请求/4500pair及受控几何诊断；新51类源1000@采样/目录预算/首请求重放及完整输入门禁；RE10K1832相机记录及265447帧核验 | 新真实1000候选输入准备运行；后续GPU队列只等待完整输入核验，尚无新位姿成绩；旧mAA@30=23.8007%为候选适配；原作者清单/权重对应未确认，正式Table1未完成；RE10K缺76 |
| §4.3；Table 3 | 7-Scenes / NRGBD 重建 | `results/nrgbd_seed42_stride40.json`；`results/7scenes_paired_seed42_stride20.json` | NRGBD完整9场景、7-Scenes全部18测试轨迹已运行核验；未对齐论文数值 |
| §4.3；Table 4 | DTU 完整 22 场景重建 | `demo_outputs/paper_eval/dtu_all.json` | 已运行；论文数值尚未对齐 |
| §5.1；Figure 5 | 测试视角数对重建质量的影响 | 新脚本 3/5/10/20 视角 | 本地均匀采样适配实验 |
| §5.1；Figures 6–7 | 不同训练视角数的模型比较 | 官方训练配置 | 缺各组训练权重；未执行 |
| §5.2；附录 A/B | 模型规模和训练数据量的影响 | 官方 model/data scaling 配置 | 缺各组训练权重；未执行 |
| §5.3；Figure 8 | 移除训练位置插值后的性能 | 代码中的 image-index embedding | 已核对机制；缺独立训练模型 |
| §5.4；Table 5 | 使用 aligned local 或 global 点图的差别 | DTU10视角；NRGBD/7-Scenes全量同次预测、双分支指标 | 三个数据集配对已完成；local优势并非所有距离指标均成立 |
| 附录 C/D/E/F | Gaussian splatting、BA、深度 benchmark 与可视化 | 已有点云；官方 robustmvd 接口 | 点云可视化部分完成；其余未执行 |

## 最新：完整输入门禁与后续位姿队列（2026-10-03 12:27，§4.2/Table1候选）

真实准备服务健康运行，12:26日志21/1000；这不是已完成输入的全量独立证明。
新独立GPU队列已启动，**只等待**全1000准备摘要/独立SHA/CRC、tensor/GT/K/RNG/trace/state重放
以及CPU服务退出，fresh offline replay后再检查GPU至少10240MiB空闲；当前模型前向0、没有新位姿分数。
没有更改正在准备的身份绑定代码、旧100请求、数据、权重或历史Notebook输出。

新入口`scripts/fast3r_hf_co3d51_pose_eval_v1.py`沿用公开HF/global点图→预测焦距→发布PnP；
网络仅img/true_shape、GT不用于模型或焦距/PnP，seed42+request/16-mixed/chunk2。
重复视图的45pair/request与PnP失败identity/零焦距全部保留；完整1000才分别汇总
request级算术均值和45000 pair pooled结果。事务目录512MiB、每次至少预留1GiB。
独立核验`scripts/verify_co3d51_candidate_pose_v1.py`不重新推理网络，只核验保存的输入/GT、位姿和指标。
作者精确processed清单/RNG及HF论文权重组别仍未知，固定横向crop为声明候选选择，正式Table1不勾选。

真实门禁快照`results/co3d51_pose_input_gate_20261003.json`记录当时18/1000文件计数、
完整摘要/证明缺失、未加载模型/network0/forward0；不是最新数量或已独立核验18请求。
105单元Notebook新报告12:27经宿主Jupyter执行，原103逐项保留；244项离线测试通过。
两服务、日志、恢复和未来候选/独立证明文件见`CONTINUATION.md`；停机/挂起或实际失败仍需处理。
已有后台进程可在Codex额度不足时继续，但不是绕过额度或精确重置唤醒保证；整篇尚未完成。

## 已归档：源目录预算核验完成、真实1000请求准备运行（2026-10-03 11:21，§4.2/Table1前提）

51类目录预算服务10:57:12正常退出0，独立核验重新计算覆盖/数量/大小/源绑定，复用38类
旧索引，仅补13类目录。报告`results/co3d_51_source_storage_v1_20261003.json`，证明
`results/co3d_51_source_storage_verified_20261003.json`；不重做旧235包完整扫描。

| 源采样集合（不是GT就绪集合） | 每种成员数 | 原始RGB/depth/mask总bytes | 十进制GB |
| --- | ---: | ---: | ---: |
| all-valid名义1000@ | 8835 | 6,539,401,033 | 6.54 |
| jitter/最多5场景尝试保守可达集合 | 103599 | 76,458,893,204 | 76.46 |

目录大小不是CRC/解码/GT或processed预算证明；旧raw2GiB边界不能称1000装得下。
实际按需准备采用独立raw8GiB/processed3GiB/journal4GiB版本、每次至少留1GiB；
旧缓存/100请求结果冻结，不下载整个补采集合。包络可装下不保证全部补采完成，碰界硬停。

首个真实请求10视图/7不同帧经过Range/CRC/GT检查与跨进程只读重放，RGB tensor、GT/K、
RNG、load trace、pool attempts和共享after-state完全相同；证据
`results/co3d51_source_inputs_prefix1_verified_20261003.json`是不可变1请求快照，**不是1000已完成**。
横向crop512×384为声明新候选协议，不宣称原作者清单/RNG等价；新阶段模型前向0。
11:20启动独立CPU后台`fast3r-co3d51-source-prepare-v1.service`，完成请求事务可恢复；
全1000完成后自动离线只读重放，再核验与归档，当前未启动新GPU评测。
恢复入口`CONTINUATION.md`。Notebook103单元，原101逐项完全相同；新目录/首请求分析
11:25经宿主Jupyter真实执行无error，结构校验及完整229项离线测试通过。
公开HF与论文训练组别映射、RE10K缺76、完整训练/独立消融/附录仍未完成。

## 已归档源采样审计（2026-10-03 10:48，§4.2 / Table 1 前提，非位姿成绩）

真正调用发布`Co3d_Multiview`采样方法和`ResizedDataset(1000)`的all-valid stub，
名义结果为51类/836轨迹/8835不同帧，581/1000请求含重复帧、最少6不同帧。
不是1000条不同轨迹，也不是先随机抽10个不同帧。完整原始候选有序池仍为51类/2511轨迹/498757帧。
旧工程提案的1000不同轨迹/10000帧摘要保留，但不用于后续评测；缓存上限不等于实测空间预算。
`results/co3d_51_released_sampler_20261003.json`绑定原候选/发布源码/计划哈希；
Notebook在宿主Jupyter重放计划逐项相同，原97单元保留，新2个分析单元无error。
新增18项离线源采样/原子断点/目录预算/独立核验测试，完整套件218项通过。
dataset seed777有作者/发布依据，combination seed42与epoch0为声明的选择，原作者RNG/清单未恢复。

可达补采上界为2205轨迹/103599帧（±4 jitter、最多5个scene尝试），不是实际有效GT集合。
新独立目录预算服务复用38类本地索引，仅补13类缺失目录；不重做已归档235包扫描。
目录长度不证明CRC/解码/GT/crop有效，也不证明processed空间可装下；新GPU评测尚未启动。
进程/恢复/核验入口见`CONTINUATION.md`。原100请求及几何诊断报告保留，正式Table1仍未完成。

## 已归档输入几何对照（2026-10-03 07:15，§4.2 / Table 1）

HF实际使用`PatchEmbedDust3R`，不是ManyAR或DINO；它忽略true_shape，384×512张量
生成24×32个patch，portrait DPT却按32×24 reshape。实际patch类合成编号测试证实
reshape≠transpose（99.7396%的编号不匹配，非预测错误率）。另有像素投影不一致：
loader将网格transpose、K行交换，3D/GT位姿轴不变；PnP改用标准K。
完美合成点的原PnP旋转误差0°，转置像素/标准K约179.84°仍返回success；这不是网络质量分数。
CPU证据`results/co3d_portrait_geometry_audit_20261003.json`。历史100候选63全portrait、
29全landscape、8混合，661/1000视角portrait，不作为作者完整分布。

固定request0/2/3、帧序/GT/权重/seed/精度/焦距/PnP不变，不补采或删除视图。
原分支引用历史；仅形状标记保留RGB字节；横向crop只去掉基类分辨率自动反转，保留
其他crop/resize/半像素K语义。共5个新前向，横向控制两条件相同复用1次。
**不是同输入/同次前向消融**；crop改变视野，不能单独分解机制贡献。

| 选定request（非无偏benchmark） | 历史mAA% | 仅形状标记mAA% | 强制横向crop mAA% |
| --- | ---: | ---: | ---: |
| 0 bicycle，portrait | 0.0000 | 0.0000 | 59.7133 |
| 2 cup，landscape控制 | 92.2581 | 92.2581 | 92.2581 |
| 3 stopsign，portrait | 0.0000 | 0.0000 | 12.4014 |

6个新分支均无PnP失败，270对含重复视图保留；独立重算saved poses/pair指标并重放
RGB/GT/crop K与RNG标记。横向控制位姿/焦距/指标匹配历史；196项离线测试通过。
结果`results/co3d_landscape_input_diagnostic_v1_20261003.json`，证明
`results/co3d_landscape_input_diagnostic_verified_20261003.json`，Notebook有真实宿主输出与图。
核验器不重跑网络，预测图未保存，prediction hashes保留为provenance限制。
6项内存损坏注入均被独立核验器拒绝，未修改存储文件；Notebook原95单元逐项保留，新输出无error。
不报告选择probe均值，不改旧100成绩，不勾选正式Table1或整篇完成。
下一步预算51类/1000@作者描述候选，精确processed清单/RNG/权重对应仍未知。

## 已归档协议纠正与受控PnP诊断（2026-10-03，§4.2 / Table 1）

原41类/100请求是历史候选适配，**不是已确认的作者benchmark**。
[作者Issue #78回复](https://github.com/facebookresearch/fast3r/issues/78#issuecomment-2844393603)
说明实际沿用DUSt3R的1000次CO3D test采样、seed777、512×384，可能超过论文所写41类。
发布配置`configs/eval/eval_cam_pose/default.yaml`仍是100次。此前按PoseDiffusion推导41类
不能继续当成作者的必要选择规则；旧数据/报告保留，后续重审51类和1000请求。
回复没有提供原processed JSON或精确RNG顺序，公开HF权重与论文组别映射仍未知。
来源/日期/正文哈希见`results/co3d_author_protocol_update_20261003.json`。

新`diagnose_co3d_candidate_pnp.py`只选原报告request0/2/3（低分/高分/零焦距），
每个请求一次前向共享给四个PnP分支。精确RGB/GT/采样state复核，baseline位姿差最大0、
指标一致；GT不输入网络/焦距/PnP，重复pair和失败回退全部保留。独立核验12分支全部45pair。

| 固定request（非随机代表性样本） | 发布版mAA% | 仅top15 mask | 仅焦距搜索 | 搜索+top15 |
| --- | ---: | ---: | ---: | ---: |
| 0 bicycle，portrait | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 cup，landscape | 92.2581 | 94.2652 | 93.0466 | 92.4731 |
| 3 stopsign，portrait | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

搜索沿发布`fast_pnp(focal=None)`的100个确定性几何间隔候选，不冒充论文随机猜测。
top15是严格`conf > quantile(.85)`，并列阈值使实际比例不一定15%；全部保留数已记录。
request3发布/仅mask时10次identity回退；搜索两分支回退为0但mAA仍0：PnP返回不等于正确。
request0也未被这两项修复；不计算挑选样本的benchmark平均，不覆盖原100请求结果。

[作者Issue #76回复](https://github.com/facebookresearch/fast3r/issues/76#issuecomment-2842609184)
曾怀疑portrait处理并回忆强制landscape裁剪。这里两个低分probe为portrait、高分为landscape，
只是线索，不是本机误差因果证明；此前只改变输出transpose的单样本诊断未修复问题。
下一步检查输入裁剪几何/PnP坐标一致性，不仅凭输出尺寸512×384认定满足作者约定。
新结果`results/co3d_candidate_pnp_diagnostic_v1_20261003.json`及独立证明
`results/co3d_candidate_pnp_diagnostic_verified_20261003.json`写入Notebook真实分析。
没有新增正式Table1成绩、训练或整篇完成声明。

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

### 11:21续接：同一次前向的方向诊断（§4.2/Table 1前提）

新scripts/diagnose_co3d_pose_orientation.py只运行一次已归档draw0的公开HF前向，
同一global点图/置信度分别走发布版landscape校正和raw portrait不转置。保持GT、seed、
focal函数、conf>1 PnP、45pairs与metric一致，检查分支未修改原tensor哈希。GT不供focal/PnP。
方向改变后focal重估，故不是固定focal的独立实验；替代分支只作诊断，不替代发布流程。

| 同次前向分支 | focal(px) | RRA@30 | RTA@30 | mAA@30 | PnP失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 发布版方向校正 | 27.9829 | 0% | 13.3333% | 0% | 0/10 |
| raw portrait不转置 | 553.6604 | 2.2222% | 11.1111% | 0% | 0/10 |

发布版分支所有指标与旧报告逐项完全相同，原低分未覆盖。取消转置仅1/45旋转对落入30°，
mAA仍0，因此不支持仅取消转置就解决此样本低分，也不证明全场景同因。没有用GT focal
凑分，不能把这个诊断分数作为Table 1新成绩。结果results/co3d_orientation_diagnostic_20261002.json
已重新verify-only核验保存poses和45pair误差，无推理重跑。145项离线测试通过；Notebook
新增1个真实分析单元、原21个未改。下一步继续预测坐标/首视图global-vs-local focal与PnP
受控接线诊断，保留所有基线；连续100@共享状态及作者划分/权重对应仍未完成。

### 11:51续接：RE10K补缺来源扫描完成，未补齐（§4.2/Table 1前提）

fast3r-re10k-missing-candidates.service已正常退出0/inactive/dead，无剩余扫描进程。
固定公开revision ea8d2427de59817b2f66f17b26276339841eb142的test.tar.gz完整流式读取
205,763,619,478压缩字节，scanner检查gzip尾部/CRC且SHA256与已审计LFS值一致：
f6055cd8ea1ccce642482ca623f98c21af1c449210760d23d78a43b09e546523。
整包未落盘，数据和历史结果未删除；完成证据见results/re10k_missing_source_completion_20261002.json。

按固定tar路径/PNG命名协议识别4137个源场景，76缺失ID均未观测，暂存0帧、补齐0场景。
未对被忽略的tar路径做完整额外路径审计；结论只限此已核对路径协议，不称所有来源永久不可用。
完整报告results/re10k_missing_candidates_full_source.json与独立一致性/GT核验
results/re10k_missing_full_source_verified_20261002.json已归档。独立脚本复核原规定split、
76缺失集合、官方GT SHA及14163条camera记录；网络0bytes，未再次下载/独立重hash205GB。
不要把scanner全流SHA证据和独立重新读完整archive混为一谈。

现有准备仍1756/1832场景、265447帧，不能以交集冒充Table 1正式成绩；没有新增图片几何或
位姿指标。148项离线测试通过，Notebook新增真实源扫描分析单元、原22个分析单元未改。
该来源扫描已结束，不重复启动/扫描此205GB源；后续继续公开补缺来源和CO3D受控诊断。

### 12:28续接：同次前向焦距来源诊断（§4.2/Table 1前提）

scripts/diagnose_co3d_pose_focal.py在固定draw0运行一次公开HF前向，将方向与预测焦距来源
组成2×2诊断。同方向下global PnP点图、conf>1掩码、100次迭代和逐视图seed保持相同；
只替换首视图预测global/未对齐local点图的焦距估计。GT不用于focal/PnP，但输入裁剪仍使用
已归档相机信息。四个来源tensor哈希在评测前后不变，重复帧和全部45pairs保留。

| 分支 | focal(px) | RRA@30 | RTA@30 | mAA@30 | PnP失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 发布方向/global焦距 | 27.9829 | 0% | 13.3333% | 0% | 0/10 |
| 发布方向/local焦距 | 14.2643 | 0% | 15.5556% | 0% | 0/10 |
| raw方向/global焦距 | 553.6604 | 2.2222% | 11.1111% | 0% | 0/10 |
| raw方向/local焦距 | 488.1309 | 2.2222% | 8.8889% | 0% | 0/10 |

两组global焦距分支逐项等于已归档方向诊断；发布方向也逐项等于历史smoke。
local焦距分支不是发布版aligned-local方法，也不是Table 5消融，不升级为正式协议。
四组mAA仍0，替换焦距来源未解决此样本旋转低分；不能推广为全场景根因或调参选优成绩。
结果results/co3d_focal_diagnostic_20261002.json独立保存，旧报告/数据未覆盖。
保存poses/GT重新计算45pair指标通过；151项离线测试通过。Notebook新增真实分析单元，
原23个分析单元保持不变。没有全量CO3D位姿成绩，也没有新增训练结果。

接续：检查预测点图坐标与实际aligned-local首视图接线，随后实现连续100@共享
invalidate/scene-tracker/补采RNG的断点与重放。RE10K仍缺76/1832场景，已完成205GB源
扫描不重复；其他公开补缺来源仍可审查。作者规定划分/权重对应及独立训练保持未完成。
宿主诊断退出0，无新后台GPU任务；下次不要重复本次前向，只读复核命令：
PYTHONPATH=.:scripts python scripts/diagnose_co3d_pose_focal.py --verify-only
现有heartbeat保留以接续，不声称整篇复现完成。

### 新检查点：aligned-local焦距与采样边界恢复（§4.2/Table 1前提）

固定draw0的一次公开HF前向中，新增发布代码local→global相似变换后的首视图焦距分支。
使用alignment percentile0，无GT valid_mask（所有预测像素可参与）；同方向的global
PnP点/掩码/seed不变。这个掩码条件不同于有GT有效深度的评测，仍是诊断，不替换正式协议。

| 分支 | focal(px) | RRA@30 | RTA@30 | mAA@30 | PnP失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 发布方向/global焦距 | 27.9829 | 0% | 13.3333% | 0% | 0/10 |
| 发布方向/aligned-local焦距 | 24.9949 | 0% | 31.1111% | 0% | 0/10 |
| raw方向/global焦距 | 553.6604 | 2.2222% | 11.1111% | 0% | 0/10 |
| raw方向/aligned-local焦距 | 498.6280 | 6.6667% | 11.1111% | 0.6452% | 0/10 |

发布方向RTA改善但RRA/mAA仍0；raw替代分支不能据此升级为论文方法或选优成绩。
两组global基线与历史结果逐项相同，原报告保留；保存poses重新计算全部45pairs通过。
新独立结果results/co3d_aligned_focal_diagnostic_20261002.json，无新全量Table 1成绩。

新增scripts/co3d_sampling_state.py：仅单worker完整请求边界，JSON保存Python补采RNG、
dataset PCG64 RNG、invalidate、invalid_scene_tracker及游标，绑定原候选池、组合、wrapper
映射和调用者协议；发生身份/结构变化时恢复报错，结构检查通过前不修改当前状态。
合成100请求在第37请求后保存恢复，余63请求及最终状态逐项相同，保留1个无效场景、
23个无效帧标记；results/co3d_sampling_resume_synthetic_20261002.json记录边界state与哈希。
这是合成无效深度fixture验证，未读真实RGB/深度、未运行模型，不是实际连续100@评测。
不保存网络/GPU RNG或多worker状态；下一阶段runner还需结果+state一致提交及失败事务恢复。

159项离线测试通过，Notebook新增真实分析输出，原24个分析单元不变。未删除数据、未覆盖
历史报告。已退出本次GPU诊断，无新后台数据任务。只读复核：
PYTHONPATH=.:scripts python scripts/diagnose_co3d_pose_aligned_focal.py --verify-only
PYTHONPATH=.:scripts python scripts/verify_co3d_sampling_resume.py --verify-only

接续：把边界state接入真实100@单worker加载/评测runner，绑定实际wrapper映射与code/data，
隔离加载器补采RNG和网络seed；先测试中断事务/错误不跳过，再运行有界数据准备与评测。
原LazyPreparer processed预算512MiB仍须重新核算，不能静默扩大或删除缓存规避预算。
作者规定划分与权重对应未确认；RE10K仍1756/1832、缺76，已完成205GB扫描不重复。
独立训练、全量位姿等仍未完成，现有heartbeat保留；不称整篇论文复现成功。

### 新检查点：真实CO3D连续准备已启动（§4.2/Table 1数据前提）

scripts/prepare_co3d_continuous.py接入原41类/2011轨迹候选池、公开100@ wrapper epoch0
映射（非作者划分等价声明），单worker顺序加载，不每请求清空invalidate/scene tracker。
初始Python补采seed=42+首个base index，dataset seed777；无网络模型seed干扰，因为本阶段
不运行模型。返回结果与after-state同一JSON事务持久化，fsync后排他link提交，不覆盖历史。
中断只接续完整事务前缀；临时文件不算完成，缺号/状态链损坏/身份变化/CRC或IO错误报错。
162项离线测试通过，包括完整事务/禁止覆盖、断链、未提交临时文件恢复测试。

第1请求base12303633、bicycle/374_41967_84033，10views/9unique，真实原加载输入
tensor、相机GT、RNG与已归档draw0逐项相等。独立只读重放还确认共享after-state完全相等，
网络0bytes，不重复GPU推理。小证据results/co3d_continuous_prefix1_verified_20261002.json
仅代表1/100请求；不是100完成或新增RRA/RTA/mAA。Notebook新增真实首请求分析。
本地results/co3d_continuous_prepare_20261002/保存不可变initial及逐请求恢复日志，
不提交这些可能增长的事务文件、数据、权重或大日志；Git只提交小核验JSON及脚本。

已启动宿主用户级服务fast3r-co3d-continuous-prepare.service（启动PID30700），
日志results/co3d_continuous_prepare.log，从第2请求接续prepare；不占GPU。
raw cache上限2GiB、processed512MiB、下载/预处理空闲至少1GiB保持不变，未扩大或清理缓存。
启动快照服务active/running，空间约39.7GiB；该状态不是后续检查时的当前保证。
健康运行不重复启动；下一次检查user service/日志/完整事务数量，失败先诊断，不跳过请求。
恢复命令（仅确认已退出后执行，不改已绑定runner代码）：
systemd-run --user --unit=fast3r-co3d-continuous-prepare --property=WorkingDirectory=/home/yyz/fast3r --property=StandardOutput=append:/home/yyz/fast3r/results/co3d_continuous_prepare.log --property=StandardError=append:/home/yyz/fast3r/results/co3d_continuous_prepare.log /bin/bash /home/yyz/fast3r/scripts/queue_co3d_continuous_prepare.sh
只读前缀验证：PYTHONPATH=.:scripts python scripts/verify_co3d_continuous_prefix.py
只读事务检查：PYTHONPATH=.:scripts python scripts/prepare_co3d_continuous.py --verify-only

接续：完整准备后先对100请求逐项只读重放、核验成员SHA/输入GT/共享状态，再接独立GPU
评测runner，隔离网络seed与采样RNG，逐项结果+恢复事务不能丢失失败场景。
100@是100请求，不是全部2011轨迹；当前仍是候选划分的明确适配，不能冒充Table 1正式成绩。
RE10K缺76及独立权重/训练等未完成；不重复旧205GB扫描，现有heartbeat继续接续。

### 当前失败检查点：第4请求正Inf深度（§4.2/Table 1前提）

宿主user service fast3r-co3d-continuous-prepare.service已failed/退出1、MainPID0，
不是仍下载/排队。完整事务3/100（bicycle、motorcycle、cup），未提交第4请求；
三请求的真实input tensor/GT/RNG、shared after-state全部只读重放一致，网络0bytes。
证据results/co3d_continuous_prefix3_verified_20261002.json；前三请求和原raw数据保留。

第4请求base12089587，stopsign/599_92267_182752/images/frame000081.jpg触发
process_frame_allow_zero原始有限性守卫。5个float16正Inf（bits31744），无NaN/负值；
RGB/深度/掩码及annotation尺寸1906×1072一致、scale_adjustment1、掩码有限正常。
三个raw成员保存的CRC/SHA均核验通过，不将这认作已证明的下载损坏，不盲目重下载。
按既有参考裁剪/resize后maximum仍Inf，故当前守卫拒绝，并非GPU/额度/空间不足。

只读诊断scripts/diagnose_co3d_nonfinite_depth.py在内存中绕过有限性拒绝来观察算术，
不修改live预处理/loader、raw文件或事务。参考depth/max*65535→uint16在本NumPy1.26.4下
全零，原loader的np.nan_to_num(maximum)乘此量化深度也全零。该算术结果说明原加载器
可能按已有全零深度无效帧规则补采，不支持把Inf像素手动补零或静默删帧/换scene。
尚未完成完整参考预处理输出文件+原loader+真实连续补采的端到端等价验证。
小证据results/co3d_nonfinite_depth_failure_20261002.json，显式retry_started=false；
目前保持失败服务停止，未实施语义修复或重试。164项离线测试通过，Notebook保存真实输出。

下一步：做独立版本化处理，保留参考量化/NPZ语义（Inf审计用字符串，不伪装成有限深度），
验证全零帧经原loader返回None并记invalidate，连续补采状态/前三个历史input完全等价。
证明安全明确后才迁移新协议/输出目录并从正确边界恢复，不能直接改已绑定旧runner导致
恢复身份不一致，不能覆盖旧历史。当前不需要用户提供账户或预算；并未穷尽安全修复，
heartbeat保留，下一次推进上述等价验证，不重复通知相同失败或重跑旧前向。
整篇未完成，无新位姿成绩；下一次不要用旧active/PID30700快照判断服务还在运行。

### CO3D正Inf参考等价与v2安全恢复（§4.2/Table 1，2026-10-02）

新增真实失败帧验证：RGB/量化深度/掩码与固定参考prepare_sequences输出逐字节一致，
NPZ数组及dtype一致，maximum_depth仍为正Inf；审计JSON用positive_infinity字符串。
没有先把Inf像素补零，没有删帧/改候选池。参考量化在本运行时得到全零uint16，
原Co3d_Multiview加载参考文件确实返回None并设invalidate；严格加载v2文件行为相同。
离线证明禁止HttpRangeFile/RecordedRanges网络构造，实际网络0、模型前向0。
证据results/co3d_inf_reference_v2_offline_verified_20261002.json。
同轮早期输出results/co3d_inf_reference_v2_verified_20261002.json保留为历史试验记录；
正式恢复绑定使用上述offline版本及其代码SHA，不依赖早期输出。

独立scripts/co3d_lazy_dataset_v2.py、data/co3d_lazy_v2_processed与v2事务目录，
原v1处理器/事务/数据/失败日志保持不变；NaN/负深度/IO/CRC异常仍硬失败。
从头重放前三请求（仅准备，没有网络前向）后，input tensor SHA、GT、rng标记、
load_trace、pool_attempts、全部共享after-state除新协议identity之外逐项等于v1；
原raw成员身份相同，RGB/深度/掩码字节及NPZ数组/dtype一致。
证据results/co3d_v2_prefix3_migration_verified_20261002.json；verify-only也通过3/100。
恢复不是把旧状态identity改名，而是新协议独立重放后写新事务。

已启动宿主用户级fast3r-co3d-continuous-prepare-v2.service，从第4请求继续。
日志results/co3d_continuous_prepare_v2.log，事务results/co3d_continuous_prepare_v2_20261002/，
启动前空间36,053,286,912bytes。raw2GiB、processed512MiB、至少1GiB预留不变；
后台只准备数据，不占GPU、不运行模型，不保证100请求已完成。
旧fast3r-co3d-continuous-prepare.service仍failed且MainPID0，不重启旧流程。
恢复前必须重新核实宿主服务/进程/空间；健康运行不重复启动，不使用本段启动快照当当前状态。

只读恢复检查：PYTHONPATH=.:scripts python scripts/prepare_co3d_continuous_v2.py --verify-only
迁移证明复核：PYTHONPATH=.:scripts python scripts/verify_co3d_v2_migration.py
实际Inf证明复核：PYTHONPATH=.:scripts python scripts/verify_co3d_inf_reference_v2.py
恢复命令（仅已退出且安全可恢复时）：
systemd-run --user --unit=fast3r-co3d-continuous-prepare-v2 --property=WorkingDirectory=/home/yyz/fast3r --property=StandardOutput=append:/home/yyz/fast3r/results/co3d_continuous_prepare_v2.log --property=StandardError=append:/home/yyz/fast3r/results/co3d_continuous_prepare_v2.log /bin/bash /home/yyz/fast3r/scripts/queue_co3d_continuous_prepare_v2.sh

本轮173项离线unittest通过（包含导入fixture测试），Notebook追加分析单元并宿主执行。
下一步检查第4请求真实补采trace及100请求完整事务；完成后独立全前缀重放，再接GPU评测。
当前只证明单失败帧参考等价和三请求迁移，不提前声称后续全部帧/请求等价。
CO3D作者划分/公开权重与论文协议等价仍未最终确认；100@不是2011轨迹全量。
RE10K仍缺规定76个RGB场景，不重复205GB来源扫描；独立权重/训练实验未完成。
保持heartbeat续接。整篇未完成，本轮没有新增Table 1正式指标。

**本轮最新状态覆盖上述启动快照**：v2服务已failed/退出1、MainPID0；仍是3/100完整事务。
离线恢复第三请求after-state并重放第4请求，trace明确frame81为zero_masked_depth_after_crop，
接着frame42为hard_error。新帧有900个正Inf、3个负深度、无NaN；RGB/深度/mask及GT尺寸
仍1906×1072一致、scale1，mask有限[0,0.996078]，缓存成员CRC/SHA通过。
证据results/co3d_v2_request4_failure_20261002.json，禁止网络构造、网络0、前向0。
这次失败由负深度守卫触发；正Inf等价证明不能自动推广到负值，未放宽守卫或跳过frame42。
下一次先核实负值的float16位模式/位置和固定参考真实输出，再考虑独立v3版本；
必须保留v1/v2事务与成果，证明后续原加载器处理/补采语义、重放前缀，不能反复重启同一失败v2。
当前不需要用户账户/预算或清理数据，安全诊断尚可继续，heartbeat保持。

### §4.2/Table 1：signed-Inf参考验证与v3恢复（2026-10-02）

frame42负值位模式64512/62917/62940对应−Inf/−23632/−24000；900个正Inf、无NaN。
真实参考处理不修补这些值，仍保存全零uint16深度和+Inf maximum_depth；
RGB/depth/mask字节、NPZ数组/dtype与独立v3完全一致，原/严格加载器均返回None并invalidate。
证据results/co3d_negative_reference_diagnostic_20261002.json（仅诊断）、
results/co3d_signed_inf_reference_v3_verified_20261002.json（实际v3文件/原加载器验证）。
v3只允许负原始值出现在processed最大值+Inf且参考量化全零的路径；
NaN、有限最大值的负深度、形状/IO/CRC异常仍硬失败，原数据不改、不删帧/换scene。
v1/v2代码、事务和结果冻结；v3独立root/identity从头重放前三请求，
网络输入tensor/GT/trace/共享after-state除新identity逐项等于v1，modalities/NPZ相同。
证据results/co3d_v3_prefix3_migration_verified_20261002.json。176项离线测试通过。
独立用户服务fast3r-co3d-continuous-prepare-v3.service已从第4请求恢复输入准备，
日志results/co3d_continuous_prepare_v3.log；启动状态不是实时完成保证，后续须宿主核实。
本轮仍没有Table 1正式位姿分数；100@候选请求不等于2011轨迹全量，作者划分等价未证实。
Notebook仅追加此阶段真实证据分析；旧实验/权重/结果不覆盖，RE10K缺76及训练仍未完成。

**最新宿主状态覆盖v3启动快照**：fast3r-co3d-continuous-prepare-v3.service已failed/退出1、
MainPID0，完整事务仍3/100。第4请求现已走过8个零深度候选，按原_get_views规则转到
stopsign/249_26596_53531（不是人工换scene，不等于验证首场景202帧全部无效）。
新轨迹frame81有效、frame54零深度、frame45触发finite-negative守卫；无NaN/正Inf，
4个有限负值−8296/−2180/−1510/−5176，raw finite max62304；尺寸940×532与GT一致。
缓存CRC/SHA通过，离线边界重放网络0/模型0；未提交失败请求。
证据results/co3d_v3_request4_failure_20261002.json。signed-Inf证明不能推广到有限最大值，
未放宽v3。下一次先执行frame45固定参考真实处理并对比NPZ/量化值/原loader，
重点确认负值uint16转换及background/crop后有效深度；不先补零、删值、跳帧或启动同一v3。
需要新适配时必须版本化并证明完整前缀/真实第4请求采样等价。heartbeat保持，无用户动作要求。

### §4.2/Table 1：有限负深度参考语义与v4恢复（2026-10-02）

frame45的4个负值在原参考裁剪/resize后仍存在；depth/max*65535→uint16使4个值成为
正量化值。固定参考未补零/删值。processed最大61952，原loader经mask/crop得到有效视图，
12985个正深度、loaded最大166.37753（只记录实际数值，不解释为论文性能/物理真值）。
RGB/depth/mask字节、NPZ数组/dtype及strict/原loader图像、depth/K/pose完全一致。
诊断results/co3d_finite_negative_reference_diagnostic_20261002.json；实际v4文件证明
results/co3d_finite_signed_reference_v4_verified_20261002.json。两者网络0、模型前向0。
v4保留固定参考有符号有限值量化，不先修补负像素；记录raw/processed负值和量化正值计数。
NaN、负processed最大、有限最大值下−Inf、shape/IO/CRC错误仍硬失败，未经证明不放行。
独立v4 root/identity和事务；复用冻结v3 loader/metadata/budget，不编辑v1–v3代码或结果。
前三真实请求输入tensor/GT/trace/共享状态除新identity外等于v1，modality/NPZ相同：
results/co3d_v4_prefix3_migration_verified_20261002.json。180项离线unittest通过。
fast3r-co3d-continuous-prepare-v4.service从第4请求恢复，日志results/co3d_continuous_prepare_v4.log。
此为启动快照，下一次检查宿主最新状态；无新正式Table 1位姿分数，整篇仍未完成。
候选100@与作者完整协议等价未最终证实，RE10K缺76、训练及独立权重实验仍未完成。

第4请求已完整提交并独立重放验证，不再停在旧3/100边界：
results/co3d_v4_prefix_snapshot_20261002T1200_verified.json核验4/100完整事务，
input tensors/GT/rng、load_trace、pool_attempts和每步共享after-state全部一致，网络0/前向0。
request3 base12089587：原_get_views经首轨迹8个零深度候选后重试到249_26596_53531，
按原补采规则返回10视图/7个不同frame；不是人工替换场景，不等于审计首轨迹全部202帧。
后台继续后续准备，此4请求证据是不可变快照，不是100已完成或最新实时数量。
下一次从v4宿主服务/日志/事务数量接续，不重放或重复归档这份4请求快照。

### CO3D候选100请求位姿评测与完整核验（§4.2 / Table 1，2026-10-03）

候选输入v4的100/100事务全部完成并通过全前缀只读重放，输入tensor、GT、RNG、
补采trace及共享状态一致。GPU评测v3于05:32:52正常退出0，全部100请求/1000输入视角、
4500个相机对已保存；独立核验逐条连接prepared事务、RGB tensor哈希、GT及checkpoint，
从保存的c2w重新计算全部pair误差、逐请求指标和两个汇总口径，均通过。
完整本地报告为`results/co3d_pose_100_seed42_adaptation_v3.json`；
可提交的小摘要为[`results/co3d_pose_100_seed42_verified_summary_20261003.json`](results/co3d_pose_100_seed42_verified_summary_20261003.json)。
全前缀证据为`results/co3d_v4_prefix_full_20261003.json`。

| 指标（百分比，越高越好） | 候选100请求实测 | 论文Table 1 Fast3R参考 |
| --- | ---: | ---: |
| RRA@5 | 28.0444 | 90.2 |
| RRA@15 | 31.2889 | 96.2 |
| RTA@5 | 21.9778 | 68.2 |
| RTA@15 | 28.5778 | 81.6 |
| mAA@30 | 23.8007 | 75.0 |

论文参考来自[arXiv v2 Table 1](https://arxiv.org/html/2501.13928v2#S4.SS2)，
只是参照；作者划分和该公开checkpoint对应论文哪组训练权重仍未确认，
不能把候选适配成绩作为正式Table 1复现。额外RRA@30/RTA@30为35.2889%/35.7111%。
聚合先对每请求45pair计算，再平均100请求；全4500pair池化结果在浮点舍入范围内一致。

实际返回99条轨迹、38类物体、884个唯一RGB帧；64请求有重复视角，共160个重复相机对。
每请求10个返回视角都保留，重复零基线pair沿用发布版指标处理，未删除以提高成绩。
8请求的发布版首视图global焦距估计为0，80/1000视角PnP失败，均以显式identity回退保留。
v2在第4请求因过严的零焦距门槛停下，原3条结果/日志保留；v3按公开`fast_pnp`处理0焦距，
非有限/负焦距和非有限预测仍硬失败。未用GT焦距替换或丢弃低分请求。
候选结果明显低于论文参考，但现有证据不能把全部差距归因于单一因素。

本机入口仅给网络`img/true_shape`，预测global点图→发布版首视图焦距估计→
`conf>1`的RANSAC-PnP（100迭代、固定逐视角OpenCV seed）→c2w→全部相对相机对。
GT/深度/mask/K参与已核验的数据准备，未输入网络或预测焦距/PnP。
论文§4.2描述随机焦距猜测和top15%置信度，而发布版当前入口使用上述估计器与`conf>1`；
此差异需要受控实验，尚未证明是误差根因。采用16-mixed、DPT chunk2、
每请求模型seed42+index并独立恢复采样RNG，均记录为本机适配。

Notebook新增分析单元读取小摘要、核对完整报告哈希及逐请求重算证据，保存真实对照表、
失败统计和科学图`results/figures/co3d_candidate100_pose.png`。旧实验输出保留。
只读核验命令：`PYTHONPATH=.:scripts python scripts/verify_co3d_candidate_pose_report.py`。

下一步按协议审计顺序核对作者CO3D processed split/100@身份、公开权重实验对应关系，
并设计固定输入下发布入口与论文PnP描述的单变量诊断。RE10K仍缺76个规定场景；
已归档205GB来源扫描不重复执行。完整训练、独立训练消融、附录BA/splatting/depth
仍未完成，不因这一候选成绩而勾选整篇完成。

### 51 类 CO3D / 1000 请求空间预算审计（2026-10-03）

本轮只做元数据与预算，不下载新 RGB/depth/mask，不运行模型或 PnP。官方候选元数据确实包含 **51 类、2511 条轨迹、498757 帧**；此前的 seen41 候选是 **41 类、2011 条轨迹、399204 帧**。两者在共同 41 类上逐项一致，新增 10 类为 `ball, book, couch, frisbee, hotdog, kite, remote, sandwich, skateboard, suitcase`，共新增 500 条轨迹/99553 帧。51 个官方 metadata ZIP 当前合计 1315929722 bytes。

新增 `scripts/audit_co3d_51_1000_protocol_budget.py` 生成固定 seed=42 的 1000 条、每条 10 视角的元数据请求计划（计划摘要 SHA256=`34847924f082fd4536fe710a688cb82fda684f2692597a785289d3f77c9a7933`，10000 个唯一 frame slot），并记录磁盘余量与懒加载边界：raw 2 GiB、processed 512 MiB、至少 1 GiB reserve，当前审计时可用 33524957184 bytes。此计划是工程断点，不是作者 1000 请求身份，也不是 Table 1 成绩；`network_bytes_transferred=0`、`model_forward_count=0`、`formal_Table1_result=false`，GT/RGB/depth 完整性和公开权重映射仍未证实。若未来真实懒加载超过边界，必须 fail-closed 保存断点，不能静默驱逐或宣称完成。

历史提案证据为`results/co3d_51_1000_protocol_budget_20261003.json`（原JSON快照33527140352bytes保持不变）。当时4项新测试、200项离线测试通过，但旧手工输出不是精确Notebook单元的宿主执行。当前`paper-co3d-51-budget`只读取历史提案并明确拒绝将其用于评测，已与新的源采样单元一起真实宿主Jupyter执行；没有GPU前向。本轮新增源采样审计见顶部，不将历史缓存上限当成能装下的证明。
