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

最新检查点（2026-10-02 05:21，§4.2/Table1）：RE10K55.60GB候选RGB归档已完整下载，
服务正常结束0；索引规定覆盖1756/1832，缺76个，已保存完整缺失ID/官方视频URL。
`fast3r-re10k-rgb-prepare.service`于05:36:54正常退出0，1756场景265447帧的ZIP CRC、
全部GT timestamp/camera与原图已核验，独立全部RGB字节/SHA与候选集合核验也通过；
保存全部候选帧及实际尺寸，不把1756交集当全量，不将部分准备计作正式成绩。
58项测试通过；其余镜像/原URL可行性初查见`results/re10k_missing_source_audit_20261002.json`。
下一步核验准备结果并研究有界流式补齐，尚不宣称全部76永久不可用或必须额外付费/存储。

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

2026-10-02新增§4.2/Table1来源审计：8MiB公开RE10K Range/gzip前缀探测通过，
后台仅暂存76个缺失ID的原PNG候选（最多16GiB、不覆盖已有JPEG），完整来源SHA及几何等价仍待核验。
CO3D仅准备官方元数据ZIP，固定SHA/CRC和公开DUSt3R选择版本，不下载RGB/depth大包。
两个数据任务的服务名、日志与恢复边界见`CONTINUATION.md`；73项离线测试通过，
Notebook记录真实源探测，不冒充新增正式位姿成绩或整篇完成。
CO3D官方51类元数据ZIP/SHA/CRC及独立选择核验现已完成，但公开默认结果为51类、
2511序列、498757候选帧，不是论文41类。后续先查明类别协议，不任意删类凑数，
也不将元数据就绪冒充图像/深度或Table1成绩就绪。

06:21续接已找到PoseDiffusion固定版本41类seen协议，并在原51类候选上过滤得到2011轨迹、
399204候选帧，不改变原始种子/类别索引或帧顺序。类别映射有来源，但原Fast3R逐轨迹与
100@采样等价性仍未证实。真实apple目录Range探测找到9563组RGB/depth/mask，标示6.07GB；
只传目录索引，不验证图像CRC、不声称完成数据。后台41类预算日志及恢复命令见CONTINUATION。
新增分析单元已实际执行；类别与Range安全合成测试纳入84项离线测试，不是新增RRA/RTA/mAA。

随后目录审计遇到banana成员数量保护阈值偏小，已只读诊断并有界修复，保留32MiB目录字节上限。
两个已完成类别经核验迁入新v2断点，不覆盖v1历史或重复网络读取。新恢复分析保存真实输出，
88项离线测试通过；它不是新的正式测评结果。当前v2恢复入口见CONTINUATION。

07:21续接完成全部41类/235个官方ZIP尾部大小预检（只传15.41MB），查明bowl目录超过32MiB。
已改为按每包实测目录预算读取，并在64MiB硬上限内恢复v3；原成员/路径/Range保护保留。
8类已完成目录检查点核验复用，bowl已通过原位置。95项测试通过，新的Notebook分析保存真实输出。
当前请使用CONTINUATION的v3路径；尾部预检不等于图像/CRC/相机GT或正式Table1成绩就绪。

08:21续接：CO3D目录预算与独立验证已完成，41类/2011轨迹/399204组每种成员覆盖；
原始成员297.20GB，约46GiB空闲，按需存储仍待设计，不直接落盘原始全量。
all-valid名义100@追踪仅99轨迹/918唯一帧且58次重复，不冒充实际采样或Table1成绩。
本地相机metadata全候选核验也完成，38869项文件名/annotation编号不等，按官方set-list连接；
car12项大坐标原值和残差保留，float32指标稳定性、图像/深度/crop仍未验证。
Notebook新增预算、源采样与metadata分析；当前续接入口见CONTINUATION。

Neural RGB-D官方9个序列已解压。`bash scripts/queue_nrgbd_reproduction.sh` 会先检查实际目录中的全部RGB/深度帧、非空文件与位姿数量；检查通过即跳过下载和解压，即使ZIP或manifest不存在。只有数据不完整时才恢复下载/解压；已有完整ZIP则复用，无需重下。可运行 `python scripts/check_nrgbd_data.py` 单独检查（这是文件布局/数量检查，不是重新解码PNG或CRC校验）。可选PID参数仅用于数据不完整时等待已有下载进程。后续评测入口支持 `python scripts/fast3r_hf_dtu_eval.py --dataset nrgbd --device cuda --output-json results/nrgbd_seed42_stride40.json`。默认stride40；Table 3比较时将native距离乘100。完整实验状态、限制与余下前提请查阅 [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md)。

2026-10-02 10:51续接已执行CO3D公开HF单样本位姿接线（§4.2/Table 1前提）：
完整原候选池第0个mapped draw，10views/9unique，RGB-only网络输入、global→focal→PnP，
45pairs及重复零基线不删除。11:05:17服务正常退出0，保存c2w指标重算通过，PnP失败0/10。
真实RRA@30=0%、RTA@30=13.3333%、mAA@30=0%，旋转误差50.1264°–171.6512°，
低分保留，不称Table 1全量成绩或论文效果复现。首次新入口方向断言错误已诊断，保留v1
证据并修复后v2重试，没有改发布算法或覆盖历史。144项离线测试通过，Notebook新增2个
报告分析单元有真实输出；下一步方向/focal单变量诊断，连续100@共享状态和训练尚未完成。

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
