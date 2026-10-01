# Fast3R 续接检查点（2026-10-02，Asia/Shanghai）

用户已授权继续整篇论文复现、Notebook归档及逐步推送GitHub，并要求额度刷新后接续。
分支 `local-demo`，远端 `myfork`。本文件是恢复入口，不是“整篇已完成”的声明。

## 已归档，不重复运行

- `96400ae`：NRGBD9场景配对与7-Scenes数据准备。
- `1e3b59f`：7-Scenes全部7类/18测试轨迹、850视角配对评测与Notebook。
- `d0e87a9`：权重/协议审计、Table1数据预检查。
- DTU22场景、NRGBD9场景、7-Scenes18轨迹已有真实报告；数值未匹配论文。
  未做完整重新训练、CO3D/RE10K正式位姿benchmark、独立训练消融。

## 本轮新增

项目分区扩至约251G，初查空闲122G；下载前另一次检查117G。
这些都是2026-10-02的快照，每次接续仍须重新 `df -hT .`。

1. 官方RE10K元数据：`fast3r-re10k-metadata.service` 正常退出0，日志
   `results/re10k_metadata_pipeline.log` 于01:55:16记录完成。
   `results/re10k_metadata_manifest.json`核验1832规定ID、各文件SHA256与相机记录。
   `data/RealEstate10K/test`约65MiB；官方archive752332631bytes、MD5
   `7dadaf85e559bc93ce75378e31064492`，SHA256
   `af18e4d560ee9f73120128e53a56da215e118f3fb52018c14a66aa0f8ed691db`。
   再跑 `scripts/prepare_re10k_metadata.py` 会校验已有清单与文件，跳过网络下载。
2. 新 `scripts/fast3r_hf_re10k_pose_eval.py`：单卡HF入口，512×288、seed42、10视角、
   DPT chunk2、16-mixed；规定全量预检查、输入/权重/代码哈希、逐scene JSON断点。
   GT不输入网络或focal/PnP。官方实际PnP mask为conf>1，不是percentile85。
   顺序PnP固定OpenCV seed是明确适配；保留并计数失败identity fallback，不删除失败scene。
   断点重算已保存poses的指标，改变协议拒绝复用；旧最终报告不覆盖。
   当前实际dry-run因1832个RGB目录均缺失而拒绝执行，尚无Table1分数。
3. RGB归档下载服务：`fast3r-re10k-rgb-archive.service`，02:02:47启动。
   日志 `results/re10k_rgb_archive.log`；临时归档 `data/re10k_test_only.zip.part`，
   作者镜像 `http://schadenfreude.csail.mit.edu:8000/re10k_test_only.zip`，55,604,889,849bytes。
   HEAD/少量ZIP头已核实；作者loader要求JPEG解码为360×640。
   完整镜像的1832覆盖、原图尺寸与GT相符仍未验证，不能标成ready。
   该服务器不支持Range；重启会从HTTP开头重新传输并逐块核对已有前缀，之后才追加，
   不重复写入前缀。消耗额外网络流量但不抹掉下载进度。
   下载完成才写 `results/re10k_rgb_archive_manifest.json`，只代表归档已下载。
4. 首个完整chunk已安全解析并核验CRC：17个clip所有图像均360×640，
   其中8个规定测试ID的timestamp/camera与官方TXT核对通过，最大相机绝对差约2.38e-7。
   `results/re10k_rgb_probe_20261002.json`只是有界format probe，不是全量coverage。
   独立smoke目录保存其中一个真实clip的全部105候选帧，单卡随机10视角接线运行成功，
   PnP失败0/10，45对相机指标有限。`results/re10k_smoke_seed42.json`明确
   `benchmark_scope=adaptation_smoke`，不作为Table1全量成绩，也不能用来宣称达到论文指标。
   新入口默认prescribed_full时拒绝非原1832-ID清单。

验证命令：`PYTHONPATH=. /home/yyz/miniconda3/envs/fast3r/bin/python -m unittest discover -s scripts -p 'test_*.py'`。
当前47项测试通过；真实smoke断点再运行未加载模型，输出
`results/re10k_smoke_resume_verified.json`与首次最终JSON逐字节相同。
Notebook分析单元通过宿主Jupyter执行，再用apply_patch保存真实输出；不重跑旧GPUbenchmark。

## 下一次接续顺序

### 05:21归档完成与覆盖核验检查点（2026-10-02）

- 下载服务05:10:00正常结束0，归档55,604,889,849bytes、SHA256
  `ce351771c966fb25ef41efc561a313ef40607c9aa8ea904ed8d582b361408097`。
  下载及03:52前缀恢复阶段已完成，不再重启下载。
- `results/re10k_rgb_index_audit.json`：镜像7286个ID中规定清单覆盖1756/1832、
  缺76个，完整缺失ID和官方YouTube URL已记录。不能用1756交集冒充正式Table1。
- `scripts/prepare_re10k_rgb_from_archive.py`逐块safe weights-only读取、CRC/SHA、
  官方全部timestamp与camera核验、JPEG解码；保留每个covered clip全部官方候选帧。
  原图存在640×338等尺寸，不能强制改成640×360。位姿入口按实际宽高换算归一化K。
  尺寸差异记录在逐scene inventory，原图不拉伸/补边，裁剪协议等价性仍不声称成立。
  首次固定尺寸检查安全停止；现恢复版本保留旧33份inventory及JPEG，另写v2清单。
- 宿主服务`fast3r-re10k-rgb-prepare.service`已恢复，MainPID22629；
  日志`results/re10k_rgb_prepare.log`，结果`results/re10k_rgb_prepared_manifest.json`。
  05:36:54服务正常结束0，1756个covered场景共265447帧全部GT/CRC/解码通过。
  独立`verify_re10k_rgb_prepared.py`又校验全部265447个已保存RGB长度/SHA、
  原官方GT文件SHA、逐scene inventory与原候选集合，记录
  `results/re10k_rgb_verification_20261002.json`，不能重复启动已结束的准备阶段。
  1753场景为360×640、2场景338×640、1场景272×640（H×W）；
  最大镜像/GT camera绝对差9.51156e-7。58项测试通过，Notebook真实输出已归档。
  结果仍是covered_prepared_split_incomplete，76缺失未补齐，不允许启动正式全量评测。
  若失败先诊断，恢复命令同下载方式，仅替换unit为fast3r-re10k-rgb-prepare、
  shell为scripts/queue_re10k_rgb_prepare.sh；原RGB、旧inventory和历史报告不得删除。
  当前无后台下载/准备/评测，下一步优先继续76缺失来源与CO3D可行性，而非重复报告。
- 缺失来源初查见`results/re10k_missing_source_audit_20261002.json`：另一个HF test
  chunk镜像同样缺76；两个匿名YouTube元数据探测均遇HTTP429，其中一个要求登录，
  不据此认定全部76视频永久不可用，不读取cookies或绕过限制。
  另发现公开`DavidYan2001/RealEstate10K`固定revision
  `ea8d2427de59817b2f66f17b26276339841eb142`的`dataset/test.tar.gz`
  为205,763,619,478bytes。下一步先读来源/使用条款、研究有界流式tar选择与coverage；
  不盲目落盘整个205GB，不把尚未验证的文件当作能补齐76个ID。
  这些可行性工作仍可推进，暂不因镜像缺76/准备排队而要求用户额外存储或删除监控。

### 03:52下载恢复检查点（2026-10-02）

首次服务于03:25:41退出1：HTTP响应提前结束，保留54,818,855,664bytes，
距预期55,604,889,849bytes还缺786,034,185bytes。只确认提前EOF，
未定位具体服务端/网络根因。03:51重新HEAD核对长度与Last-Modified未变；
03:52:39重启同一服务，MainPID20847、active/running。
`results/re10k_rgb_recovery_20261002.json`记录该时间点，不代表下载完成。
恢复先从HTTP开头逐字节验证已有前缀；此期间文件大小不增长正常，
请检查日志`Resume prefix verified`与宿主进程，不将其误判为卡死。
脚本新增每分钟恢复核对进度日志，截断响应保留原文件测试通过，全部48项测试通过。
沿用同一日志追加，不覆盖历史结果；没有正式Table1成绩。

1. `git status --short`、近期历史、本文件和PAPER_REPRODUCTION先确认哪些已归档。
2. 用宿主权限检查服务、进程、GPU与磁盘。下载健康时不重启、不并行重复启动，不通知无变化。
   若下载失败先读日志；安全重试：
   `systemd-run --user --unit=fast3r-re10k-rgb-archive --property=WorkingDirectory=/home/yyz/fast3r /bin/bash /home/yyz/fast3r/scripts/queue_re10k_rgb_archive.sh`。
   仅确认旧服务已结束后重试；若旧失败unit仍占名，先对这个unit执行`systemctl --user reset-failed fast3r-re10k-rgb-archive.service`。
   保留.partial和历史结果，至少预留1GiB。
3. 完整ZIP后审计目录/index：规定1832 ID必须全部可覆盖，不能用交集冒充全量。
   只读取必要torch chunk；用 `torch.load(..., weights_only=True, map_location='cpu')`，
   不使用不受限制pickle；逐成员读取验证CRC，记录实际输入SHA与镜像身份。
   在解包写入前核算峰值预算，逐chunk仅保留规定scene的RGB（不整包解压）。
   逐timestamp与官方TXT核对，核对mirror camera与官方19列、RGB尺寸/裁剪/归一化intrinsics；
   在验证前不认定mirror与论文协议等价，不自动改相机GT“修复”不匹配。
4. RGB齐备：完整 `check_pose_data.py --dataset re10k` 和新runner `--dry-run`。
   可先用独立子集清单/输出做真实GPU接线检查，明确不是正式全量benchmark。
   全量GPU空闲后用独立service启动新runner，固定完整清单与progress目录。
   如需阶段性稀疏存储，必须保留完整原始frame清单并固定原清单采样；不能让稀疏目录
   改变random.sample的候选集。否则保留规定scene的全部可用帧。
5. 每个完整report核验ID集合、per-scene/PnP失败计数/aggregate，Notebook分析单元执行，
   保存真实输出并更新文档，测试后提交推送myfork/local-demo。不得覆盖旧结果。
6. 然后推进CO3D41类物体中未见测试轨迹的选择、预算与公开权重适配，以及资源允许的附录实验。
   独立训练消融仍需要独立权重或训练资源，未做实验不打勾。

## 额度恢复机制与边界

### 05:51公开来源续接检查点（2026-10-02）

- `89b1974`已推送：1756场景265447帧准备/独立核验已完成，不重复。
- 宿主约49GiB空闲、GPU约10900MiB空闲；原archive/prepare服务已正常退出。
  这只是本轮快照，每次仍须宿主只读检查。
- 新`fast3r-re10k-missing-candidates.service`06:03:58启动，
  日志`results/re10k_missing_candidates_stream.log`，
  最终审计`results/re10k_missing_candidates_full_source.json`。
  固定公开205,763,619,478bytes tar.gz来源，HTTP 206真实8MiB前缀探测通过。
  不存205GB归档，仅在`data/RealEstate10K_missing_candidate`暂存76缺失ID的官方timestamp PNG；
  总保存<=16GiB、单图<=8MiB、留1GiB，PNG不转JPEG/不覆盖旧数据。
  源镜像卡仅CC-BY-4.0、没有解码/裁剪详情；整包SHA/GT候选/图片几何均须核验。
  完成报告出现也不自动标full ready、不直接启动formal pose。没有进程重启/源SHA结果前勿称补齐。
  同进程Range故障重试可恢复压缩偏移；重启需从gzip开头重读并核对已存RGB。
- 新`fast3r-co3d-test-metadata.service`06:05:01启动，
  日志`results/co3d_test_metadata.log`，最终`results/co3d_test_selection_manifest.json`。
  只下载51类_000元数据ZIP（每包<=128MiB，固定CO3D官方revision/SHA/CRC），不下载RGB/depth。
  `data/co3d_test_metadata/selected_seqs_test_reconstructed.json`由公开DUSt3R选择推导，
  与作者processed清单等价性仍未确认；类别/序列/候选数量以实际完整报告为准，不猜为41或2050。
  公开fewview_train的test键、quality>0.5、最多50序列、seed42+类别索引，记录set-list顺序。
  首次进程退出1：官方fewview_train JSON解压最大已观测50,989,373bytes，超过初版32MiB读取上限。
  第二次安全复用时确认book_000.zip为86,868,326bytes，超过初版64MiB ZIP上限。
  50个元数据ZIP保留，宿主可用内存约24GiB；ZIP/JSON成员上限调整为128MiB，
  加入解码清单/并发写入的额外空间预算后再安全重跑当前流程；不会放开无界读取。
  同时补充“没有fewview_train时明确空集合”的兼容检查；ZIP/SHA校验复用，不删数据或终止其他进程。
- 元数据服务最终已正常退出0，51类ZIP共1,315,929,722bytes全部官方SHA/CRC通过。
  当前公开默认规则真实得到51非空类别、2511序列、498757候选帧，**不是论文41类**。
  独立`verify_co3d_test_selection.py`再次核验所有ZIP SHA、参考哈希、quality、稳定选序列和帧顺序。
  小结果`results/co3d_test_selection_summary_20261002.json`（约35KB）及
  `results/co3d_test_selection_verified_20261002.json`可提交；约6.34MB全帧索引报告留本地、git忽略。
  下次不重启元数据服务；优先查PoseDiffusion/DUSt3R规定41类列表与Fast3R原选择关系，
  在51→41有证据前不下载这一默认集合的全部图像/深度、不称规定benchmark已准备完成。
- 已纠正“41未见类别”为“41类物体中的未见轨迹”；配置100 @是长度100的随机重采样包装，
  不是全序列集合。不要凭配置100或初版元数据选择宣称CO3D全量论文benchmark完成。
- 新流式HTTP/安全tar/重试/前缀不完整/存储保护及CO3D选择测试通过，共73项离线测试。
  合成测试不是Table1测评。真实8MiB探测仅见1个场景，未见任何缺失ID，不推导全部不可用。

下一次先查RE10K服务/日志；CO3D元数据已结束并独立核验，不重复启动。健康无变化安静。
RE10K全流扫描结束才验证来源LFS SHA及76集合、全部官方timestamps、图像/GT几何；
PNG原文件先保留独立候选。必要时添加格式明确的输入适配并重新验证，不把PNG改后缀冒充JPEG。
CO3D元数据报告完成后核验逐类选序列与官方文件/参考哈希，计算必要RGB/depth/mask成员预算，
研究Range逐成员提取，不落盘全5.5TB，不用single-sequence挑战划分替换。
所有新增小结果/Notebook真实输出验证后归档；仍未做全量Table1或训练消融。

Codex当前聊天heartbeat `fast3r` 已创建并读回核验ACTIVE，每30分钟接续。
本地机器须开机、应用保持运行；额度不足时定时任务也可能不能执行。
这是后续触发时尝试接续，不保证精确重置时刻或无缝恢复。不购买额度、不兑换reset权益。
已启动systemd下载/评测与模型额度独立，但断电/休眠/网络故障仍会中断。
任务提示要求无变化安静，仅新阶段完成、失败或需用户动作才通知。
