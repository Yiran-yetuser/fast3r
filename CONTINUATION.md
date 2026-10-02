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

### 06:21续接归档：CO3D41类有来源候选与后台空间审计

#### 08:21续接：目录预算、采样与相机metadata完成（当前恢复入口）

`731f389`已推送，旧footer/恢复快照不重复归档。宿主CO3D目录进程已结束，日志记录
`ALL 41 DIRECTORY BUDGETS COMPLETE`；完整v3报告及新独立验证已生成。核对41类集合、
2011轨迹、399204组每种成员的期望路径SHA、235包URL/ETag/长度/Range、所有大小汇总和指纹，
8类迁移来源/旧值不变；没有重新传输远端目录。结果`results/co3d_storage_budget_verified_20261002.json`。
原始成员标示297197632174bytes（276.79GiB）：RGB36297726723、depth246352518781、mask14547386670。
当前宿主约46GiB空闲，原始全量落盘不适配；继续按需预处理设计，不清理旧数据或立刻要求再扩容。
空间/相机恢复诊断快照`results/co3d_storage_feasibility_20261002.json`不作为当前实时进程状态。

真实仓库采样方法以组合seed42、dataset seed777、epoch0和全有效stub名义追踪：
100@的100次请求只涉及99轨迹/38类/918唯一帧，58次有重复视角；不是正式图片输入或论文draws确认。
另一个全部2011轨迹适配追踪为18588唯一帧，1149次重复；不替换100@，不作正式全量成绩。
`results/co3d_sampling_trace_20261002.json`保存100次trace与限制，全2011名义清单留data不提交。
真实depth/mask无效会补采或换轨迹，名义清单不能直接当稀疏下载预算/ready状态。

本轮相机metadata审计也完整结束，日志`ALL 41 CAMERA METADATA PREFLIGHTS COMPLETE`，进程不存在。
结果`results/co3d_camera_metadata_audit_v2_20261002.json`覆盖41类/2011轨迹/399204候选annotation：
官方ZIP SHA/CRC、set-list编号/路径、ndc_isotropic、depth.scale_adjustment=1、有限K/c2w与rotation核验通过。
38869项filename编号与annotation编号不同，按官方set-list连接，不按名字猜GT；源数据不修改。
car有12项平移绝对值>1e6，最大8.296149380025549e16；绝对inverse残差16，最大component-scaled残差
6.758143057927225e-16。记录原值和风险，不做静默归一化/删轨迹，尚未证明float32指标稳定。
首次错误的11类v1记录和历史日志保留；当前v2断点`results/co3d_camera_metadata_v2_progress/`。
已结束的`fast3r-co3d-camera-audit.service`不重复启动；queue为`scripts/queue_co3d_camera_metadata.sh`。
上次08:38:10恢复MainPID33430只是历史快照，当前无CO3D审计进程。

下一次先检查RE10K来源扫描（仍健康，最终`results/re10k_missing_candidates_full_source.json`待生成），
只在完整来源SHA/76集合/图片裁剪核验后升级数据状态，不重复启动或删原1756场景。
CO3D下一步：先审计源大坐标在官方float32相对位姿指标中的稳定性；再设计源候选不变的按需Range
缓存和DUSt3R crop/depth处理，按实际有效性跟踪补采/scene重试，不让稀疏目录改变候选池。
没有作者processed清单/具体100@身份仍须标协议适配，不能用99轨迹或2011候选冒充已复现Table1。
当前图像成员CRC/解码、camera NPZ/crop等价性、正式RRA/RTA/mAA均未完成。
111项离线测试通过，Notebook的16个paper分析单元全部已有真实输出；新增单元只读新JSON，未重跑旧GPU实验。

#### 09:21续接：完整候选GT单精度诊断（已归档）

`58abff5`已推送，旧目录/相机metadata不重复运行。本轮新脚本
`scripts/audit_co3d_pose_precision.py`在CPU读取固定metadata并比较float32/float64：
严格先将OpenCV w2c构造成float32再np.linalg.inv，沿用固定DUSt3R顺序；
随后调用公开Fast3R closed_form_inverse与translation_angle，不修改GT/指标、不加载模型。
结果`results/co3d_pose_precision_20261002.json`覆盖41类/2011轨迹/399204帧、
39,624,443个同轨迹候选相机对；按类别sum/max独立核对，通过Notebook新分析单元。

float64直接相机中心相同1181对；公开计算float32零平移1375对，另194对在float64
相对矩阵中非零而float32变零（不能自动解释为真实物理基线丢失）。可定义方向的对数39623068，
3301对方向差>1°，最大89.9804°。为区分近零基线，额外记录
norm(c64_j-c64_i)>1e-6×max(1,norm(c64_i),norm(c64_j))这个**诊断分组**，
39619868对中仍340对方向差>1°、最大5.9845°，变零0对；它不是论文阈值，未用于删正式pair。
GT与自身比较float32平移角有1375对>1°（最大90°），属于零向量/数值计算行为，不是预测误差。
旋转自检最大0.4070°来自公开角函数的近零处理，不静默更换公式。
风险条件命中201轨迹，报告只列前20条，完整诊断行的canonical SHA另存；并非只评测20轨迹。
所有轨迹/源坐标保留，不能宣称“全部float32稳定”、不能把诊断值当RRA/RTA/mAA。
CPU与当前NumPy/Torch版本不证明CUDA/TF32或真实预测稳定；正式runner需保留原metric并显式报告诊断。

117项离线测试通过，Notebook17个paper分析单元都已有真实输出。仅新增的精度分析执行，旧输出不覆盖。
RE10K扫描宿主09:31仍active/running、PID24596，压缩字节130755598920/205763619478，
未生成最终完整来源报告；这是历史快照，后续重新检查，不重启健康进程。
当前宿主约49.35GB可用，保留1GiB；本轮未下载CO3D图像或改变GPU任务。

下一步实际数据推进：实现有界的官方ZIP Range按成员缓存及CRC/解码验证，先小规模真实
RGB/depth/mask→固定DUSt3R crop/depth/K/float32 NPZ接线，再接源候选不变的延迟加载。
保留完整候选池，记录有效mask/depth导致补采和scene retry的实际draws；不把918名义帧当真实ready。
大ZIP全SHA尚不能由成员CRC代替，作者processed清单与100@采样身份仍未确认，明确协议适配。
RE10K全来源扫描完成后先核验SHA/76集合/timestamps/图像几何，再考虑正式1832-ID入口。

#### 09:51续接：十帧真实Range/预处理接线完成（已归档）

`1009cc4`已推送，GT精度诊断不重复。新`co3d_range_cache.py`固定原候选manifest、官方links、
235包footer大小/ETag，按需解析并缓存整类的原候选成员索引；不存整ZIP。
中央目录精确Range经inventory验证，内存重放提取ZipInfo不再次网络读取。
逐成员核对local header/ZIP64尺寸/名称、压缩边界、有界inflate与CRC，SHA后独占写缓存。
单成员32MiB、raw/index总缓存2GiB、写入前至少留1GiB；不自动删旧数据或驱逐已有缓存。
缓存身份/代码变化拒绝复用，已有成员重读长度/SHA；没有完整大ZIP SHA证明。

`scripts/probe_co3d_preprocess.py`明确只取apple/608_95658_192033的前十个原候选帧：
1,2,3,4,5,11,12,13,14,15。30个RGB/depth/mask成员全部CRC/解码通过，原始保存7738425bytes。
首次Range传输91655026bytes包含apple完整中央目录；缓存后再次完整复核网络0bytes，未重做下载。
缓存约25MiB，独立probe目录约5.2MiB（原始reference fixture为hard links，du按遍历可能重复计数）。
只执行固定DUSt3R已审阅的函数定义，不执行远程代码顶层；相同NumPy/PIL/OpenCV运行环境中，
适配输出与原prepare_sequences实际输出十帧JPEG/PNG字节完全相同，NPZ逐数组相同。
原始深度为uint16的float16位模式，先reinterpret再float32，不能当整数/65535读取。
crop主点、半像素K变换、NEAREST深度/掩码、max-depth量化和float32 w2c求逆顺序均保留。
结果`results/co3d_preprocess_probe_20261002.json`；固定参考源码SHA与运行库版本在report中。
预处理衍生代码许可说明见`NOTICES_CO3D_PREPROCESSING.md`，不提交数据或参考副本。

独立`scripts/verify_co3d_preprocess_probe.py`核验raw长度/SHA、processed/reference字节与NPZ数组，
再调用真实Co3d_Multiview._load_view_data。原202帧pool未变，只加载已准备的十帧，不造稀疏split。
十帧均有效，mask后正深度覆盖0.05223–0.05686；这是GT覆盖比例，不是模型accuracy。
第一次检查错误假定只输出512×384；本批近方形图像按公开base逻辑可随机交换方向，
现在保留512×384或384×512并记录(H,W)，不拉伸图像或改GT；补充离线回归测试。
独立结果`results/co3d_preprocess_probe_verified_20261002.json`。尚未运行_get_views补采/scene retries，
未调用模型/全量RRA/RTA/mAA，不能将此probe标为100@或全2011数据ready。
128项离线测试通过，Notebook18个paper分析单元有真实输出，旧输出不覆盖。
两个probe/verification命令都已正常退出，无新增后台CO3D任务；只归档新阶段。

宿主10:08快照：RE10K扫描仍active/running，PID24596，153997066625/205763619478压缩字节，
未生成完整来源报告；空间49315729408bytes可用，GPU约10904MiB空闲。每次接续重新核实。
不要重启健康扫描，不再重复十帧probe通知/提交。必要时本地复核用以上verifier；probe复用网络0。

下一步：将原完整2011/399204候选池与按需Range/参考一致预处理接到严格的lazy loader，
缺下载/CRC/相机文件不能被原loader catch-all误当GT无效而跳过；真实零有效深度按公开规则记录。
固定采样seed、100@映射与候选顺序，记录每次加载、重复补采和scene retry；只称协议适配，
作者原processed清单/100@draws与HF权重对应关系未确认。先真实单样本接线，再考虑规模化。
2GiB缓存是当前探测的保护上限，不证明所有真实采样都能放下；达到边界时先预算，不能自动清数据。
RE10K全流结束后先核验来源SHA/76缺失集合/timestamps/图像几何，再推进1832-ID评测。

#### 10:21续接：严格lazy loader与一个真实映射样本（最新恢复入口）

`3de46a0`已推送，apple十帧接线不重复。新增`co3d_lazy_dataset.py`保留
41类/2011轨迹/399204帧的原候选pool与顺序；processed另存`data/co3d_lazy_processed`，
不生成替代稀疏split，raw/index仍2GiB、processed512MiB上限，每次写保留1GiB，无自动驱逐。
固定本地loader SHA+AST形状，只移除_load_view_data的catch-all，采样组合、jitter、补采、
scene retry、base归一化/3D反投影/landscape transpose直接继承公开方法。
下载/CRC/解码/相机失败立即抛出；裁剪后真正零有效深度仍None并显式trace，
原始全零深度保留max0/quantized0而非损坏报错；合成回归验证零深度补采和场景重试。

`probe_co3d_lazy_sample.py --wrapper-index 0`实际执行100@epoch0第一个映射样本，
base_index=12303633，bicycle/374_41967_84033原133帧pool不变。
顺序144,42,41,151,20,28,52,144,10,1，共10views/9unique，保留重复144。
实际10次读取均有效、1次pool尝试，无真实补采/scene retry；不是整个100@已执行。
组合seed42、dataset seed777、epoch0、Python补采seed42+base_index为声明适配，非作者draws。
base.__getitem__的全部返回值与原Co3d_Multiview独立读取精确相同：img tensor、K/pose、
pts3d、valid_mask、true_shape、rng等。未调用模型。独立`verify_co3d_lazy_sample.py`在独立
fixture执行固定DUSt3R原prepare_sequences，9帧JPEG/PNG字节及NPZ数组一致；核验27个raw成员，
9unique原字节4541640bytes。有效GT比例0.13252–0.47588，不是预测准确率。

首次报告保存因NumPy int64 idx不能JSON化失败，非数据失败；转native int，补回归，使用新v2路径。
6040bytes未完整文件保留于`results/co3d_lazy_serialization_progress/partial_draw0_20261002.json`。
成功`results/co3d_lazy_draw0_v2_20261002.json`、独立核验`results/co3d_lazy_draw0_verified_20261002.json`
和`results/co3d_lazy_serialization_recovery_20261002.json`；成功缓存重试与独立核验网络0bytes。
首轮已下载目录/成员但总传输未完整归档，不把0当初次下载流量。138项离线测试通过，
Notebook新增真实样本分析，旧输出未覆盖。两个前台命令均已结束，无新CO3D服务。
RE10K扫描仍健康；接续重新核实服务/空间/GPU，不重启扫描。
10:37宿主归档前快照：服务active/running、PID24596，173227966215/205763619478压缩字节，
尚无完整来源JSON。分区可用49282215936bytes、GPU10913MiB空闲；raw缓存约48MiB、
新processed及独立reference各约2.1MiB。Notebook19个paper代码单元已执行且原18个完全未改。

下一步：将已核验draw接入公开HF+原focal/PnP/相对位姿metric的单样本smoke，GT只评分，
不输入网络/PnP。记录45pairs（含重复视图的零基线）、PnP失败fallback，明确协议适配，非Table1。
再预算/断点化更多100@draws：每进程重置invalidate/scene tracker和per-request Python seed
适合单样本复核，但不等于原连续100@共享状态；规模化前须保存/重放状态、追踪真实重试。
缓存限额不是完整实际采样容量证明。不要重复apple/bicycle阶段通知/提交。
RE10K全流结束后核验完整SHA/76集合/timestamps/几何，再推进1832-ID入口。

#### 07:21续接：全ZIP尾部预检与v3恢复（历史恢复入口）

`3d5dd0f`已推送，旧banana恢复分析不要重复归档。v2目录预算服务06:59:12退出1：
bowl_001.zip中央目录37,498,281bytes超过32MiB字节预算。此次先只读预检全部41类、
235个数据ZIP尾部，每包最多65557bytes，合计15,405,895bytes；实际最大目录37,498,281bytes，
最大300491成员。`results/co3d_zip_footer_preflight_20261002.json`完整记录每包大小/计数/ETag。
这是footer层面的全部来源预检，不是完整中央目录/成员路径/图像CRC准备完毕。

v3仍保留400000成员上限；目录硬上限64MiB，每包实际读取预算更严格：
footer中的目录大小+65557+128，身份/大小/计数变化拒绝继续；Range/ETag/路径检查不撤销。
预检逐包与汇总、官方链接/协议SHA已核验，95项离线测试通过。
07:29:59恢复同名`fast3r-co3d-seen41-storage.service`，MainPID29975，已确认running。
apple/backpack/banana/baseballbat/baseballglove/bench/bicycle/bottle共8类的v2记录，
通过固定已审计v2代码SHA、输入/路径SHA、逐包统计及新footer身份复核后导入v3，
不重读其中央目录，v1/v2历史均保留。bowl已通过原失败点且完成10098组目录路径核验。

当前queue仍是`scripts/queue_co3d_seen41_storage.sh`，显式开启v1/v2受控导入。
当前断点`results/co3d_seen41_storage_v3_progress/`；当前最终
`results/co3d_seen41_storage_budget_v3_20261002.json`，不要再等旧v1/v2最终文件。
日志继续追加`results/co3d_seen41_storage.log`。同版本安全重试仍先确认服务已结束，
使用`systemctl --user`检查；确认退出后`systemctl --user reset-failed`，再`systemd-run --user`
运行同一个queue；已完成v3也必须输入/代码/路径/统计/新footer身份一致。
小记录`results/co3d_storage_footer_recovery_20261002.json`和Notebook保存恢复时快照，
尚未完成全41类成员预算、图片CRC/解码/相机转换或正式位姿评测。

下一次先核实两个宿主服务及v3结果；健康未完成时安静。v3完整结束后核验41类集合、
2011轨迹/399204组每种成员路径、逐类/全部大小汇总及fingerprint，执行新增分析并归档。
再研究固定采样候选不变的稀疏存储和相机GT预处理，不将已有8类/bowl目录当图像就绪。
RE10K完整来源扫描仍正常，未确认76补齐。恢复/读取不占GPU、不删除数据、留1GiB。

#### 06:51续接修复（历史恢复入口，旧阶段记录保留）

`ec3097a`已推送，41类候选及apple探测不要重复归档。目录预算服务06:40:13退出1：
官方banana_001.zip实际229525成员，超过初版200000上限。只读核验中央目录
29,035,659bytes < 32MiB，进程峰值RSS683928KiB，宿主约24GiB可用内存；
这是目录成员保护阈值偏小，不是已证实的数据损坏、GPU故障或正式测评失败。
成员上限有界提高至400000，32MiB目录字节上限及206/ETag/路径/重复/链接检查不变。

06:54:25安全恢复同名`fast3r-co3d-seen41-storage.service`，MainPID28538，确认active/running，
banana已越过原失败位置。新的断点目录是`results/co3d_seen41_storage_v2_progress/`，
最终报告改为`results/co3d_seen41_storage_budget_v2_20261002.json`，不要等待旧文件名。
通过`--import-verified-v1`只导入已核验apple/backpack：固定v1代码SHA、输入fingerprint、
完整期望路径SHA、各ZIP URL/官方checksum、Range字节和所有逐类汇总，禁止宽泛忽略代码变化。
v1目录/报告保持原样，v2记录v1文件SHA与旧fingerprint；未重放两类网络索引。
失败类别重新读取目录，绝不静默跳过。日志仍追加`results/co3d_seen41_storage.log`。
同版本重试命令仍用下面的queue脚本；它已显式带v1导入选项，后续直接核验复用v2。

小记录`results/co3d_storage_recovery_20261002.json`是恢复时快照，不是最终41类完成。
88项离线测试通过；Notebook新的恢复分析单元保存真实输出、旧单元不覆盖。
RE10K完整来源扫描仍健康，缺失76/全SHA/图像几何尚待检查。磁盘约46GiB空闲仅为快照。
下一次先查服务和v2最终报告；没有新完成/失败时安静，不重启健康服务。

- `6d7bdf1`的51类元数据/独立核验及RE10K扫描启动已归档，不重复。
- 06:35宿主约46GiB空闲；`fast3r-re10k-missing-candidates.service`仍active/running，
  MainPID24596、日志压缩字节持续增长，尚无完整来源报告。健康时不重启、不宣称76补齐。
- PoseDiffusion固定revision `b138198e2891a0f1a1c3435614b9490adb7fd4d6` 的seen名单为41类，
  配置seen/10views。新`results/co3d_seen41_protocol_20261002.json`记录源SHA与类别映射的推断边界。
  仅过滤原51类候选：2011轨迹/399204候选帧；原seed+51类别索引/帧順序保持。
  完整manifest `data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json` 已保存且哈希核验。
  仍未证明Fast3R作者逐轨迹processed清单/100@采样范围等价，不能称正式benchmark准备完毕。
- apple真实6个ZIP目录Range探测完成：83,941,069bytes网络目录，9563组RGB/depth/mask，
  广告目录标示总量6,073,570,411bytes。小报告`results/co3d_apple_range_probe_20261002.json`，
  不下载图像、不验证成员CRC/大包SHA；旧报告不覆盖。
- 新后台`fast3r-co3d-seen41-storage.service` 于06:35:46启动，MainPID27694，已确认running。
  queue：`scripts/queue_co3d_seen41_storage.sh`；日志`results/co3d_seen41_storage.log`。
  逐类断点：`results/co3d_seen41_storage_progress/`（git忽略）；最终
  `results/co3d_seen41_storage_budget_20261002.json`。只读41类大ZIP中央目录，精确校验完整候选
  image/depth/mask路径集合及跨ZIP无重复；每ZIP最多32MiB目录、类别报告落盘前保留1GiB。
  没有RGB下载、训练或GPU任务；不能把预算报告当正式成绩。
  已结束且失败时先查日志。安全重试同一版本会复用完整类别JSON、失败类别重新读目录：
  `systemctl --user reset-failed fast3r-co3d-seen41-storage.service`，然后
  `systemd-run --user --unit=fast3r-co3d-seen41-storage --property=WorkingDirectory=/home/yyz/fast3r /bin/bash /home/yyz/fast3r/scripts/queue_co3d_seen41_storage.sh`。
  数据/代码fingerprint改变时拒绝旧断点，不覆盖或删除，另起显式版本输出再验证。
- 新Notebook `paper-co3d41`已真实执行并保存输出。新增类别AST/过滤及Range/路径/跨包重复
  合成测试通过；全套84项，合成测试不是位姿成绩。

下一次优先查两个服务；健康无变化安静。全41类预算完成后核对各类/399204×3成员与汇总，
更新Notebook实际结果并提交推送。预算若超过磁盘空间，继续研究协议明确的固定抽帧稀疏存储，
保留完整候选manifest和采样签名；不能自行缩减轨迹或换100@范围凑出full成绩。
然后推进原frame_annotations相机转换/裁剪与HF CO3D入口。RE10K全源扫描完成后独立核验SHA、
76集合/GT候选/图片几何，未通过不自动提升正式full-ready。不要重复元数据下载或旧NRGBD/7Scenes实验。

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

## 10:51续接检查点（最新恢复入口，2026-10-02）

上个提交c0ccf8a已推送，真实mapped draw/data核验不重做。本轮接通公开HF单样本位姿：
- scripts/fast3r_hf_co3d_pose_smoke.py复核draw0、raw/processed/候选/权重SHA；网络只有
  RGB与true_shape，GT K仍参与crop，但GT不进网络/focal/PnP。初始化后重设seed12303675。
- fast3r-co3d-pose-smoke.service首次11:01:12退出1，错误仅是新入口形状断言：HF
  landscape_only=False输出portrait，发布版校正回loader landscape，断言误要求portrait。
  修正断言、补真实head wrapper测试；旧预检查与失败runner保留，另写v2预检查，日志追加。
- 同一服务安全等GPU≥10240MiB空闲后重试，11:05:17退出0/inactive/dead/MainPID0。
  results/co3d_pose_draw0_seed42_20261002.json完整，恢复核验
  results/co3d_pose_smoke_recovery_verified_20261002.json记录SHA、退出码和失败原因。
  保存c2w的45pair公开指标重算通过，不重复启动已完成服务。验证命令：
  PYTHONPATH=.:scripts python scripts/fast3r_hf_co3d_pose_smoke.py --verify-only
- 10views/9unique、重复pair[0,7]保留，PnP失败0/10；但RRA@30=0%、RTA@30=13.3333%、
  mAA@30=0%，旋转误差50.1264°–171.6512°，首视图focal27.9829px。低分不删，不能称
  Table1全量/论文效果复现。GT-self RTA@30=97.7778%是零基线诊断，不是网络成绩。
- raw confidence[1,512,384]→发布版校正后[1,384,512]；未改发布算法。144项测试通过，
  Notebook新增CPU预检查与实际位姿分析两单元有真实输出，旧19个不变。
- RE10K扫描仍健康PID24596，11:04历史快照190928682120/205763619478压缩字节；
  没有最终完整来源报告。宿主11:04可用42677755904bytes、GPU11145MiB空闲。
  以上均是历史快照，下次重新查宿主服务/空间/GPU，不重启健康扫描。

接续顺序：先检查RE10K最终源报告；完整时核验全源SHA、76集合、timestamps和原图/GT几何，
不将候选PNG直接升级正式数据。CO3D先固定同输入、同前向保存发布方向基线与替代方向/focal
诊断；不是为凑分改公式，不用GT估计focal/PnP，另写独立结果，不能覆盖这个低分基线。
再设计连续100@共享invalidate/scene tracker/补采RNG的状态保存与重放；目前只有draw0，
逐请求重新seed/清tracker不等于公开连续100@。作者划分/权重对应、训练消融仍未完成。
仅新阶段完成/真实失败/需用户动作通知，无变化保持安静；现有额度续接heartbeat保留。

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

最新恢复入口：提交1d4f32b已推送，旧draw0/方向断言恢复阶段不重复。此次单样本
同次前向诊断已结束，无新后台GPU进程。复核命令：
PYTHONPATH=.:scripts python scripts/diagnose_co3d_pose_orientation.py --verify-only
11:25宿主RE10K扫描仍active/running、PID24596，202714404873/205763619478压缩字节，
未有完整来源JSON；不要重启健康扫描。下一次优先检查全源报告和SHA/76集合/geometry，
再继续CO3D受控诊断与连续100@状态设计。原报告不得覆盖，不跳过失败scene。

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

最新恢复入口：fc829c3已推送，旧CO3D方向诊断和本次RE10K扫描不要重复执行。
只读核验命令：PYTHONPATH=.:scripts python scripts/verify_re10k_missing_full_source.py
scan service已结束，不再使用旧active/PID24596快照判断排队；目前没有本项目新后台数据任务。
下次继续CO3D预测坐标/global-vs-local focal/PnP受控诊断与连续100@共享状态设计，
并审查其他公开RE10K RGB补缺可行性；不登录、不读cookies、不接受账户条款或付费。
缺76仍不足以认定所有安全替代已穷尽，不删除现有heartbeat；不要反复通知同一零补缺结论。
需要作者划分/独立训练权重或预算的实验保持未完成，整篇尚未完成。
