# Fast3R 续接检查点（2026-10-03，Asia/Shanghai）

用户已授权继续整篇论文复现、Notebook归档及逐步推送GitHub，并要求额度刷新后接续。
分支 `local-demo`，远端 `myfork`。本文件是恢复入口，不是“整篇已完成”的声明。

## 结项交接（2026-10-03）

用户授权继续现有论文复现，并允许在遇到重大数据/资源问题或工作明显超出本科项目范围时停止、总结并推送。当前结论为公开权重部分复现；
Table 1缺少76个RE10K场景、CO3D由用户暂缓，训练型消融/完整训练与附录C–E需要缺失的独立权重、规定数据或研究规模算力。
这些项目不以不完整子集或小规模试跑代替。工作流、实测结果、剩余边界、科研方法和简历写法见`REPRODUCTION_HANDOFF.md`。
未来若恢复CO3D，必须由用户重新授权；不自动下载或启动。

## 最新授权范围（2026-10-03：用户暂缓CO3D，并清理本地数据）

**用户最新指令优先：暂不做CO3D这项评测，删除之前为它下载/生成的本地数据，节省空间。**
不是运行失败；不得按下方历史恢复命令自动修复、重启或重下载CO3D。
原“整篇复现”目标现在有明确用户暂缓项，保留部分公开权重复现，不声称整篇完成。

两个宿主服务`fast3r-co3d51-source-prepare-v1.service`和`fast3r-co3d51-pose-eval-v1.service`
已按用户要求停止，inactive、MainPID0；停止时输入提交26/1000，位姿前向尚未开始。
16个精确核实的`data/co3d*`目录已永久删除：旧/新raw、processed、metadata、ZIP目录索引、
Range/probe和深度诊断输入。没有删除代码、Notebook、results报告/图/事务、公共权重、
DTU/NRGBD/7-Scenes/RealEstate10K及其他项目数据。
凭据`results/co3d_evaluation_deferred_cleanup_20261003.json`列出每个绝对路径、原占用、
前后可用量及59份已提交CO3D报告/图SHA；删除后59份逐项完全相同，16目标均不存在。
可用空间实测增加4,562,313,216bytes（约4.25GiB），清理后37,773,111,296bytes（约35.18GiB），
这是当次快照，不是以后固定空间。永久删除不能从回收站恢复；若以后重新授权需要重新准备数据。

此清理凭据也是禁止自动恢复标记：两个queue入口在任何日志/数据/模型操作之前检查它并退出。
实际调用两个入口已证实只输出用户暂缓、不启动任务。不要移除标记绕过用户指令。
历史Notebook输出、原始JSON/指标仍保留，但源输入和metadata已删除；相关完整本地重放不再可用，
历史“数据保留/服务运行/恢复命令”都是当时快照，不是当前状态或新的授权。

**自动化权限限制：**已尝试通过官方automation_update工具更新`fast3r`提示，两次均被
`MCP tool call requires approval, but approval policy is never`拒绝；没有绕过权限编辑TOML。
定时任务文字尚未更新，不得声称更新成功。未来heartbeat必须先读本段，并遵守代码暂停标记；
如用户希望修改定时卡片文字，需要允许该工具更新或在应用中手动把CO3D重新下载/评测移出。

接下来只用已保留的数据推进：
1. 已完成DTU22/NRGBD9/7-Scenes18的Table3/4/5 CPU汇总复核：逐场景重算值与保存aggregate一致，
   补齐DTU Table5 local/global对照。结果见`PAPER_REPRODUCTION.md`。
2. 固定输入阈值诊断已扩展到DTU `scan1`、`scan10`、`scan11`，报告分别为
   `results/diagnostics/dtu_scan1_threshold_sensitivity_seed42_v1.json`、
   `results/diagnostics/dtu_scan10_threshold_sensitivity_seed42_v1.json`、
   `results/diagnostics/dtu_scan11_threshold_sensitivity_seed42_v1.json`。
   每场景一次前向，在stride1完整序列的rounded linspace 10视角上对照7组阈值；seed为1052/10052/11052。
   三组输入标签及85/0 baseline的全部8个local指标都与历史`dtu_paper_experiments.json`对应行完全相同；
   三报告代码/权重指纹相同，预测SHA前后相同，指标均有限。三场景描述性mean Acc/Comp：85/0为6.9304/2.4108，
   metric75为3.4452/13.1753。阈值升高表现为Accuracy距离下降、Completion恶化；这是覆盖率敏感性，不是全量Table4/5
   或GT调参推荐。`alignment percentile`参数同时控制local-to-global对齐筛点和GT配准权重，解释结果时需保留这一耦合。
   14:35时PID143582仍在跑OpenVLA任务，因此当时没有启动Fast3R；15:48后GPU空闲，固定场景任务随后完成。
   初次stride5/标签混用在加载模型前被拒绝，修正后才正式前向。Notebook现116单元；前114个来自阈值诊断阶段，
   原始HEAD的112单元逐项相同。新RE10K审计单元在项目Python中CPU执行并保存输出；隔离Jupyter启动无返回后中断，
   不声称本轮通过宿主Jupyter执行。
   旧245项套件（241 pass、4 CO3D输入缺失skip）与新增1项采样回归通过，合计242 pass、4 skip。
   恢复时先确认原计算进程结束、GPU计算进程列表为空，
   并两次确认至少10240MiB空闲；该实验只诊断本机公开权重/指标差距，不是训练消融，也不用于
   按GT挑阈值凑论文分数。
3. 已只读审计附录C/D/E：InstantSplat与rmvd缺失，CO3D已暂缓清理，Tanks & Temples、
   ScanNet/ETH3D及RobustMVD规定split未准备；现有DTU数据不等同Appendix E测试split。
   本次没有下载或安装依赖。后续有可用数据/GPU后再审查C/D/E支持部分；
   splatting/独立训练模型/128A100条件不足时明确列未完成，不临时大下载/付费。
4. RE10K仍缺76/1832，正式Table1未完成。2026-10-03只读发现公开候选
   `ghuijo/realestate10k`（页面标注429GB，预览出现test metadata/image路径），但viewer失败，76个规定ID覆盖、
   图像几何和来源条款均未验证；本轮下载0 bytes。详见
   `results/re10k_missing_source_followup_20261003.json`。不下载429GB归档、不访问受限源，
   不重复已完成的205GB扫描或以子集冒充完整Table1。

## CO3D清理阶段验证快照（2026-10-03；当时Notebook为107单元）

清理阶段的Notebook107单元中，原105逐项完全相同，新清理分析经宿主Jupyter真实执行无error。
测试套件245项：241项执行通过，4项依赖已删除真实CO3D清单的集成检查明确skip并注明原因。
清理后首次运行这4项报缺manifest，随后仅在用户暂缓标记存在且清单缺失时标明不可运行；
不自动重下载、不伪称这4项或删除后的全量CO3D输入重放已通过。两个queue保护和bash语法检查通过。

## 历史恢复入口（2026-10-03 12:27：队列当时已接线；现已由用户暂缓）

对应§4.2/Table1候选适配，**没有新位姿成绩，正式Table1/整篇仍未完成**。
上一已推送检查点`9281815`；旧目录审计/1请求重放/100请求/几何诊断均已归档，不重启。
12:26宿主准备服务`fast3r-co3d51-source-prepare-v1.service`active/running、MainPID115575，
最新日志已提交21/1000，继续按需CPU准备。宿主可用33,245,532,160bytes（快照）；
准备身份绑定的12份代码SHA全部未变，旧raw/processed/历史报告保持不变。

新`scripts/fast3r_hf_co3d51_pose_eval_v1.py`必须先拥有完整1000准备摘要与独立只读重放证明，
逐项核对1000请求SHA/采样初末state并fresh offline replay，之后才允许加载本地公开HF模型。
1/100/999请求、缺证明、误标正式成绩、输入/GT/指标损坏均拒绝。GT不输入模型/focal/PnP。
声明候选为source51、固定横向512×384、seed42+request、16-mixed、DPT chunk2；
发布conf>1/100次PnP迭代、零焦距原样传递，失败identity和重复视图的全部45pair/request保留。
1000 request级算术均值与45000 pair pooled指标分别保存，不能拿旧100报告充新1000。

12:26:17启动独立宿主队列`fast3r-co3d51-pose-eval-v1.service`，MainPID124489，
active/running且日志明确只等待准备服务正常退出、全1000摘要及完整独立证明；**此时未加载模型/占GPU**。
日志`results/co3d51_pose_eval_v1.log`。满足输入门禁后才等待GPU>=10240MiB空闲两次检查再前向；
空闲检查不是预约或跨进程互斥，不终止其他任务。源准备失败/退出但缺结果时队列硬停，保留检查点。
新位姿事务`results/co3d51_pose_seed42_progress_v1`上限512MiB、每次至少留1GiB，
initial和每request原子新建、权重/config/代码/完整输入证明绑定；完整已验证request可复用。
任意断电/服务重启恢复未做完整故障实测，不能保证通用崩溃恢复；不覆盖既有最终JSON。

最终候选`results/co3d51_pose_seed42_candidate_v1.json`，独立证明
`results/co3d51_pose_seed42_verified_20261003.json`；当前两者均未生成。
队列完成后会自动fresh输入核验和`scripts/verify_co3d51_candidate_pose_v1.py`独立重算
保存GT/位姿/45000误差/聚合/失败与重复计数。它不是第二次独立网络重推理；保留该限制。
若最终JSON已有而独立证明未写完，恢复只补核验，不重新前向。

`results/co3d51_pose_input_gate_20261003.json`是12:17采集的18/1000文件计数快照，
不是最新数量或18请求独立输入证明。实证摘要/证明均缺失时model_loaded=false/forward0/network0。
105单元Notebook保留原103逐项不变，新门禁分析12:27经宿主Jupyter执行无error；
完整244项离线测试通过，含完整覆盖门禁、GT隔离、重复/identity/损坏注入及代码路径边界。

下一次顺序：
1. 读此入口/git；宿主检查两个服务、进程/日志/空间。健康准备或等待中不重复启动。
2. 准备摘要和完整输入证明出现且准备服务退出后核验归档；GPU队列会自行接续，无需重启。
3. 仅服务退出且尚未完成时诊断。同一身份恢复准备用
   `bash scripts/queue_co3d51_source_prepare_v1.sh`；恢复位姿队列用
   `bash scripts/queue_co3d51_pose_eval_v1.sh`，必须确保对应原服务/进程均已退出，不能并行启动第二份。
4. 新完整位姿报告/独立证明均出现且位姿服务退出后，再核验唯一实际输入集合/45000pair与聚合，
   写Notebook真实成绩、更新各文档、测试/提交/推送；所有作者等价/论文组别/正式Table1标记仍false。
5. RE10K仍1756/1832、缺76；旧来源扫描不重跑。完整训练/独立权重消融/附录仍未完成。

独立后台进程不依赖Codex推理额度，但停机/挂起、数据或资源失败会中断；不能保证额度刷新精确唤醒。
现有heartbeat继续按检查点接续，不购买额度/重置权益，不因排队慢重复通知或报阻塞。

## 已归档恢复入口（2026-10-03 11:21：目录预算已核验，真实1000请求按需准备运行中）

§4.2/Table1前提继续推进，**不是新位姿成绩或整篇完成**。前一提交`e9c3ee0`已推送`myfork/local-demo`。
`fast3r-co3d51-source-storage.service`10:57:12正常退出0、MainPID0；阶段已完成，不重启。
完整目录结果`results/co3d_51_source_storage_v1_20261003.json`，独立证明
`results/co3d_51_source_storage_verified_20261003.json`：51类，复用38类旧索引/补13类，
all-valid名义8835帧的RGB/depth/mask原始成员共6,539,401,033bytes；保守补采可达集合
103599帧共76,458,893,204bytes。只代表目录大小，非CRC/解码/真实GT有效证明。
旧raw2GiB缓存不能据此称1000请求能装下；不预下载整个76.46GB补采集合。

新`scripts/co3d51_source_lazy_v1.py`用独立root和声明资源包络：
`data/co3d51_source_raw_v1`8GiB、`data/co3d51_source_processed_v1`3GiB、
`results/co3d51_source_prepare_v1_progress`事务4GiB，下载/写入前至少留1GiB。
旧raw2GiB/processed512MiB及100请求缓存/数据/结果全部冻结，只允许核验后只读借用旧raw。
11:20宿主free33,452,740,608bytes（快照）；新完整资源包络15GiB+reserve1GiB可装下，
但真实补采可能先碰上限，不能保证1000全部完成。到界/IO/GT错误硬停，不清理旧数据或静默跳场景。

首个实际source51/1000@请求10views/7unique已完成、原始成员Range/CRC/SHA和GT检查通过；
`results/co3d51_source_inputs_prefix1_verified_20261003.json`独立只读重放RGB tensor/GT/K/RNG/
load trace/pool attempts/每步共享after-state完全相同，network0/forward0。这是不可变1请求快照，
不是最新实时数量、1000已完成或全部新帧预处理参考的独立重算。

宿主`fast3r-co3d51-source-prepare-v1.service`11:20:51启动，11:21active/running，MainPID115575。
独立CPU服务从已核验请求边界继续，日志`results/co3d51_source_prepare_v1.log`；不使用GPU。
单worker完整请求事务绑定代码/候选/资源/crop及invalidity、Python/NumPy RNG、输入SHA；
跨进程恢复拒绝身份变化。未提交processed文件若断电损坏仍需诊断，不称通用崩溃恢复。
首个真实前缀跨进程只读重放通过；尚未测试所有异常中断情形。

下一次顺序：
1. 先读本入口/git，宿主查服务、日志、空间；健康运行不重复启动，不重复归档目录审计/1请求证明。
2. 仅服务已退出且未生成完整摘要时诊断失败；同一身份恢复命令
   `bash scripts/queue_co3d51_source_prepare_v1.sh`，保留所有事务与数据。
3. 完整摘要`results/co3d51_source_prepare_v1_summary_20261003.json`出现后，服务还会自动
   运行只读`scripts/verify_co3d51_source_inputs_v1.py`，输出
   `results/co3d51_source_inputs_full_verified_20261003.json`。有两份完整结果且进程退出才归档。
4. 全1000请求准备/独立重放完成后，按实际GT/重复视图/补采集合再适配公开HF新候选GPU评测。
   不复用旧100成绩充1000；精确作者processed JSON/RNG和HF论文组别仍未知，正式Table1不自动打勾。

新固定横向512×384 crop取消portrait/square目标反转，其余源采样/补采及V4参考预处理保留；
crop会改变真实depth validity/RNG消耗，名义8835只是all-valid对照，不是实际最终输入清单。
RE10K仍1756/1832、缺76；旧205GB来源扫描/235包目录/旧100前向/5诊断前向均已归档不重跑。
完整训练/独立消融权重/附录实验保持未完成，无新增账号/付费授权需要；heartbeat继续。
Notebook103单元，新预算/1请求分析11:25经宿主Jupyter真实执行无error；
Notebook结构校验和原101逐项完全相同比较通过，完整离线229项测试通过。

## 已归档恢复入口（2026-10-03 10:48：51类发布采样追踪完成；下文目录运行是当时快照）

额度已恢复，宿主权限和Jupyter可用。10:33宿主可用33,523,810,304bytes、GPU空闲11,102MiB；
这些仅是快照。两个旧沙箱Notebook尝试PID95939/95996经命令/路径核实后已结束，未终止GPU任务。
历史工程提案`results/co3d_51_1000_protocol_budget_20261003.json`保留不变；
其强制1000不同轨迹/10000帧的策略**不是源采样器**，其缓存上限**不是存储能装下的证明**，后续评测不使用。

新`results/co3d_51_released_sampler_20261003.json`实际调用发布采样器的all-valid stub和1000@：
51类、836不同轨迹、8835不同帧、581请求含重复视图，最少6不同帧；不是实际GT有效输入。
data plan`data/co3d_test_metadata/released_nominal_51_1000_seed777_v1.json` SHA256
`cd7a0b1ef5ea4633a88360f91823a55e5a4e09d028af4c5967df0068ff75ff32`，不提交大清单到Git。
dataset seed777有发布/作者依据；combination seed42和epoch0为审计选择，原作者RNG/清单仍未知。
保守±4 jitter/5次scene尝试集合2205轨迹/103599帧，不冒充实际采样或完整GT。

`fast3r-co3d51-source-storage.service`已启动，日志`results/co3d_51_source_storage_v1.log`；
按archive/category原子断点`data/co3d_51_source_storage_v1`，独立进程可在Codex额度不可用时继续。
复用38类旧本地索引，只补缺13类的目录记录；不重做旧235包完整扫描、不下载图片、不用GPU。
代码`scripts/plan_co3d_51_source_storage.py`，512MiB审计文件上限、至少1GiB磁盘预留。
完成文件`results/co3d_51_source_storage_v1_20261003.json`；有结果且服务退出后，
运行独立`scripts/verify_co3d_51_source_storage.py`。健康运行不重启；失败先诊断并保留断点。
仅服务已退出且结果不存在时恢复：`bash scripts/queue_co3d_51_source_storage.sh`。
旧raw2GiB/processed512MiB缓存及100请求/5个诊断前向全部冻结，不静默扩大或清理。
下一步按真实目录预算另算processed/断点峰值并声明新版本资源上限，再启动51类1000请求按需准备。
原作者清单/HF实验对应、RE10K缺76、训练/独立消融/附录实验仍未完成。

Notebook101单元：原97逐项不变，旧工程提案读取及新源采样追踪两个分析单元
于宿主Jupyter真实执行（10:48），无error；不重跑任何GPUbenchmark。
完整离线套件218项通过（较此前200项新增18项），Notebook结构与原97单元独立比较通过。
heartbeat继续，从本入口恢复，不购买额度或使用reset权益，不保证精确重置时刻唤醒。

## 历史恢复入口（2026-10-03 07:15：portrait输入几何对照完成）

`fast3r-co3d-landscape-inputs.service`07:15:10正常退出0，inactive/dead、MainPID0；
日志`results/co3d_landscape_input_diagnostic_v1.log`。不要重启此阶段或原100请求。
结果`results/co3d_landscape_input_diagnostic_v1_20261003.json`，独立证明
`results/co3d_landscape_input_diagnostic_verified_20261003.json`，CPU审计
`results/co3d_portrait_geometry_audit_20261003.json`。原100报告SHA仍e52f2101…036d。

确认公开HF实际为PatchEmbedDust3R（忽略true_shape），不是ManyAR/DINO；portrait时
encoder24×32 token与DPT32×24 reshape不是空间transpose。第二个问题：loader转置像素后
K行交换、det<0，发布PnP却用标准K，3D/GT相机轴没相应变化。
完美点合成测试原PnP误差0°，转置像素+标准K约179.84°仍返回成功。
原100请求63全portrait/29全landscape/8混合、661个portrait视角；不是作者全量分布。

固定request0/2/3共5个新前向（不是同次前向）：仅改true_shape，两portrait mAA仍0；
只取消输入crop的portrait/square目标反转后，mAA分别0→59.7133%、0→12.4014%；
横向cup控制RGB完全不变，两新条件共享1次预测，保持92.2581%，位姿/焦距/指标匹配历史。
固定帧顺序/重复帧/GT不变，GT不输入网络/焦距/PnP，全部270个新branch pair独立重算。
改变crop也改变视野，不能分解各机制贡献或报告选择probe平均；仍非正式Table1。
核验器不重跑训练模型，prediction hash只是代码绑定provenance，限制已列。
196项离线测试通过；6项内存损坏注入（正式误标/缺request/同次前向误标/RGB/指标/identity）
均被独立核验器拒绝，存储文件未改。Notebook97单元，原95逐项不变；新宿主真实输出无error，图目视核验。

下一次顺序：
1. 先读git/本入口并核实宿主后台；此5前向、上次PnP诊断、原100结果全部已完成。
2. 按作者#78描述重审51类DUSt3R候选/1000@、seed777和强制横向crop，先做
   候选清单身份、采样/补采变化与**新的可执行空间预算**，再启动新版本按需预处理。
   不给旧100请求换crop后当作者1000次成绩；原作者清单/RNG/权重组别仍未确认。
   保留41类/100历史输入、报告及2GiB raw/512MiB processed缓存，不清理旧数据腾空间。
3. 不重跑旧235 ZIP目录或205GB RE10K来源扫描；RE10K仍1756/1832、缺76。
   缺独立权重/128A100的训练保持未完成；继续安全审查公开来源和附录可行性。

末次磁盘空闲约31GiB仅为本轮快照；下载前宿主重查、至少留1GiB。heartbeat继续，无新增付费/账号需要。
CPU核验：`PYTHONPATH=.:scripts python scripts/verify_co3d_landscape_inputs.py`。
仅有未提交request时后台恢复：`bash scripts/queue_co3d_landscape_inputs.sh`；
完整结果存在只核验而不加载模型。下方均为历史入口，不作重新启动指令。

## 历史恢复入口（2026-10-03：作者协议纠正与3个固定probe完成）

`fast3r-co3d-pnp-diagnostic.service`于06:42:22正常结束0，inactive/dead、MainPID0；
不重启，日志`results/co3d_candidate_pnp_diagnostic_v1.log`。
新`results/co3d_candidate_pnp_diagnostic_v1_20261003.json`有request0/2/3的12分支，
每个request一次前向，baseline poses最大差0，pair/metric独立重算。
证明`results/co3d_candidate_pnp_diagnostic_verified_20261003.json`；
入口`scripts/diagnose_co3d_candidate_pnp.py`，核验`scripts/verify_co3d_pnp_diagnostic.py`。
零焦距probe搜索后回退10→0但mAA仍0；高分probe仅mask mAA92.2581→94.2652%。
按历史表现挑选的诊断不做总体平均、不覆盖100请求、不作正式Table1。186项离线测试通过。
5项内存损坏注入（正式误标/缺request/预测哈希/metric/丢identity）全部被独立核验器拒绝，
没有改存储JSON。Notebook原93单元逐项保留，新2单元已有宿主Jupyter真实输出且无error；图已目视检查。

**重要纠正**：作者#78说明实际1000次CO3D test采样、可能超41类，发布配置却是100次。
来源`results/co3d_author_protocol_update_20261003.json`；旧41类/100仅候选适配，
不继续作为作者必要规定。精确processed JSON/采样顺序/HF论文组别仍未确认。
作者#76提供portrait/landscape输入裁剪线索；两个低分probe是portrait，高分是landscape，
相关性不证明根因，也不能只把已transpose输出再改一遍。

下一次顺序：
1. 读最新git历史/本入口，核实宿主后台；新诊断和100请求基线已完成，不重启。
2. 优先审计**输入裁剪/相机像素几何**并做固定帧对照：作者回忆全landscape512×384，
   发布基类却自动反转portrait分辨率再transpose tensor。保留GT/RNG；不以GT焦距补分，
   不修改源数据，不将不同crop输入的对照藏成同输入/同次前向。
3. 重审51类DUSt3R候选/1000@范围、补采及空间预算；既有41类缓存/输入/报告保留。
   不重复已归档ZIP目录/100请求实验，不未经预算启动1000前向。原作者清单未确认前，
   新51类/1000也先标作者描述候选，不自动变正式成绩。
4. RE10K1756/1832仍缺76；已结束205GB来源扫描不重跑。继续安全推进附录可行性，
   缺独立权重/128A100的训练实验不以短训练或Demo填补。

本轮磁盘约32GiB只是快照，每次仍宿主重查且保留1GiB。heartbeat保持，无需新增账号/付费预算。
下方旧启动/失败记录是历史，不当实时指令。

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

当前恢复入口：scripts/prepare_co3d_continuous_v3.py；事务
results/co3d_continuous_prepare_v3_20261002/；processed data/co3d_lazy_v3_processed。
旧v1/v2失败服务不要重启；不编辑已经绑定的v3代码/证据。下载仍raw2GiB、processed512MiB，
至少1GiB预留、无清理/扩预算；本轮宿主可用36,049,698,816bytes为时间点快照。
先检查systemctl --user show fast3r-co3d-continuous-prepare-v3.service、宿主进程与日志。
健康运行不重复启动；失败先诊断并保留未提交请求，不能静默跳过新异常。
只读恢复校验：PYTHONPATH=.:scripts python scripts/prepare_co3d_continuous_v3.py --verify-only
迁移复核：PYTHONPATH=.:scripts python scripts/verify_co3d_v3_migration.py
仅确认已退出且原因可安全恢复后：
systemd-run --user --unit=fast3r-co3d-continuous-prepare-v3 --property=WorkingDirectory=/home/yyz/fast3r --property=StandardOutput=append:/home/yyz/fast3r/results/co3d_continuous_prepare_v3.log --property=StandardError=append:/home/yyz/fast3r/results/co3d_continuous_prepare_v3.log /bin/bash /home/yyz/fast3r/scripts/queue_co3d_continuous_prepare_v3.sh
后续：核验第4请求真实补采trace；100请求完整后独立全前缀重放，才接GPU评测。
保持heartbeat；现不需要用户提供许可/预算，不重复已归档证明或旧205GB扫描。

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

最新恢复入口scripts/prepare_co3d_continuous_v4.py；事务results/co3d_continuous_prepare_v4_20261002/；
processed data/co3d_lazy_v4_processed。旧v1–v3失败服务不要重启，v4健康运行不重复启动。
本轮宿主可用36,032,704,512bytes为时间点快照；raw2GiB/processed512MiB/至少1GiB预留不变。
新全前缀只读重放工具scripts/verify_co3d_continuous_v4_prefix.py检查code lineage、
raw/processed SHA、input tensor/GT/rng、load_trace、pool_attempts和每步共享after-state。
完整请求出现后用--output指定新的小结果名；不覆盖旧快照，未完成请求不计入进度。
只读恢复校验：PYTHONPATH=.:scripts python scripts/prepare_co3d_continuous_v4.py --verify-only
全前缀核验：PYTHONPATH=.:scripts python scripts/verify_co3d_continuous_v4_prefix.py --output results/新名称.json
仅已退出且原因可安全恢复时：
systemd-run --user --unit=fast3r-co3d-continuous-prepare-v4 --property=WorkingDirectory=/home/yyz/fast3r --property=StandardOutput=append:/home/yyz/fast3r/results/co3d_continuous_prepare_v4.log --property=StandardError=append:/home/yyz/fast3r/results/co3d_continuous_prepare_v4.log /bin/bash /home/yyz/fast3r/scripts/queue_co3d_continuous_prepare_v4.sh
先核验真实第4请求跨scene重试/补采完整事务；100请求准备完成后必须全前缀重放，才接GPU。
禁止静默跳过失败请求、不改已绑定代码/证据、不重复旧205GB来源扫描。heartbeat保持。

第4请求已完整提交并独立重放验证，不再停在旧3/100边界：
results/co3d_v4_prefix_snapshot_20261002T1200_verified.json核验4/100完整事务，
input tensors/GT/rng、load_trace、pool_attempts和每步共享after-state全部一致，网络0/前向0。
request3 base12089587：原_get_views经首轨迹8个零深度候选后重试到249_26596_53531，
按原补采规则返回10视图/7个不同frame；不是人工替换场景，不等于审计首轨迹全部202帧。
后台继续后续准备，此4请求证据是不可变快照，不是100已完成或最新实时数量。
下一次从v4宿主服务/日志/事务数量接续，不重放或重复归档这份4请求快照。

### 2026-10-03：CO3D候选100请求全前缀核验完成，启动独立位姿评测

- v4准备服务已正常退出0；完整提交100/100请求。服务日志
  `results/co3d_continuous_prepare_v4.log`；逐请求事务位于
  `results/co3d_continuous_prepare_v4_20261002/`，处理数据位于
  `data/co3d_lazy_v4_processed/`，原始Range缓存仍受2GiB上限约束。
- 使用 `scripts/verify_co3d_continuous_v4_prefix.py --output
  results/co3d_v4_prefix_full_20261003.json` 对全100请求逐项只读重放。
  成功证据确认输入tensor、相机GT、RNG、load trace、pool attempts和共享状态完全一致；
  network_bytes=0、model_forward_count=0。此结果验证输入准备与断点链，不是位姿成绩。
- 新增候选评测入口 `scripts/fast3r_hf_co3d_100_pose_eval.py` 和队列脚本
  `scripts/queue_co3d_pose_100_eval.sh`。策略按100个已固定请求顺序逐个重放并再核对
  RGB输入/GT/共享状态；仅`img`与`true_shape`送入Fast3R，GT不用于网络、焦距或PnP。
  每个请求使用独立seed `42 + request_index`，保留10视角中的重复项、45对全部相对位姿，
  PnP失败的identity fallback保留并计数；输出请求级指标平均及4500 pair pooled口径。
  这100请求仅是候选100@适配，不是作者CO3D测试划分，不能标作正式Table 1。
- 新评测service `fast3r-co3d-pose-100-eval.service` 于本检查点启动，日志
  `results/co3d_pose_100_eval_v1.log`，逐请求checkpoint在
  `results/co3d_pose_100_seed42_progress_v1/`，最终文件预定
  `results/co3d_pose_100_seed42_adaptation_v1.json`。启动时MainPID60744、active/running；
  queue脚本等候至少10240MiB空闲GPU，再双重检查，磁盘约32GiB可用。以上均为启动快照，
  下一次必须重新查询服务/GPU/空间。评测独立续跑；错误时保留已保存请求，不跳过失败。
- 当最终候选报告生成后，用其真实汇总更新Notebook、PAPER_REPRODUCTION.md和
  PROTOCOL_AUDIT.md，运行Notebook分析单元并逐步提交/推送小型代码和报告。
  RE10K仍缺76个规定RGB场景；公开checkpoint与论文权重映射、CO3D author split等价、
  独立训练消融/完整训练仍未完成。候选分数不升级为全量Table 1成绩。

**评测启动纠正记录（同日）**：评测v1 dry-run发现初始化采样state漏设Python seed，服务在模型加载/前向前退出1；保留日志 `results/co3d_pose_100_eval_v1.log` 与v1空progress目录。将准备state核对前显式恢复初始seed，使用独立v2 progress/output身份，不复用v1检查点；修复后dry-run再次核验100请求，network0、forward0。评测v2服务现为active/running，启动PID61381，追加日志 `results/co3d_pose_100_eval_v2.log`，新progress `results/co3d_pose_100_seed42_progress_v2/`，最终输出 `results/co3d_pose_100_seed42_adaptation_v2.json`。本次资源快照磁盘32GiB、GPU空闲约10810MiB；启动服务会重新等待并双检10240MiB阈值。v1失败未使用GPU、未写pose结果、100条输入数据和v1/v4历史文件均保留。

### 候选位姿评测v2焦距边界诊断与v3续跑（2026-10-03）

- v2按固定输入完成3/100请求后，在request 3因参考焦距估计为0触发入口前置保护而退出1；保留日志`results/co3d_pose_100_eval_v2.log`与v2断点，不把3条升格成正式结果。
- 固定request 3在真实模型前向后的诊断记录：`results/co3d_candidate_request003_focal_diagnostic_20261003.json`。模型预测点图/置信度有限，confidence p10=1.0，但发布版估计焦距为0。公开`fast_pnp`接受0焦距并捕获OpenCV求解错误、返回None；原协议随后使用显式identity fallback。因此v3只移除过严的`focal<=0`门槛：0沿公开PnP路径处理并计为PnP失败回退；负值或非有限焦距、非有限点图/置信度仍硬失败。不以GT焦距替换、不修正焦距，不静默丢弃请求。
- 新版`fast3r_hf_co3d_100_pose_eval.py`使用独立v3进度目录和最终文件，禁止混合v2状态；`--dry-run`全前缀核验通过（100请求、网络0、模型前向0），`git diff --check`通过。
- `fast3r-co3d-pose-100-eval.service`已于2026-10-03 05:26启动，当前检查时active/running，MainPID 64413；GPU空闲10817MiB，项目盘余量34211807232bytes。实时状态需每次续接重新查询。日志为`results/co3d_pose_100_eval_v3.log`，progress为`results/co3d_pose_100_seed42_progress_v3/`，最终文件`results/co3d_pose_100_seed42_adaptation_v3.json`。后台先校验完整输入前缀，再至少等待10240MiB可用显存。进程正运行；此100请求仍是候选协议适配，作者split等价未经证明，不得称为正式Table 1或完整论文成绩。

### 当前恢复入口：CO3D候选100请求评测完成并独立核验（2026-10-03）

宿主fast3r-co3d-pose-100-eval.service于05:32:52正常退出0、inactive/dead、MainPID0；
全部100请求/4500pair已完成，无须重启v4准备或v3评测。完整结果
results/co3d_pose_100_seed42_adaptation_v3.json的SHA256为
e52f2101a05475dcac4b3a784a27dc7c5e0f10a4e8f35683a62601671c90036d。
新独立核验脚本scripts/verify_co3d_candidate_pose_report.py逐项连接GT/输入、
prepared/eval事务和完整报告，重算全部pair误差、metrics/aggregate，所有identity回退保留。
小摘要results/co3d_pose_100_seed42_verified_summary_20261003.json记录实测：
RRA15=31.2889%、RTA15=28.5778%、mAA30=23.8007%，8零焦距请求/80PnP失败视角。
实际99轨迹/38类/884唯一返回RGB，64重复请求/160重复pair；此100@候选适配
不等于2011轨迹全量，作者split和论文checkpoint映射仍未证实，正式Table1未完成。
本轮宿主磁盘约32GiB；GPU当前有其他占用，后续GPU实验仍要重新检查10240MiB阈值。

Notebook新增paper-co3d-candidate100-pose单元及真实执行输出；文档主表已更新，
旧GPU报告/Notebook输出保留。完整大JSON和progress目录仅留本地，
提交小摘要、全前缀proof、焦距诊断小JSON、代码及科学图。
只读复核：PYTHONPATH=.:scripts python scripts/verify_co3d_candidate_pose_report.py

下一步优先核对公开作者processed split/100@身份与checkpoint实验映射，
再设计同一固定输入下发布版与论文PnP描述的受控诊断（随机焦距猜测、top15% confidence）。
不得以改变协议后更高分覆盖基线，不用GT焦距凑分。
RE10K仍1756/1832、缺76；已完整扫描的205GB来源不再重跑。
后续安全推进附录评测可行性；缺独立权重/128A100训练资源的训练实验不打勾。
任务仍有可推进的协议工作，heartbeat保持；无新增用户账户/预算要求。

本轮附加核验：6项内存损坏注入（正式成绩误标、缺请求、aggregate、RGB哈希、
非identity失败回退、重复pair）全部被独立核验器拒绝；没有修改原报告。
Notebook原91单元逐项等于提交前历史版本，新2单元含宿主Jupyter真实执行输出，
无error；科学图已渲染检查，git diff --check通过。

### 2026-10-03 07:52：51 类/1000 请求元数据与空间预算检查点

新增 `scripts/audit_co3d_51_1000_protocol_budget.py` 与小结果 `results/co3d_51_1000_protocol_budget_20261003.json`。不读取图像成员、不联网、不加载模型，仅核验 reconstructed 官方候选 51 类/2511 轨迹/498757 帧、旧 seen41 41 类/2011 轨迹/399204 帧，以及新增 10 类、500 轨迹、99553 帧。固定 seed=42 的 1000 条 distinct-sequence × 10-frame 计划摘要 SHA256=`34847924f082fd4536fe710a688cb82fda684f2692597a785289d3f77c9a7933`。

该历史JSON的磁盘快照为33527140352bytes；raw2GiB/processed512MiB只是缓存上限，不能证明1000请求可装下。4项新增测试和当时200项离线测试通过，但旧Notebook单元先前并非宿主Jupyter输出；本次已修正并真实执行。后续不使用该不同于源采样器的工程计划，不把它当作者Table1。当前可执行入口见本文件顶部。
