# 公开权重与评测协议审计（2026-10-01）

对应论文 §3.2、§4.2–4.3、§5.4。证据快照见
[`results/protocol_audit_20261001.json`](results/protocol_audit_20261001.json)。
这是代码、权重和数据前提的审计，不是新增训练或位姿测评成绩。

## 1. 确认了哪份权重？

重新计算本地两个文件的 SHA256，并只读取公开 HF 小型元数据与配置。
本地 `model.safetensors`（2,590,286,416 bytes）和 `config.json`（941 bytes）
均与当前公开仓库 revision
`a2c770b768ceb3a53c36c4f7a3619db0413dc3a1` 一致：

```text
model:  1357d42e74539ba03510ea1986c7464a7facf17157779e2b7ebd6284e79e0427
config: 88edc53ba652ad835b86af27d980a46808d722b4d8f9d0d4360d05c789ada92b
```

这排除了“本地文件与当前公开发布文件不同”这一问题，但没有证明它对应
Table 1/3/4/5 所用的具体训练模型。原始下载 revision 未记录，不倒填成上述版本。
[模型卡](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512)未给出每项论文实验、
训练数据变体和消融权重的完整映射。不能据此将当前权重称作 Fast3R-no-outdoor，
也不能把同一权重冒充不同训练视角/模型规模/位置插值的独立模型。

## 2. 已核实的代码与论文文字差异

| 位置 | 论文/使用者可能理解 | 本次实际运行的公开代码 |
| --- | --- | --- |
| §4.3、§5.4 对齐 | 论文称 local→global 使用 ICP | `align_local_pts3d_to_global` 使用同像素对应点的 RoMa 相似变换注册，包含 scale；没有迭代最近邻 ICP |
| reconstruction GT 对齐 | 代码变量/参数含 `icp` | `evaluate_reconstruction` 对对应像素预测/GT 调用带权重的同一注册函数 |
| §3.2 Eq. (3) | 公式印为 `+ alpha * log(confidence)` | `losses.py` 实现为 `- alpha * log(confidence)`，本项目数值验证沿用公开实现 |
| 输入尺寸 | “512”易被误读为只把长边缩放至512 | 当前重建 loader 的整数分辨率经基类变成 `(512,512)`；不是 Demo 的512×288 |

源码入口：[`multiview_dust3r_module.py`](fast3r/models/multiview_dust3r_module.py)、
[`losses.py`](fast3r/dust3r/losses.py)、
[`base_stereo_view_dataset.py`](fast3r/dust3r/datasets/base/base_stereo_view_dataset.py)。
论文依据为[arXiv v2](https://arxiv.org/html/2501.13928v2)。

GT 有效深度掩码参与重建注册与指标掩码，不是 RGB-only Demo 的纯推理输出。
local→global 用 global confidence 的高置信度部分，预测→GT 用被评测 head 的置信度；
两者不能笼统描述成完全相同的过滤。现有重建运行记录 percentile85/0、seed42、
DPT chunk2、混合精度；各项均保留，不为了接近论文数值而悄悄改变算法。

这些是可核实的协议事项，不是已经证明的误差根因。尚未用固定输入的单变量实验
量化不同对齐/裁剪/权重的影响。7-Scenes `office/seq-06` 的大误差仍保留在完整18轨迹汇总中。

## 3. Table 1 位姿入口的数据要求与风险

### CO3D

官方 pose 配置使用 `Co3d_Multiview`，不是重建用的 Spann3R CO3D loader。
配置要求 processed root 中的 `selected_seqs_test.json`，RGB、相机 NPZ、深度和 mask，
评测尺寸 `(512,384)`。当前该目录和清单不存在，不能用任意几条 CO3D 视频冒充全量。
论文 §4.2 指明41类物体中的未见轨迹（unseen trajectories），不是41个未见类别；
本机尚无可据以核验这些类别/序列的作者原始processed选择清单。

代码另有一项需明确的采样风险：`_generate_combinations` 生成很多组合，
但 `_fetch_views_for_pool` 总是读取 `self.combinations[0]` 再加随机 jitter，
并可能用已有有效帧补足视角。它不能直接被描述成每次从全部轨迹均匀随机选10张。
原论文采样身份没有公开在本机产物中；不擅自修补后把新协议标成原协议。

[DUSt3R预处理](https://github.com/naver/dust3r/blob/main/datasets_preprocess/preprocess_co3d.py)
从 set lists 与 camera/depth annotations 选择序列，默认至多50条/类别、seed42、
质量过滤，输出测试清单。正式数据预算必须依据规定的选择清单计算。

### RealEstate10K

[`scripts/re10k_test_1800.txt`](scripts/re10k_test_1800.txt) 实际包含 **1,832个非空且唯一的视频ID**，
不是根据文件名推测的1,800。SHA256：
`6d55e2006523c2388f40f3d6a2a6bb2ef062790edb5dde9a10c1bf90f18c4796`。

原入口 [`fast3r_re10k_pose_eval.py`](scripts/fast3r_re10k_pose_eval.py)
绑定作者的 `/home/jianingy/...` 和 `/data/jianingy/...`，仅加载作者 Lightning
checkpoint/.hydra 配置，默认两 GPU；缺目录、TXT、RGB或匹配时间戳时会 `continue`。
只运行这个脚本再读取最后平均值，不能保证完整覆盖指定测试清单。
该脚本中的 `FlashDUSt3R` 类在仓库内存在，不能把它误报成不存在的导入。

Google[数据格式说明](https://google.github.io/realestate10k/download.html)规定时间戳与19列相机记录，
extrinsics 是 world→camera，须求逆变成 Fast3R 的 camera→world。
720MB官方包只有相机轨迹TXT，不含 RGB 视频；下载元数据不等于准备好测评数据。
当前目标路径下1,832个规定视频/相机记录对均未准备，未启动位姿评测，也没有RRA/RTA/mAA实测。

### 新的只读预检查

新增 [`scripts/check_pose_data.py`](scripts/check_pose_data.py)，先枚举指定清单，
检查独立10视角、时间戳/GT匹配、有限相机参数、RGB解码；CO3D检查全部清单帧的
RGB/depth/mask/NPZ。缺失或损坏返回 `incomplete`、CLI退出码2，不自动下载或静默跳过。
10项合成离线测试通过；这些测试不是真实位姿指标。
`ready` 只代表给定清单数据可读，不证明清单/采样与论文相同。

```bash
python scripts/check_pose_data.py --dataset co3d \
  --data-root /your/data/co3d_50_seqs_per_category_subset_processed
python scripts/check_pose_data.py --dataset re10k \
  --data-root /your/data/RealEstate10K/videos/test \
  --metadata-root /your/data/RealEstate10K/test
python scripts/test_pose_data.py
```

## 4. 下一步需要的具体资源

本节保留2026-10-01历史资源快照。2026-10-02用户扩容后空间限制已解除，见第5节；不再要求用户为此重复提供存储。

本次宿主检查：项目文件系统剩余2,550,480,896 bytes（约2.38GiB），
每次下载/预处理仍须预留1GiB。未发现已挂载的额外数据存储。
大 NTFS 分区未挂载，不把其容量当作已获授权的可用目录，也不改分区。

[CO3D官方](https://github.com/facebookresearch/co3d#download-the-dataset)列出的全体ZIP为5.5TB，
较小的single-sequence挑战子集为8.9GB；后者不是规定的Fast3R测试划分。
这不意味着 Table 1 一定要下载5.5TB。稀疏HTTP Range/逐序列处理可研究，
但正确序列选择、服务器支持与峰值预算尚未验证，不能假称已能在当前空间完成。
RealEstate10K RGB的选择帧大小、视频可用性也未验证。没有启动无预算的大下载。

继续数据阶段需要用户提供：**允许用于数据的、更大可写目录**（外置盘或已挂载分区），
或已有上述规定测试数据的路径。拿到目录后先计算所需划分与临时空间预算，
只下载必要部分；不要求盲目预留整个5.5TB。后续还需原脚本的路径/HF适配及可追踪采样，
通过预检查后才运行完整 Table 1。

独立训练消融仍需要各组权重及来源，或额外训练算力/明确预算授权。
单一公开权重不满足此条件，短训练不等同论文128张A100-80GB的完整训练。
附录C/D/E仍缺规定数据和可用的splatting/BA/RobustMVD完整评测环境，尚无新增成绩。
当前后台重建阶段已结束；到此是需资源选择的检查点，不是整篇论文完成。

## 5. 扩容后更新（2026-10-02）

初查项目分区约251G、可用122G。RE10K官方归档752332631bytes已核验MD5与SHA256，
1832个规定相机TXT全部解析通过，清单记录各文件SHA。仅代表metadata_prepared，RGB还未就绪。
pixelSplat作者test-only镜像55,604,889,849bytes已启动预算约束的后台下载；
服务器不支持Range，重启通过验证并重新传输已有前缀后追加，禁止把200响应直接附加导致损坏。
镜像coverage、分辨率、cropped intrinsics和官方GT一致性要在下载后验证，不自动认定等价。

新HF单卡位姿入口不再绑定原作者绝对路径/两GPU/Lightning checkpoint，
仍沿用原RE10K512×288 crop、first-view global focal以及公开PyTorch relative-pose/AUC函数。
固定逐scene随机采样、OpenCV seed与顺序PnP、16-mixed、chunk2都是记录在案的本机适配。
实际公开PnP mask为`conf>1.0`；函数虽然接收`min_conf_thr_percentile=85`，相关mask代码被注释，
因此不能称该PnP按85-percentile过滤（重建对齐的percentile85是另一条路径）。
失败identity fallback保留但单独计数；不将默认identity伪装为成功PnP。
恢复时校验协议/输入哈希，并从存储的pred/GT位姿重新计算指标。
正式全量RGB预检查当前仍失败，没有新增Table1成绩。

## 6. RGB归档完成与实际覆盖审计（2026-10-02 05:36）

完整test-only ZIP长度55,604,889,849bytes、SHA256
`ce351771c966fb25ef41efc561a313ef40607c9aa8ea904ed8d582b361408097`。
CRC核验后的index含7286个ID，但规定1832仅覆盖1756，缺76；
缺失ID及官方URL完整保留在`results/re10k_rgb_index_audit.json`，不改测试集合。
逐块safe weights-only+CRC/SHA读取，1756场景265447帧的mirror camera与官方TXT匹配，
全部官方时间戳/候选帧保存；原图尺寸不是一律360×640：1753场景360×640、
2场景338×640、1场景272×640（H×W）。原位姿loader按实际图像宽高缩放normalized K，
保留图像实际尺寸，不用拉伸/补边“修复”数据，也不将相机数字匹配当作裁剪协议等价证明。
首次固定尺寸假设检查停下后，旧inventory与原图保留，v2记录实际尺寸；
服务05:36:54正常退出0，独立核验所有已保存RGB长度/SHA、官方GT SHA和候选集合通过。
见`results/re10k_rgb_prepared_manifest.json`及`results/re10k_rgb_verification_20261002.json`。
58项离线测试通过，不是位姿成绩。当前有可用的covered数据，但全量仍不ready。

其他来源证据见`results/re10k_missing_source_audit_20261002.json`。
另一公开HF test index同样缺76；两个匿名YouTube探测遇429/登录错误，不能推导76全部永久缺失。
未使用账号、cookies或绕过限制。公开205GB test.tar.gz候选需要先审计来源/条款与
有界流式读取可行性，不落盘整包，也不假称缺失集合可覆盖。仍可研究公开来源，暂不要求用户动作。

## 7. 公开候选来源的有界读取（2026-10-02）

DavidYan2001镜像固定revision的模型卡仅标CC-BY-4.0，没有详细解码/裁剪来源说明。
实际HTTP返回206和精确Content-Range；2MiB格式探测首个PNG解码为640×360，
随后真实8MiB流式探测见`results/re10k_missing_prefix_probe_20261002.json`，
仅观察1个场景，尚未观察到76缺失ID中的任何一个。前缀SHA不是完整205GB来源SHA。
`stream_re10k_missing_candidates.py`只将缺失ID匹配官方timestamp的PNG保存至独立候选目录，
保留PNG原始字节，不覆盖1756场景已有JPEG；单图8MiB、总保存16GiB、磁盘留1GiB。
重试精确HTTP偏移，检查ETag/Range；gzip完整遍历后才检查整包LFS SHA。
完整CRC/SHA及timestamp集合通过也不自动证明图片/GT裁剪等价，候选不会自动提升为正式数据。
进程重启需从gzip开头重读网络，核对已存PNG后跳过写入，不声称能从gzip任意偏移恢复解码。

CO3D官方`co3d/links.json`（README的旧路径不是当前真实文件路径）给出51类元数据ZIP。
实际apple_000.zip为31,587,519bytes，CRC及官方SHA匹配；含sequence/frame annotations、
fewview/manyview set lists和LICENSE，不是RGB/深度包。元数据准备仅请求各类_000.zip，
book_000.zip实际86,868,326bytes，单包/JSON上限据此调整为128MiB，
仍先核算保留元数据ZIP及解码清单的保守预算，固定CO3D revision `eb51d7583c56ff23dc918d9deafee50f4d8178c3`。
按固定DUSt3R revision `4c24a6ebf04809f2cfe59915e51779c8984aaa40`重建公开选择：
fewview_train清单的test键、quality严格>0.5、每类最多50序列、seed42+category index。
记录ZIP内set-list顺序；原作者os.listdir顺序/processed清单未提供，不能称作者划分已等价复现。
某类别没有fewview_train或没有合格test轨迹时保留为空，不用challenge单序列子集替代。
公开eval配置的`100 @ Co3d_Multiview(...)`是ResizedDataset长度100，
不是自动完整遍历全部序列；正式入口需要明确记录序列集合与实际采样，不能从“100”猜全量。
两个后台任务仅准备/审计数据，没有新增Table1位姿分数。

元数据准备最终正常退出0：51类官方ZIP共1,315,929,722bytes，独立再次核验SHA/参考哈希及
质量过滤/种子选序列/帧顺序。公开默认规则得到51非空类别、2511序列、498757候选帧，
与论文41类不同，不能未经证据删除10类或把51类当论文全量。该差异是需要继续解决的协议前提，
不是模型测评失败，也不是所有数据下载完成。小汇总与独立核验见
`results/co3d_test_selection_summary_20261002.json`、`results/co3d_test_selection_verified_20261002.json`。

### 06:21续接：41类来源与ZIP Range预算

[PoseDiffusion官方数据类](https://github.com/facebookresearch/PoseDiffusion/blob/b138198e2891a0f1a1c3435614b9490adb7fd4d6/pose_diffusion/datasets/co3d_v2.py#L398)
明确列出41个`TRAINING_CATEGORIES`与10个`TEST_CATEGORIES`；
[固定评测配置](https://github.com/facebookresearch/PoseDiffusion/blob/b138198e2891a0f1a1c3435614b9490adb7fd4d6/cfgs/default_test.yaml)
使用`category: seen`与`num_frames: 10`。这与Fast3R §4.2描述的“41类物体中的未见轨迹”相符，
但把这份名单对应到Fast3R是基于所引评测协议的推断，并非作者processed清单确认。
`audit_co3d_41_categories.py`固定源revision和源码/配置SHA，只AST解析字面量，不执行下载的代码。

排除的10类是ball、book、couch、frisbee、hotdog、kite、remote、sandwich、skateboard、suitcase。
在已独立验证的51类候选上仅过滤类别、不改变种子索引/序列/帧顺序，真实得到41类、2011条轨迹、
399204候选帧。parkingmeter为41条、tv为20条，其余39类各50条，不凭41×50猜为2050。
`results/co3d_seen41_protocol_20261002.json`记录证据、各类数量及manifest SHA；
完整候选仍留`data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json`，不冒充作者划分。

apple官方6个图像ZIP真实HTTP Range目录探测完成：传输83,941,069bytes目录，
目录标示9563组RGB/depth/mask、合计6,073,570,411bytes（不是已下载图像量）。
固定精确206/Content-Range/ETag并限制每包目录读取32MiB；ZIP64由标准zipfile解析，
禁止路径穿越、重复名称、链接和加密成员。这里只核实目录名称/标示长度，未读取成员或验证其CRC，
也未完成大ZIP全SHA；探测的计数相同本身不证明跨ZIP无重复。

因此另启动`fast3r-co3d-seen41-storage.service`，逐类精确核对所有候选成员路径、跨ZIP无重复，
记录每类目录预算后再汇总全41类；`results/co3d_seen41_storage_progress/`保存断点。
它不下载RGB，不运行GPU，不改变帧采样；未知的Fast3R原始清单、100@评测范围、相机裁剪转换、
正式采样/指标口径仍须继续审计。全量预算未出前不外推apple结果成其他40类的真实预算。

06:51续接诊断了目录预算服务的实际停止：banana_001.zip有229525个成员，超过初版200000。
独立只读Range解析得到29,035,659bytes目录、峰值RSS683928KiB；宿主约24GiB可用内存。
成员上限调整至400000，仍保持32MiB字节/精确Range/ETag及路径等安全上限。
不从这个错误断言ZIP数据损坏；不改模型、样本、类别或GPU流程。
新v2检查点显式核验固定v1代码SHA、输入SHA、路径SHA和逐包/逐类汇总后导入apple/backpack，
不覆盖旧数据、不重读已完成两类网络。banana恢复后通过原位置；最终41类预算仍未生成。
恢复证据`results/co3d_storage_recovery_20261002.json`与v2输出路径见CONTINUATION。

07:21续接发现v2在bowl目录触发字节预算。全41类235个官方数据ZIP的EOCD/ZIP64尾部预检
通过：每包<=65557bytes、合计15405895bytes，最大目录37498281bytes、最大300491成员。
footer解析检查单磁盘、记录完整/范围、目录偏移；没有读取图像或声称大ZIP全SHA通过。
固定候选和源URL顺序、所有逐包/全部sum/max均独立核验。结果见`results/co3d_zip_footer_preflight_20261002.json`。
目录v3硬上限64MiB/400000成员，逐包预算仅实际目录大小+65685bytes；
重读/复用均对照新footer的源大小/ETag/成员数。新v3指纹记录footer报告SHA。
8类v2旧记录通过批准的固定代码SHA、输入/路径SHA及汇总校验迁入，不宽泛接受代码更改。
v1/v2历史不覆盖，失败bowl重新读目录并完成。`results/co3d_storage_footer_recovery_20261002.json`
仍是恢复快照，不是全41类中央目录预算或RGB/GT就绪证明，更不是正式测评结果。

### 08:21续接：全目录、源采样方法与相机metadata

完整v3目录报告与独立验证核对41类/2011轨迹/399204组每种路径和235包汇总。
原始成员标示RGB36.30GB、depth246.35GB、mask14.55GB，共297.20GB（276.79GiB）；
记录的2,832,716,261bytes目录传输含旧记录复用，不是本轮新增网络量。
目录核验不包括图像CRC/解码、大ZIP SHA或作者processed划分等价性。

`audit_co3d_sampling_trace.py`调用仓库组合生成、ResizedDataset.set_epoch和_get_views，
仅替换_load_view_data为明确all-valid stub。组合seed42、dataset seed777、epoch0为审计选择。
10000个组合使base长度20110000；实际仍读combinations[0]，不按mapped组合编号选帧。
100@名义100次为99轨迹/38类/918唯一帧，58次重复；全部2011轨迹适配追踪为18588唯一帧、1149次重复。
真实mask/depth无效、补采和scene重试未执行，不能直接当正式全量协议或稀疏下载ready证明。

相机metadata审计核验完整399204个官方annotation、set-list到image path的关系与NDC相机转换。
38869项filename编号不同于annotation编号，如apple/110_13072_25709的annotation18为frame000019.jpg；
按官方对应关系连接，不修改GT。原depth.scale_adjustment=1、ndc_isotropic和有限R/T/K均通过。
PyTorch3D row-vector转换为OpenCV w2c：前两列R与前两项T翻转，R转置；c2w为此矩阵的逆。
car有12项原平移绝对值>1e6，最大8.296e16；绝对逆残差16、全数据最大component-scaled残差6.758e-16。
新v2保存两种残差和大坐标计数，不做静默归一化/删场景；float32 metric稳定性仍待独立验证。
完整metadata核验不是RGB/深度解码、processed crop等价或正式位姿分数，旧v1记录和日志保留。

### 09:21续接：相对位姿数值诊断

新增CPU-only检查覆盖41类/2011轨迹/399204帧的39,624,443个同轨迹候选相机对。
遵循固定DUSt3R先构造float32 w2c、再求逆的顺序，与float64对应计算比较；
相对位姿和方向角仍调用公开Fast3R函数，没有重设原点/尺度或丢弃轨迹。
结果见`results/co3d_pose_precision_20261002.json`，这是GT诊断，不是网络推理成绩。

直接float64相机中心相同1181对；相对矩阵计算float32零平移1375对，
194对在float64产品中非零、float32为零，不能因此认定194条物理基线被截断。
可定义方向的39623068对中3301对精度间角差>1°，最大89.9804°。
另用基线大于1e-6×max(1,两个中心范数)作诊断分组，39619868对中仍340对>1°，
最大5.9845°；该1e-6不是论文筛选条件，不用于正式指标过滤。
GT与自身比较在1375对上产生>1°平移角（最大90°），反映零向量与数值实现行为。
旋转自检最大0.4070°与公开so3近零角外推有关，不以改公式“修复”历史成绩。

数学上C_i^{-1}C_j的平移是R_i^T(t_j-t_i)，但公开实现先分别计算负平移再矩阵乘积，
float32舍入与消减可能不同；大坐标逆的尺度残差小不代表小基线角度稳定。
保留全部源数据与pair计数，报告按类别sum/max和201条风险轨迹总数（仅列前20条）。
这个全候选诊断不等于原100@实际draws，也不证明CUDA/TF32或预测位姿稳定；
下一步真实输入评测仍须保留公开metric与显式诊断，不能用这些角差冒充模型RRA/RTA/mAA。

### 09:51续接：按需成员与十帧参考预处理比较

新Range缓存核对原候选路径、固定footer大小/ETag与精确206范围；从已读中央目录的内存重放
提取成员信息，然后验证local header/ZIP64/压缩边界、有界inflate、CRC和原字节SHA。
成员CRC及ETag不是完整大ZIP的SHA，所有报告保持该标志false；没有改动测试候选manifest。
单成员32MiB，raw/index缓存总计2GiB，写入保留至少1GiB，不自动删除历史数据。

`probe_co3d_preprocess.py`只用apple/608_95658_192033前十个原候选帧，
三个原始成员×十帧全部解码；与固定DUSt3R原prepare_sequences在同一运行库上的真实输出比较，
JPEG/PNG bytes和NPZ arrays一致。NPZ ZIP时间戳不纳入数组等价性；不会靠NPZ二进制相同猜数值一致。
FP16原始depth位模式与processed UINT16/max-depth编码严格区分；相机仍先float32 w2c再inverse。
主点crop后resize的K使用Colmap/OpenCV半像素转换，不以单纯fx/cx乘scale替代。
[固定cropping参考](https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/dust3r/datasets/utils/cropping.py)。
原始参考副本只放data、保留来源许可，不提交数据集；衍生函数许可见NOTICES_CO3D_PREPROCESSING.md。

独立检查raw/processed哈希与数组后，只执行公开_load_view_data，原202帧候选pool保持不变。
near-square输入可按rng输出512×384或384×512，保留实际朝向，不将初次尺寸断言错误解释成数据损坏。
十帧有效GT深度比例约5.22%–5.69%，不是网络深度/位姿准确率。
尚未验证_get_views的真实补采/重试、全量crop或100@作者draws，未产生Table1正式分数。

### 10:21续接：严格lazy adapter与一个真实mapped draw

固定loader SHA+AST后，仅从本地_load_view_data移除except Exception -> None；原文件未改。
_get_views/_fetch_views_for_pool/base.__getitem__继承不变，IO/CRC/解码异常直抛，
裁剪后真正零深度仍None并标记invalidate；raw全零深度保留max0/quantized0，非有限/负值报错。
零深度补采与scene retry只做合成测试，不假冒真实数据实验。
完整41/2011/399204池保持，100@epoch0 mapping[0]=12303633，bicycle原133帧pool
得到144,42,41,151,20,28,52,144,10,1。重复144保留，全部有效，无实际补采/换场景。
完整输出（含tensor/pts3d/valid_mask/rng）与原loader精确相同；独立固定预处理9unique帧
JPEG/PNG bytes、NPZ arrays一致，GT覆盖13.25%–47.59%，没有网络位姿分数。
组合seed42、dataset seed777、epoch0、per-request Python seed42+base_index为声明适配。
单样本状态不证明作者draw身份，也不证明连续100@跨样本invalidate/scene tracker等价。
首次NumPy idx序列化失败保留未完整文件，用v2报告安全恢复；缓存重试0bytes不是首次下载流量。
没有完整ZIP SHA证明，独立fixture从未替换原候选manifest。

### 10:51续接：CO3D公开HF方向契约与真实单样本位姿

固定公开HF配置head_args.landscape_only=False：输入landscape tensor [1,3,384,512]，
原true_shape [512,384]，实际raw global confidence [1,512,384]。发布版评测
correct_preds_orientation对portrait转置一次，结果[1,384,512]，不是校正后仍为原portrait。
首次新入口断言方向相反而退出，无结果JSON；旧预检查/失败代码/日志保留，修复后另写v2。
没有改发布head/方向算法，也不将断言修复当数学/精度问题已解决。

网络仅RGB/true_shape；GT K参与crop，但GT不供模型或focal/PnP。初始化后重设seed12303675，
16-mixed、chunk2、原global focal/conf>1 PnP、CPU公开相对角、45pairs和重复pair均保留。
11:05:17服务退出0，真实结果重算通过：PnP失败0/10，但RRA@30=0%、RTA@30=13.3333%、
mAA@30=0%，focal约27.9829px，不能称论文效果复现成功。GT-self RTA@30=97.7778%
仅诊断重复零基线角，不用作模型成绩或删pair依据。方向/focal单变量诊断须另写结果、保留
低分基线；未证明方向就是低分根因。仅一个mapped draw，连续100@共享状态、原作者
processed划分/权重对应仍未确认。
