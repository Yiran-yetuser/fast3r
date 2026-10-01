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
论文 §4.2 指明41个未见物体类别；本机尚无可据以核验这些类别/序列的原始选择清单。

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
