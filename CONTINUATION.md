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
6. 然后推进CO3D规定41未见类别选择、预算与公开权重适配，以及资源允许的附录实验。
   独立训练消融仍需要独立权重或训练资源，未做实验不打勾。

## 额度恢复机制与边界

Codex当前聊天heartbeat `fast3r` 已创建并读回核验ACTIVE，每30分钟接续。
本地机器须开机、应用保持运行；额度不足时定时任务也可能不能执行。
这是后续触发时尝试接续，不保证精确重置时刻或无缝恢复。不购买额度、不兑换reset权益。
已启动systemd下载/评测与模型额度独立，但断电/休眠/网络故障仍会中断。
任务提示要求无变化安静，仅新阶段完成、失败或需用户动作才通知。
