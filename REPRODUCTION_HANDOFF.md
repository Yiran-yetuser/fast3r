# Fast3R 论文复现：结项交接

> 2026-10-07结项清理：用户授权删除本地数据/下载包及模型权重，定时续接已关闭；代码、Notebook、历史结果和科学图保留。本地数据依赖评测与推理须重新下载才能重跑，不将历史输出误标为当前执行。记录见`results/project_storage_cleanup_20261007.json`。

更新日期：2026-10-03。项目结论是**公开权重部分复现**，不是整篇论文复现完成。详细实验、参数和每轮证据见[逐项复现记录](PAPER_REPRODUCTION.md)、[Notebook](fast3r_reproduction.ipynb)和[续接记录](CONTINUATION.md)。

## 复现流程

1. **把论文主张拆成可核对的实验。** 按表格和图逐项记录要验证的指标、数据划分、视角采样、模型权重、随机种子和硬件要求。把“论文声称什么”与“本地能测什么”分开。
2. **确认代码和权重身份。** 检查公开仓库实现、配置与checkpoint，记录版本和SHA256。公开权重可用于推理，但没有证据证明它与论文各张表/消融使用的训练权重完全对应。
3. **准备数据并守住规定划分。** 用官方测试划分检查场景ID、帧号、相机参数、GT、RGB解码和文件完整性；生成manifest、CRC/SHA记录。缺失场景不从分母里删掉，不用相近子集冒充完整测试集。
4. **运行主重建基线。** 在公开权重下运行DTU 22场景、Neural RGB-D 9场景和7-Scenes 18条测试轨迹。保存逐场景结果、seed、输入视图、分辨率、head分块、环境与GPU信息，再按论文各表指定的median/mean和缩放规则聚合。
5. **做可比较的配对分析。** local/global用同一次前向和同一份输入/GT，避免把模型随机性误当head差异。DTU `scan1/10/11`的阈值诊断也固定一份预测，只改变评测阈值，并核对输入标签、预测SHA和历史baseline。
6. **独立复核并写清边界。** 从逐场景JSON重新聚合，与保存值逐项比对；留存失败、有限性、覆盖率和资源证据。报告结果时区分“完整跑完数据划分”“与论文数值接近”“协议等价”和“正式复现成功”。它们是不同结论。

## 已完成与实测结果

| 项目 | 已完成内容 | 结果边界 |
| --- | --- | --- |
| 推理和损失 | 验证多图输入的global/local点图与置信度推理链；对归一化点图损失做数值和梯度检查 | 没有据此声称重新训练模型 |
| Table 3 / NRGBD | 9场景运行和聚合复核 | Accuracy 4.0165 对论文3.40（高18.13%）；Completion 1.2001 对1.01（高18.82%） |
| Table 3 / 7-Scenes | 官方18条测试轨迹运行和聚合复核 | Accuracy 3.3035 对1.58（高109.08%）；Completion 3.1058 对0.93（高233.95%） |
| Table 4 / DTU | 22场景运行和聚合复核 | Accuracy 2.0827 对1.706（高22.08%）；Completion 1.0311 对0.857（高20.31%） |
| Table 5 | DTU、NRGBD、7-Scenes同次前向的local/global配对分析 | 本地结果没有在三个数据集上都复现论文的local优势方向 |
| Figure 5 / 视角数 | DTU 3/5/10/20视角本地适配实验，88次评测前向 | 本地均匀采样不同于论文关键帧采样；不能把结果当作论文原设置复现 |
| 阈值敏感性 | DTU `scan1/10/11`各1次前向，比较7组阈值 | 三场景均值中，85/0的Acc/Comp为6.9304/2.4108，metric75为3.4452/13.1753；Accuracy距离下降同时Completion明显变差。这是敏感性诊断，不是调参建议 |

指标与论文有明显差异。来源/协议审计确认了公开代码和权重身份的一部分，但没有定位所有差异的因果，也没有证明公开checkpoint与论文各实验权重等价。

## 尚未完成，以及停止位置

- **Table 1 / Figure 4 位姿评测：** RE10K规定1,832场景中目前准备1,756，缺76，尚无完整正式RRA/RTA/mAA。已完成的205GB来源扫描没有补回缺失ID；429GB公开候选集也未核实覆盖，不曾下载。CO3D按用户要求暂停，相关本地数据已删除；历史100请求候选分数不算正式Table 1。
- **训练型实验：** Figures 6–7、§5.2/附录A–B需要不同训练视角、模型规模或训练数据规模对应的独立权重；Figure 8目前只核对了位置插值机制，没有独立训练消融。论文完整训练配置是128张A100-80GB、174K steps，本项目没有执行。
- **附录 C–E：** Gaussian Splatting、GS-BA和RMVD/Table 7所需依赖及规定数据没有准备，论文实验未运行；已有点云展示只覆盖部分可视化工作。
- **性能与协议完全等价：** 本机单卡性能/视角实验不是论文A100多卡、大视角设置；关键采样、训练权重映射和所有重建差距仍未完全对齐。

到这里停止整篇复现是资源与数据边界判断：缺少Table 1数据、用户已暂停CO3D，训练消融和完整训练又需要研究规模算力。继续硬跑会得到不完整或不可比的结果。以后若要继续，应先解决具体数据/权重来源与预算，再一次只改变一个因素；CO3D仍需用户重新授权。

## 本科生可以带走的科研方法

- **先操作化主张。** 把论文中的“更快”“更准”“扩展性更好”拆成输入、对照、统计量和可证伪的问题。
- **建立数据血缘。** 固定划分、保存manifest与哈希、保留原始ID；记录缺失值和异常，不静默删难例。
- **控制变量。** 同一输入、同一前向再比较输出分支；做消融时一次只改变一个因素，并确认预测或配置没有意外变化。
- **重做统计而不只看总分。** 从逐场景数据独立计算median/mean，核对缩放、单位、分母和聚合顺序。
- **区分证据等级。** 把“代码看起来支持”“数据准备完成”“本地跑通”“协议一致”“复现论文结果”分别报告。
- **保留负结果。** 指标对不上也保留，定位差距并设计下一步检验，不按GT挑参数凑分。
- **为资源设门槛。** 前向前检查显存/磁盘、记录预算和断点；遇到超出项目资源的工作，先量化缺口，不用小实验冒充大规模训练。

## 简历写法

**中文（稳妥版）**

> 基于Fast3R官方代码与公开权重开展部分复现，搭建可审计的多视图3D重建评测流程，完成DTU 22场景、Neural RGB-D 9场景及7-Scenes 18条官方测试轨迹评测；实现数据清单/哈希核验、固定seed与逐场景指标复核，并完成local/global同次前向对照和DTU三场景阈值敏感性分析。量化了本地结果与论文Table 3–5的差距，记录了缺失数据、权重映射和训练资源限制。

**英文（稳妥版）**

> Conducted a partial reproduction of Fast3R using the official codebase and public checkpoint. Built an auditable multi-view reconstruction evaluation workflow covering 22 DTU scenes, 9 Neural RGB-D scenes, and all 18 7-Scenes test trajectories; added data-manifest/hash checks, seeded per-scene evaluation, same-forward local/global comparisons, and a three-scene DTU threshold sensitivity study. Quantified deviations from the paper and documented unresolved dataset, checkpoint-provenance, and compute limitations.

不要写“完整复现Fast3R论文”或“达到论文指标”。如果简历空间有限，可以保留第一句、数据规模和一项可量化分析，并在面试中主动说明Table 1及训练消融未完成。

## 相关记录

- [论文表格、协议与逐项结果](PAPER_REPRODUCTION.md)
- [Notebook和保存的分析输出](fast3r_reproduction.ipynb)
- [当前范围与恢复边界](CONTINUATION.md)
- [RE10K补缺来源审计](results/re10k_missing_source_followup_20261003.json)
