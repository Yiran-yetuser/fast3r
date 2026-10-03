<div align="center">

# ⚡️Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass


${{\color{Red}\Huge{\textsf{  CVPR\ 2025\ \}}}}\$


[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2501.13928)
[![Project Website](https://img.shields.io/badge/Fast3R-Website-4CAF50?logo=googlechrome&logoColor=white)](https://fast3r-3d.github.io/)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=Gradio&logoColor=red)](https://fast3r.ngrok.app/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512/)
</div>

> Latest source sampling checkpoint (2026-10-03, §4.2/Table1 prerequisites): the released
> 1,000@ sampler's all-valid nominal trace covers 51 categories, 836 trajectories and 8,835 unique
> frames; 581 requests contain duplicate views. The old 1,000-distinct-sequence engineering proposal
> is retained but will not be used for evaluation; cache ceilings are not a storage fit proof.
> [Source-bound report](results/co3d_51_released_sampler_20261003.json) and host-executed
> [Notebook](fast3r_reproduction.ipynb) retain all97 prior cells. The directory audit has now
> completed and been [independently verified](results/co3d_51_source_storage_verified_20261003.json):
> nominal raw6.54GB, conservative retry/jitter closure76.46GB; 38 frozen indices reused/13 filled.
> The [first actual request](results/co3d51_source_inputs_prefix1_verified_20261003.json) (10views/7unique)
> passed offline tensor/GT/RNG/state replay. An independent CPU service continues the1,000-request
> preparation under NEW raw8GiB/processed3GiB/journal4GiB caps and1GiB reserve; old caches remain frozen.
> These are fail-closed ceilings, not a guarantee all retries fit. No new pose score/GPU evaluation yet.
> See [continuation](CONTINUATION.md). Exact author split/RNG and formal Table1 remain unresolved.

> Archived local diagnostic (2026-10-03): the public HF portrait path has distinct token-grid and
> pixel-projection interface mismatches. With fixed frames/GT and five new forwards, forced landscape
> cropping improved two selected probes' mAA from 0 to 59.7133% and 12.4014%; a landscape control
> stayed at 92.2581%. Shape metadata alone did not fix either low-score probe. Changed crops mean
> changed inputs/forwards, not a same-forward ablation or an unbiased Table 1 result. See the
> [independent proof](results/co3d_landscape_input_diagnostic_verified_20261003.json) and
> [executed notebook](fast3r_reproduction.ipynb). Original reports/data/weights are retained;
> next is the 51-category/1,000-request candidate protocol and storage budget.

> Archived protocol update (2026-10-03): the author's public clarification describes 1,000 CO3D
> test requests and potentially more than 41 categories, unlike the released 100-request config.
> The archived seen-41/100-request results remain a candidate adaptation, not confirmed Table 1.
> Three selected same-forward focal/mask diagnostics reproduced the baseline exactly; focal search
> removed one probe's 10 PnP fallbacks but did not improve its zero mAA. These selected probes are
> not a benchmark average. See [protocol evidence](results/co3d_author_protocol_update_20261003.json),
> [verified diagnostics](results/co3d_candidate_pnp_diagnostic_verified_20261003.json), and the
> [executed notebook](fast3r_reproduction.ipynb). Full-paper reproduction remains incomplete.

![Teaser Image](assets/teaser.png)

Official implementation of **Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass**, CVPR 2025

*[Jianing Yang](https://jedyang.com/), [Alexander Sax](https://alexsax.github.io/), [Kevin J. Liang](https://kevinjliang.github.io/), [Mikael Henaff](https://www.mikaelhenaff.net/), [Hao Tang](https://tanghaotommy.github.io/), [Ang Cao](https://caoang327.github.io/), [Joyce Chai](https://web.eecs.umich.edu/~chaijy/), [Franziska Meier](https://fmeier.github.io/), [Matt Feiszli](https://www.linkedin.com/in/matt-feiszli-76b34b/)*

## Installation

本地复现进度（2026-10-02）：已完成DTU22场景、NRGBD9场景、7-Scenes全部18测试轨迹的公开权重评测，尚未匹配论文数值或完成整篇实验。RealEstate10K规定1832个相机记录已齐备、55.60GB候选RGB归档已下载；已核验1756场景/265447帧的RGB、官方GT与完整候选集合，仍缺76场景，继续研究补齐来源。新单卡HF位姿入口支持固定采样和断点恢复，正式Table1指标尚未运行。[逐项论文对应](PAPER_REPRODUCTION.md)、[Notebook](fast3r_reproduction.ipynb)、[下一步检查点](CONTINUATION.md)。

CO3D §4.2/Table1类别审计：已找到固定PoseDiffusion协议的41类seen名单，生成2011轨迹/399204候选帧的有来源候选，尚未确认Fast3R作者逐轨迹清单等价。41类完整目录预算已结束；按需数据接线继续，不下载整个大ZIP、不冒充全量图像或正式成绩就绪。Notebook保存真实分析输出。

目录审计的banana成员数量阈值错误已诊断并安全恢复：保留旧检查点，核验复用apple/backpack，使用新v2结果路径。88项离线测试通过；[恢复证据](results/co3d_storage_recovery_20261002.json)不是正式位姿成绩。

07:21历史恢复阶段：已完成41类/235个官方ZIP的[尾部大小预检](results/co3d_zip_footer_preflight_20261002.json)，按每包实测目录大小安全读取，复用8类旧目录记录并通过bowl原失败点。当时95项测试通过、目录预算仍在后台；最新完成状态见下一条。

08:21已归档：CO3D完整[目录预算与独立核验](results/co3d_storage_budget_verified_20261002.json)已完成，41类/2011轨迹/399204组每种成员，原始大小297.20GB。全部候选[相机metadata核验](results/co3d_camera_metadata_audit_v2_20261002.json)完成，保留编号差异与car大坐标风险；名义源采样追踪不作为论文成绩。

09:51已归档：全候选[GT精度诊断](results/co3d_pose_precision_20261002.json)已记录单精度风险，不宣称整体稳定。十帧真实CO3D [Range/CRC/参考预处理接线](results/co3d_preprocess_probe_20261002.json)与[独立loader核验](results/co3d_preprocess_probe_verified_20261002.json)通过，重跑缓存网络0bytes；未改变原候选池。这不是100@采样或全量Table1成绩。128项离线测试通过，Notebook保存真实输出；衍生预处理代码[许可与来源](NOTICES_CO3D_PREPROCESSING.md)另列。
10:21新阶段：完整候选池的严格lazy loader完成[一个真实100@映射样本](results/co3d_lazy_draw0_v2_20261002.json)，10views/9unique，重复视图保留；完整输出与原loader相同，[独立固定预处理核验](results/co3d_lazy_draw0_verified_20261002.json)通过。不是全部100@或Table1分数。138项离线测试、19个paper分析单元保存真实输出；[JSON序列化安全恢复](results/co3d_lazy_serialization_recovery_20261002.json)保留历史，没有重下载或改原候选pool。

```bash
# clone project
git clone https://github.com/facebookresearch/fast3r
cd fast3r

# create conda environment
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r

# install PyTorch (adjust cuda version according to your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt

# install fast3r as a package (so you can import fast3r and use it in your own project)
pip install -e .
```

Note: Please make sure to NOT install the cuROPE module like in DUSt3R - it would mess up Fast3R's prediction.

## Demo

Use the following command to run the demo:

```bash
python fast3r/viz/demo.py
```
This will automatically download the pre-trained model weights and config from [Hugging Face Model](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512).

The demo is a Gradio interface where you can upload images or a video and visualize the 3D reconstruction and camera pose estimation.

`fast3r/viz/demo.py` also serves as an example of how to use the model for inference.

<div>
  <img src="assets/fast3r_demo_upload.gif" width="45%" alt="Demo GIF 1" />
  <img src="assets/fast3r_demo_control.gif" width="45%" alt="Demo GIF 2" style="margin-left: 5%;" />
  <br>
  <em>Left: Upload a video. Right: Visualize the 3D Reconstruction</em>
</div>

<details>
<summary>Click here to see example of: visualize confidence heatmap + play frame by frame + render a GIF</summary>
<div style="display: flex; justify-content: center;">
  <img src="assets/fast3r_demo_coloring.gif" width="100%" alt="Demo GIF 3" />
</div>
</details>

## Reproduction Notebook

For a step-by-step, educational reproduction of the local inference pipeline, see [`fast3r_reproduction.ipynb`](fast3r_reproduction.ipynb) and the accompanying [`REPRODUCTION.md`](REPRODUCTION.md). The notebook covers video frame extraction, one-pass multi-view inference, fast-PnP camera estimation, confidence visualization, point-cloud export, and a small view-scaling experiment.

The notebook records inference and dataset evaluation separately. Released HF weights can be evaluated through the official inference wrapper and metric implementation; the original `fast3r/eval.py` entry point expects a Lightning checkpoint. See the section-by-section audit in [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md) for completed experiments and remaining requirements.

When only the public Hugging Face checkpoint is available, [`scripts/fast3r_hf_dtu_eval.py`](scripts/fast3r_hf_dtu_eval.py) evaluates DTU with the repository's official reconstruction metric implementation. See [`REPRODUCTION.md`](REPRODUCTION.md) for the dry-run, single-scene, and full-22-scene commands.

### Completed DTU run

The full 22-scene run completed locally on 2026-09-30 with the public Hugging Face checkpoint on an RTX 5070 Ti Laptop GPU (CUDA, 16-mixed, 512 resolution, 10 views per scene). The aggregate means are recorded in [`demo_outputs/paper_eval/dtu_all.json`](demo_outputs/paper_eval/dtu_all.json):

| Metric | Mean over 22 scenes |
| --- | ---: |
| Accuracy | 5.2460 |
| Accuracy (median) | 2.9306 |
| Completion | 3.6110 |
| Completion (median) | 1.8200 |
| Normal consistency 1 | 0.6720 |
| Normal consistency 1 (median) | 0.7526 |
| Normal consistency 2 | 0.6364 |
| Normal consistency 2 (median) | 0.7041 |

These are local results from the public HF checkpoint and the repository's official metric implementation. Table 4 reports **medians**, so the corresponding local values are **2.9306 / 1.8200**, compared with the paper's **1.706 / 0.857**. The paper's numerical result has not been matched. Weight provenance and the full evaluation protocol still need to be checked.

The notebook now includes paper references, seeded DTU view/head ablations, warmed-up timing and VRAM figures. Run [`scripts/fast3r_paper_experiments.py`](scripts/fast3r_paper_experiments.py) to reproduce the local adaptations of Sections 4.1, 5.1 and 5.4.

### Completed Neural RGB-D run (Section 4.3 / Table 3)

The 2026-10-01 local run completed all 9 scenes (278 sampled views) with the public HF weights, stride 40, seed 42, CUDA 16-mixed and DPT head chunks of 2. [`results/nrgbd_seed42_stride40.json`](results/nrgbd_seed42_stride40.json) contains finite per-scene metrics and verified aggregate means. The mean of scene median distances, multiplied by 100 as in Table 3, is **4.0165 / 1.2001** (Accuracy / Completion), versus paper references **3.40 / 1.01**. These local errors are 18.13% / 18.82% higher; the paper numbers have not been matched, and correspondence of public weights to the specific paper checkpoint remains unverified. The notebook saves actual result tables; see [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md) for protocol details and remaining experiments.

[`scripts/queue_nrgbd_reproduction.sh`](scripts/queue_nrgbd_reproduction.sh) verifies the extracted scene/frame/pose inventory before downloading. Complete extracted data can be reused without the ZIP or preparation manifest. This checks layout and nonempty files, not image decoding or CRC.

### Paired NRGBD result and prepared 7-Scenes data

The complete 9-scene report is [`results/nrgbd_paired_seed42_stride40.json`](results/nrgbd_paired_seed42_stride40.json). All local metrics exactly reproduced the earlier seeded run. Under the Table 5 **mean-distance x100** convention, local Accuracy/Completion are **9.7108 / 3.1292**, versus global **9.2594 / 3.1789**. Local improves Completion and normal consistency but not Accuracy, so this run does not fully reproduce the paper's local-head advantage. These means must not be compared to Table 3 medians.

All official 7-Scenes TestSplit data are prepared: **7 scene categories, 18 trajectories, 850 stride-20 views**, verified against splits, frame inventories and sequence CRC/SHA256 records in [`results/7scenes_data_manifest.json`](results/7scenes_data_manifest.json). The complete paired GPU evaluation finished successfully on 2026-10-01 at 12:44:47 Asia/Shanghai. [`results/7scenes_paired_seed42_stride20.json`](results/7scenes_paired_seed42_stride20.json) passed inventory, seed/view-count, finite-metric and aggregate validation via `python scripts/validate_7scenes_report.py`.

For **Table 3 median distances x100**, local Accuracy/Completion are **3.3035 / 3.1058**, versus paper references **1.58 / 0.93** (errors 109.08% / 233.95% higher). For **Table 5 mean distances x100**, local scores are **6.3613 / 7.0516**, versus global **6.9567 / 6.3510**; local improves Accuracy but not Completion. These results do not match the paper's scores or fully reproduce its local-head advantage. The notebook saves actual tables and a [per-trajectory difference plot](results/figures/7scenes_paired_heads.png). Public-weight provenance and protocol equivalence remain unverified; complete benchmark coverage does not imply complete-paper reproduction.

### Next paper experiments (reconstruction runs complete; paper reproduction incomplete)

New Table 1 data-source work uses a bounded HTTP Range/gzip reader to stage only
the 76 missing RE10K IDs as original PNGs in a separate candidate directory.
A real 8 MiB prefix probe succeeded, but it does **not** establish missing-ID
coverage or image/GT crop equivalence. Full-source auditing runs independently
of Codex; it never stores the entire 205 GB archive or overwrites prepared JPEGs.
CO3D metadata-only preparation pins official ZIP SHA/CRC and the public DUSt3R
selection source; original Fast3R split equivalence and RGB/depth readiness are
still unverified. The paper describes unseen trajectories from 41 categories,
not 41 unseen categories. All 73 offline tests pass; none is a pose score.

Official CO3D metadata preparation and independent SHA/selection checks are now
complete: 51 metadata ZIPs (1,315,929,722 bytes), yielding **51 categories / 2,511
sequences / 498,757 candidate frames** under the pinned default public rule.
This is **not** the paper's 41-category selection. Only compact count/SHA reports
are committed; RGB/depth and the exact paper subset still require verification.

The next-stage [protocol audit](PROTOCOL_AUDIT.md) verifies that local weight/config hashes match current public HF revision `a2c770b768ceb3a53c36c4f7a3619db0413dc3a1`; the original download revision and mapping to individual paper experiments remain unknown. It records corresponding-pixel RoMa registration versus the paper's ICP wording and the confidence-loss sign discrepancy without claiming either causes the metric gap. The notebook saves this audit and live, read-only pose-data preflight output.

[`scripts/check_pose_data.py`](scripts/check_pose_data.py) fails on missing prescribed scenes, malformed GT or corrupt images. The RE10K split has **1,832 unique IDs**, not 1,800. A portable single-GPU HF pose entry with fixed draws/resume is implemented and one real smoke was verified; neither is a full Table1 benchmark. After storage expansion, official camera metadata and the test-only RGB archive are present. Safe chunk CRC/RGB/GT preparation and independent byte/inventory verification completed for 1,756 scenes / 265,447 frames, but 76 prescribed IDs remain missing and are never silently dropped. The CO3D prescribed split/provenance and full RE10K data still require verification. All 148 offline tests pass; these are not paper scores.

The CO3D public-HF one-draw pose smoke is now complete and revalidated from saved poses: 10 views (9 unique), 45 pairs, no PnP failures, but RRA@30=0%, RTA@30=13.3333%, mAA@30=0%. These poor real results are retained, not full Table 1 scores. An initial portrait/landscape shape assertion was diagnosed and corrected without changing the released algorithm; v1 evidence and a separate v2 preflight are preserved. The notebook records two new executed analysis cells. Continuous 100-draw state, paper split/checkpoint equivalence and retraining are still incomplete. See [continuation checkpoint](CONTINUATION.md).

A new same-forward CO3D orientation diagnostic reproduces the saved published-branch scores exactly. Removing the transpose changes the estimated focal from 27.98px to 553.66px but yields only RRA@30=2.2222%, RTA@30=11.1111%, mAA@30=0%; it does not solve the low-score case or establish paper equivalence. Both branches and all 45 pairs are retained; one new notebook analysis cell was executed. This is not an isolated fixed-focal experiment or a replacement benchmark protocol.

The additional RE10K full-source scan has ended successfully: 205,763,619,478 compressed bytes and the pinned LFS SHA matched, without storing the full archive. Under the verified tar-path/PNG naming protocol it recognized 4,137 source scenes but recovered none of the 76 missing prescribed IDs (0 staged frames). Independent saved-report/official-GT consistency checks passed, not a second full-archive rehash. Coverage remains 1,756/1,832; this does not establish permanent unavailability or full Table 1 readiness. The scan is archived and must not be restarted.

Paired NRGBD local/global evaluation used `--head both` to share one forward pass. The 7-Scenes preparation pipeline reads official TestSplit archives via HTTP Range, checks sequence CRC/SHA256, registers depth using pinned SimpleRecon calibration, and stores only the original stride-20 selected frames. The loader preserves original frame numbering and refuses incompatible sparse-storage protocols. GPU evaluation queues behind NRGBD and free-memory checks; disk usage retains a 1GiB reserve. Eight offline preparation tests and a real 50-view heads-sequence dry-run passed. Full result JSONs must be validated before reporting scores; these pipelines do not constitute complete-paper reproduction. See [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md) and the notebook for status and remaining training/data requirements.

## Using Fast3R in Your Own Project

To use Fast3R in your own project, you can import the `Fast3R` class from `fast3r.models.fast3r` and use it as a regular PyTorch model.

```python
import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

# --- Setup ---
# Load the model from Hugging Face
model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
precision = "16-mixed" if device.type == "cuda" else "32"

# Create a lightweight lightning module wrapper for the model.
# This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

# Set model to evaluation mode
model.eval()
lit_module.eval()

# --- Load Images ---
# Provide a list of image file paths. Images can come from different cameras and aspect ratios.
filelist = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
images = load_images(filelist, size=512, verbose=True)

# --- Run Inference ---
# profiling=True uses CUDA synchronization, so disable it on CPU.
inference_result = inference(
    images,
    model,
    device,
    dtype=precision,
    verbose=True,
    profiling=(device.type == "cuda"),
)
output_dict, profiling_info = (
    inference_result if device.type == "cuda" else (inference_result, None)
)

# --- Estimate Camera Poses ---
# This step estimates the camera-to-world (c2w) poses for each view using PnP.
poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
    output_dict['preds'],
    niter_PnP=100,
    focal_length_estimation_method='first_view_from_global_head'
)
# poses_c2w_batch is a list; the first element contains the estimated poses for each view.
camera_poses = poses_c2w_batch[0]

# Print camera poses for all views.
for view_idx, pose in enumerate(camera_poses):
    print(f"Camera Pose for view {view_idx}:")
    print(pose.shape)  # np.array of shape (4, 4), the camera-to-world transformation matrix

# --- Extract 3D Point Clouds for Each View ---
# Each element in output_dict['preds'] corresponds to a view's point map.
for view_idx, pred in enumerate(output_dict['preds']):
    point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
    print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")  # shape: (1, 368, 512, 3), i.e., (1, Height, Width, XYZ)
```

## Training

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python fast3r/train.py experiment=super_long_training/super_long_training
```

You can override any parameter from command line following [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/):

```bash
python fast3r/train.py experiment=super_long_training/super_long_training trainer.max_epochs=20 trainer.num_nodes=2
```

To submit a multi-node training job with Slurm, use the following command:

```bash
python scripts/slurm/submit_train.py --nodes=<NODES> --experiment=<EXPERIMENT>
```

After training, you can run the demo with a lightning checkpoint with the following command:
```bash
python fast3r/viz/demo.py --is_lightning_checkpoint --checkpoint_dir=/path/to/super_long_training_999999
```

## Evaluation

To evaluate on 3D reconstruction or camera pose estimation tasks, run:

```bash
python fast3r/eval.py eval=<eval_config>
```
`<eval_config>` can be any of the evaluation configurations in [configs/eval/](configs/eval/). For example:
- `ablation_recon_better_inference_hp/ablation_recon_better_inference_hp` evaluates the 3D reconstruction on DTU, 7-Scenes and Neural-RGBD datasets.
- `eval_cam_pose/eval_cam_pose_10views` evaluates the camera pose estimation on 10 views on CO3D dataset.


To evaluate camera poses on RealEstate10K dataset, run:

```bash
python scripts/fast3r_re10k_pose_eval.py  --subset_file scripts/re10k_test_1800.txt
```

To evaluate multi-view depth estimation on Tanks and Temples, ETH-3D, DTU, and ScanNet datasets, follow the data download and preparation guide of [robustmvd](https://github.com/lmb-freiburg/robustmvd), install that repo's `requirements.txt` into the current conda environment, and run:

```bash
python scripts/robustmvd_eval.py
```

## Dataset Preprocessing

Please follow [DUSt3R's data preprocessing instructions](https://github.com/naver/dust3r/tree/main?tab=readme-ov-file#datasets) to prepare the data for training and evaluation. The pre-processed data is compatible with the [multi-view dataloaders](fast3r/dust3r/datasets) in this repo.

For preprocessing the DTU, 7-Scene, and NRGBD datasets for evaluation, we follow [Spann3r's data processing instructions](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md).

## FAQ

- Q: `httpcore.ConnectError: All connection attempts failed` when launching the demo?
  - See [#34](https://github.com/facebookresearch/fast3r/issues/34). Download the example videos into a local directory.
- Q: Data pre-processing for BlendedMVS, `train_list.txt` is missing?
  - See [#33](https://github.com/facebookresearch/fast3r/issues/33).
- Q: Loading checkpoint to fine-tune Fast3R?
  - See [#25](https://github.com/facebookresearch/fast3r/issues/25)
- Q: Running demo on Windows? (TypeError: cannot pickle '_thread.RLock' object)
  - See [#28](https://github.com/facebookresearch/fast3r/issues/28). It seems that some more work is needed to make the demo compatible with Windows - we hope the community could contribute a PR!
- Q: Completely messed-up point cloud output?
  - See [#21](https://github.com/facebookresearch/fast3r/issues/21). Please make sure the cuROPE module is NOT installed.
- Q: My GPU doesn't support FlashAttention / `No available kernel. Aborting execution`?
  - See [#17](https://github.com/facebookresearch/fast3r/issues/17). Use `attn_implementation=pytorch_auto` option instead.
- Q: `TypeError: Fast3R.__init__() missing 3 required positional arguments: 'encoder_args', 'decoder_args', and 'head_args'`
  - See See [#7](https://github.com/facebookresearch/fast3r/issues/7). It is caused by a networking issue with downloading the model from Huggingface in some countries (e.g., China) - please pre-download the model checkpoint with a working networking configuration, and use a local path to load the model instead.
## License

The code and models are licensed under the [FAIR NC Research License](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citation

```
@InProceedings{Yang_2025_Fast3R,
    title={Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass},
    author={Jianing Yang and Alexander Sax and Kevin J. Liang and Mikael Henaff and Hao Tang and Ang Cao and Joyce Chai and Franziska Meier and Matt Feiszli},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2025},
}
```

## Acknowledgement

Fast3R is built upon a foundation of remarkable open-source projects. We deeply appreciate the contributions of these projects and their communities, whose efforts have significantly advanced the field and made this work possible.

- [DUSt3R](https://dust3r.europe.naverlabs.com/)
- [Spann3R](https://hengyiwang.github.io/projects/spanner)
- [Viser](https://viser.studio/main/)
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=facebookresearch/fast3r&type=Date)](https://star-history.com/#facebookresearch/fast3r&Date)

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

### CO3D candidate 100-request pose evaluation completed (2026-10-03)

The verified continuous preparation and GPU evaluation completed all 100 requests / 4,500 camera pairs.
The [compact verified summary](results/co3d_pose_100_seed42_verified_summary_20261003.json) reports
RRA@15 **31.2889%**, RTA@15 **28.5778%**, and mAA@30 **23.8007%**.
Eight zero-focal requests caused 80/1,000 PnP view failures; all identity fallbacks remain included.
The actual inputs cover 99 trajectories / 38 categories / 884 unique returned RGB frames;
64 requests contain repeated views, retaining all 160 duplicate camera pairs.
An independent read-only audit recomputed every pair error, request metric, and aggregate from saved poses,
and checked the preparation input/GT hashes, bound code/weights, checkpoints and fallback accounting.
Run `PYTHONPATH=.:scripts python scripts/verify_co3d_candidate_pose_report.py` to reverify locally.

The notebook saves new executed analysis cells and a [scientific figure](results/figures/co3d_candidate100_pose.png).
Author processed-split equivalence and public-checkpoint mapping to paper experiments remain unverified,
so these are candidate adaptation scores; the full Table 1 reproduction and full-paper training/ablations
remain incomplete. Paper references, protocol differences and next steps are recorded in
[PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) and [PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md).

历史 CO3D 工程提案（2026-10-03）：候选51类/2511轨迹/498757帧与seen41差异已核验，旧提案摘要`34847924f082fd4536fe710a688cb82fda684f2692597a785289d3f77c9a7933`及JSON保留。但强制1000不同轨迹/随机10不同帧不是发布源采样器，后续不使用；raw2GiB/processed512MiB上限不是空间能装下的证明。当前源1000@采样、真实目录预算和有界实际准备见顶部；不能把提案、首请求快照或旧100请求当正式Table1。
