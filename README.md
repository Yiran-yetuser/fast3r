<div align="center">

# ⚡️Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass


${{\color{Red}\Huge{\textsf{  CVPR\ 2025\ \}}}}\$


[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2501.13928)
[![Project Website](https://img.shields.io/badge/Fast3R-Website-4CAF50?logo=googlechrome&logoColor=white)](https://fast3r-3d.github.io/)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=Gradio&logoColor=red)](https://fast3r.ngrok.app/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512/)
</div>

![Teaser Image](assets/teaser.png)

Official implementation of **Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass**, CVPR 2025

*[Jianing Yang](https://jedyang.com/), [Alexander Sax](https://alexsax.github.io/), [Kevin J. Liang](https://kevinjliang.github.io/), [Mikael Henaff](https://www.mikaelhenaff.net/), [Hao Tang](https://tanghaotommy.github.io/), [Ang Cao](https://caoang327.github.io/), [Joyce Chai](https://web.eecs.umich.edu/~chaijy/), [Franziska Meier](https://fmeier.github.io/), [Matt Feiszli](https://www.linkedin.com/in/matt-feiszli-76b34b/)*

## Installation

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

### Next paper experiments (running, not completed)

Paired NRGBD local/global evaluation now uses `--head both` to share one forward pass. The 7-Scenes preparation pipeline reads official TestSplit archives via HTTP Range, checks sequence CRC/SHA256, registers depth using pinned SimpleRecon calibration, and stores only the original stride-20 selected frames. The loader preserves original frame numbering and refuses incompatible sparse-storage protocols. GPU evaluation queues behind NRGBD and free-memory checks; disk usage retains a 1GiB reserve. Eight offline preparation tests and a real 50-view heads-sequence dry-run passed. Full result JSONs must be validated before reporting scores; these pipelines do not constitute complete-paper reproduction. See [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md) and the notebook for status and remaining training/data requirements.

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
