# CO3D preprocessing attribution

The `process_frame` preprocessing adapter in `scripts/probe_co3d_preprocess.py`
and `process_frame_allow_zero` in `scripts/co3d_lazy_dataset.py`
is derived from Naver's DUSt3R CO3D preprocessing implementation:

- Copyright (C) 2024-present Naver Corporation. All rights reserved.
- Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
- [Pinned source](https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/datasets_preprocess/preprocess_co3d.py)
- [License](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The derived preprocessing code retains those terms. The adapter adds bounded
image checks, fail-closed validation and separate saved-byte/array comparison;
it does not claim to be the author's original benchmark selection. The lazy
adapter preserves zero maximum depth as zero quantized depth without division
warnings, so the released loader can explicitly invalidate zero-depth views.

The strict lazy loader uses the pinned local Fast3R loader method, removing
only its exception-swallowing handler through a guarded AST adaptation.
The source method's original Meta copyright and repository license remain
applicable; the original source file is not changed.

Pinned cropping and geometry definitions are downloaded locally for reference
execution with their original copyright/license notices retained. They and
CO3D dataset files are not committed. This notice is scoped to the derived
preprocessing code and reference material, not a blanket change to the original
Fast3R repository license. Dataset use remains subject to its separate license.
