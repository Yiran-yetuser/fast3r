"""Offline replay of new request4 failure; preserves v1/v2/v3 committed state."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import co3d_lazy_dataset_v3 as m
from co3d_sampling_state import restore
from co3d_request_journal import read_prefix
from prepare_co3d_continuous_v3 import ROOT, CANDIDATE, validate_result
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from prepare_re10k_rgb_from_archive import sha, save_identical

OUTPUT = Path('results/co3d_v3_request4_failure_20261002.json')


def diagnose():
    p = m.LazyPreparer(); ds = m.StrictLazyCo3d(json.loads(CANDIDATE.read_text()), p)
    wrapper = ResizedDataset(100, ds); wrapper.set_epoch(0)
    initial = json.loads((ROOT/'initial.json').read_text())
    rows, state = read_prefix(ROOT, initial, validate_result)
    if len(rows) != 3: raise ValueError('Failure boundary changed')
    restore(ds, [int(x) for x in wrapper._idxs_mapping], state, initial['identity']['bindings'])
    ds.active_request = 3
    original = m.process_frame_allow_zero; observed = {}
    def inspect_failure(paths, annotation, ns):
        try: return original(paths, annotation, ns)
        except ValueError:
            depth = np.asarray(Image.open(paths['depths']), np.uint16).view(np.float16).astype(np.float32)
            mask = plt.imread(paths['masks'])
            with Image.open(paths['images']) as image: size = list(image.size)
            observed.update(image_path=annotation['image']['path'], raw_depth_shape=list(depth.shape),
                annotation_size_hw=annotation['image']['size'], image_size_wh=size,
                mask_shape=list(mask.shape), mask_finite=bool(np.isfinite(mask).all()),
                mask_min=float(mask.min()), mask_max=float(mask.max()),
                depth_nan_count=int(np.isnan(depth).sum()),
                depth_positive_inf_count=int(np.isposinf(depth).sum()),
                depth_negative_count=int((depth<0).sum()),
                depth_finite_max=float(depth[np.isfinite(depth)].max()) if np.isfinite(depth).any() else None,
                negative_positions_bits_values=[{'position_yx':[int(y),int(x)],
                    'uint16_bits':int(np.asarray(Image.open(paths['depths']),np.uint16)[y,x]),
                    'value':float(depth[y,x]) if np.isfinite(depth[y,x]) else 'negative_infinity'}
                    for y,x in np.argwhere(depth<0)[:30]],
                scale_adjustment=annotation['depth']['scale_adjustment'],
                raw_members={k: {'sha256':sha(path),'bytes':path.stat().st_size,
                    'cached_crc_verified':json.loads(Path(str(path)+'.json').read_text())['member_crc_verified']}
                    for k,path in paths.items()})
            raise
    def no_network(*args, **kwargs): raise AssertionError('Diagnostic attempted network')
    with patch.object(m, 'process_frame_allow_zero', inspect_failure), patch('co3d_range_cache.HttpRangeFile', no_network), patch('co3d_range_cache.RecordedRanges', no_network):
        try: wrapper[3]
        except ValueError as error:
            error_type, error_text = type(error).__name__, str(error)
        else: raise ValueError('Expected real failure did not recur')
    if not observed: raise ValueError('Failure not in preprocessing')
    report = {'status':'new_v3_request4_failure_reproduced_offline_service_stopped',
        'paper_mapping':'section 4.2 / Table 1 input preparation only',
        'completed_request_count':3,'request_zero_based':3,
        'base_index':int(wrapper._idxs_mapping[3]),'failure':observed,
        'load_trace':ds.trace,'pool_attempts':ds.pool_attempts,
        'error_type':error_type,'error_text':error_text,
        'source_sha256':{'diagnostic':sha(__file__),'v3_preparer':sha(m.__file__)},
        'network_bytes':p.cache.network_bytes,'model_forward_count':0,
        'old_journals_preserved':True,'failed_request_committed':False,
        'formal_pose_metrics_available':False,'full_paper_completed':False}
    save_identical(OUTPUT,report);return report


if __name__ == '__main__': print(json.dumps(diagnose(),indent=2,allow_nan=False))
