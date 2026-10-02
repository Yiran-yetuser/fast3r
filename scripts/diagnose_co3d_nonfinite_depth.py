"""Read-only failure diagnosis; never changes preprocess or sampler behavior."""
import inspect
import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import co3d_lazy_dataset as m
from fast3r_hf_re10k_pose_eval import sha,save_new

OUTPUT=Path('results/co3d_nonfinite_depth_failure_20261002.json')
MEMBER='stopsign/599_92267_182752/images/frame000081.jpg'


def reference_quantize(depth):
    maximum=np.max(depth)
    with warnings.catch_warnings(),np.errstate(invalid='ignore',divide='ignore'):
        warnings.simplefilter('ignore',RuntimeWarning)
        quantized=(depth/maximum*65535).astype(np.uint16)
    return quantized,maximum


def diagnose():
    p=m.LazyPreparer();annotation=p.annotations('stopsign')[MEMBER]
    root=Path('data/co3d_range_cache/raw/stopsign/599_92267_182752')
    paths={'images':root/'images/frame000081.jpg',
        'depths':root/'depths/frame000081.jpg.geometric.png','masks':root/'masks/frame000081.png'}
    records={}
    for kind,path in paths.items():
        r=json.loads(Path(str(path)+'.json').read_text())
        if sha(path)!=r['sha256'] or path.stat().st_size!=r['entry']['file_size'] or not r['member_crc_verified']:
            raise ValueError('Cached raw member identity changed')
        records[kind]={'sha256':r['sha256'],'bytes':path.stat().st_size,'member_crc_verified':True}
    depth=np.asarray(Image.open(paths['depths']),np.uint16).view(np.float16).astype(np.float32)
    mask=plt.imread(paths['masks'])
    # Local pinned source only. Remove finite rejection in this in-memory
    # diagnostic to observe the released arithmetic; never modify live loader.
    source=inspect.getsource(m.process_frame_allow_zero)
    source=source.replace('or not np.isfinite(depth).all()','')
    source=source.replace("if not np.isfinite(maximum) or maximum < 0: raise ValueError('Invalid max depth')",
                          "if np.isnan(maximum) or maximum < 0: raise ValueError('NaN or negative maximum')")
    namespace=dict(vars(m));exec(compile(source,'local_readonly_diagnostic','exec'),namespace)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        values=namespace['process_frame_allow_zero'](paths,annotation,p.ns)
    quantized=values[1];maximum=values[5]
    effective=quantized.astype(np.float32)/65535*np.nan_to_num(maximum)
    initial=json.loads(Path('results/co3d_continuous_prepare_20261002/initial.json').read_text())
    return {'status':'real_request4_nonfinite_depth_diagnosed_service_stopped',
        'request_zero_based':3,'base_index':initial['identity']['mapping'][3],'member':MEMBER,
        'source_sha256':{'diagnostic':sha(__file__),'lazy_loader':sha(m.__file__),
            'reference_preprocessor':p.fingerprint['preprocessor_sha256']},
        'raw_member_identity':records,'raw_depth_size_hw':list(depth.shape),
        'annotation_size_hw':annotation['image']['size'],'scale_adjustment':annotation['depth']['scale_adjustment'],
        'mask_shape':list(mask.shape),'mask_finite':bool(np.isfinite(mask).all()),
        'raw_depth_positive_inf_count':int(np.isposinf(depth).sum()),'raw_depth_nan_count':int(np.isnan(depth).sum()),
        'raw_nonfinite_positions':np.argwhere(~np.isfinite(depth)).tolist(),
        'after_reference_crop_resize_maximum':'positive_infinity' if np.isposinf(maximum) else float(maximum),
        'reference_uint16_shape':list(quantized.shape),'reference_uint16_all_zero':bool((quantized==0).all()),
        'reference_loader_effective_depth_finite':bool(np.isfinite(effective).all()),
        'reference_loader_effective_depth_all_zero':bool((effective==0).all()),
        'numpy_version':np.__version__,'completed_request_count':3,
        'network_bytes':0,'live_preprocessing_changed':False,'retry_started':False,
        'formal_pose_metrics_available':False,'full_paper_completed':False,
        'next_step':'Versioned preprocessing must preserve reference uint16/NPZ semantics and explicitly trace zero-depth retry. Validate original-loader equality and prior committed inputs before resuming; do not replace Inf with zero silently.'}


if __name__=='__main__':
    result=diagnose();json.dumps(result,allow_nan=False);save_new(OUTPUT,result)
    print(json.dumps(result,indent=2))
