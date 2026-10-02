#!/usr/bin/env python3
"""Ten-frame real Range/CRC/preprocess wiring probe, NEVER Table 1 coverage.

Keep raw originals, reference outputs and adapted outputs in separate roots.
Execute only reviewed definitions from the pinned public DUSt3R preprocessor
and cropping/geometry helpers; no remote top-level code or arbitrary pickle.
"""
import ast
import fcntl
import gzip
import hashlib
import io
import json
import os
import random
import shutil
import types
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import torch

from co3d_range_cache import RawCache, MAX_MEMBER, RESERVE
from audit_co3d_camera_metadata import ROOT, MAX_JSON, annotation_mapping
from prepare_re10k_rgb_from_archive import save_identical, sha
from fast3r.dust3r.datasets.utils import cropping as local_cropping

REV = '4c24a6ebf04809f2cfe59915e51779c8984aaa40'
BASE = 'https://raw.githubusercontent.com/naver/dust3r/' + REV + '/'
WORK = Path('data/co3d_range_probe')
MAX_PIXELS = 16*1024**2


def exclusive_bytes(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes()!=data: raise ValueError('Existing probe data differs; preserve history')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path.parent).free < RESERVE+len(data): raise RuntimeError('Preserve 1GiB reserve')
    with path.open('xb') as out: out.write(data)


def reference_file(relative, name):
    path = WORK / 'references' / name
    if not path.exists():
        with urllib.request.urlopen(BASE+relative,timeout=45) as r: data=r.read(256*1024+1)
        if len(data)>256*1024: raise ValueError('Reference code exceeds bound')
        exclusive_bytes(path,data)
    return path


def definitions(path, names, namespace):
    """Only selected reviewed AST definitions; imports/top-level actions excluded."""
    tree=ast.parse(Path(path).read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in selected}!=set(names): raise ValueError('Pinned reference definitions missing')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def reference_modules():
    crop_path=reference_file('dust3r/datasets/utils/cropping.py','cropping.py')
    geometry_path=reference_file('dust3r/utils/geometry.py','geometry.py')
    # These two reviewed helpers only copy K and add/subtract half a pixel.
    helpers=definitions(geometry_path,['colmap_to_opencv_intrinsics','opencv_to_colmap_intrinsics'],{'np':np})
    ns={'PIL':PIL,'cv2':cv2,'np':np,'lanczos':PIL.Image.Resampling.LANCZOS,
        'bicubic':PIL.Image.Resampling.BICUBIC,**{n:helpers[n] for n in ['colmap_to_opencv_intrinsics','opencv_to_colmap_intrinsics']}}
    names=['ImageList','rescale_image_depthmap','camera_matrix_of_crop','crop_image_depthmap','bbox_from_intrinsics_in_out']
    definitions(crop_path,names,ns)
    return types.SimpleNamespace(**{n:ns[n] for n in names}),crop_path,geometry_path


def preprocessing_namespace(cropping):
    source=ROOT/'references/preprocess_co3d_reference.py'
    ns={'random':random,'gzip':gzip,'json':json,'os':os,'osp':os.path,'torch':torch,
        'PIL':PIL,'np':np,'cv2':cv2,'plt':plt,'tqdm':lambda x:x,'cropping':cropping}
    names=['convert_ndc_to_pinhole','opencv_from_cameras_projection','get_set_list','prepare_sequences']
    return definitions(source,names,ns)


def process_frame(raw_paths, annotation, ns, cropping):
    """Small adapter retaining all reference decode/crop/quantization choices."""
    with PIL.Image.open(raw_paths['images']) as im:
        if im.width*im.height>MAX_PIXELS: raise ValueError('RGB pixel guard')
        image=im.convert('RGB')
    with PIL.Image.open(raw_paths['depths']) as im:
        if im.width*im.height>MAX_PIXELS: raise ValueError('Depth pixel guard')
        bits=np.asarray(im,dtype=np.uint16)
    depth=bits.view(np.float16).astype(np.float32)
    with PIL.Image.open(raw_paths['masks']) as im:
        if im.width*im.height>MAX_PIXELS: raise ValueError('Mask pixel guard')
    mask=plt.imread(raw_paths['masks'])
    if (list(depth.shape)!=annotation['image']['size'] or mask.shape!=depth.shape
            or image.size!=depth.shape[::-1] or not np.isfinite(depth).all()
            or not np.isfinite(mask).all() or (depth<0).any() or (mask<0).any() or (mask>1).any()):
        raise ValueError('Raw geometry/depth/mask mismatch; no silent repair')
    if annotation['depth']['scale_adjustment']!=1.: raise ValueError('Unexpected depth scale')
    v=annotation['viewpoint']
    if v['intrinsics_format']!='ndc_isotropic': raise ValueError('Unexpected intrinsics format')
    R,t,K=ns['opencv_from_cameras_projection'](np.array(v['R']),np.array(v['T']),np.array(v['focal_length']),
                                               np.array(v['principal_point']),np.array(annotation['image']['size']))
    K=K.numpy(); H,W=depth.shape
    cx,cy=K[:2,2].round().astype(int)
    mx,my=min(cx,W-cx),min(cy,H-cy)
    if min(mx,my)<=0: raise ValueError('Invalid reference principal crop')
    bbox=(cx-mx,cy-my,cx+mx,cy+my)
    image,dm,K=cropping.crop_image_depthmap(image,np.stack((depth,mask),axis=-1),K,bbox)
    scale=(384/min(H,W))+1e-8
    target=np.floor(np.array([W,H])*scale).astype(int)
    if max(target)<512:
        scale=(512/max(H,W))+1e-8
        target=np.floor(np.array([W,H])*scale).astype(int)
    image,dm,K=cropping.rescale_image_depthmap(image,dm,K,target)
    depth,mask=dm[:,:,0],dm[:,:,1]
    maximum=np.max(depth)
    if not np.isfinite(maximum) or maximum<=0: raise ValueError('No finite positive reference depth')
    w2c=np.eye(4,dtype=np.float32);w2c[:3,:3]=R;w2c[:3,3]=t
    pose=np.linalg.inv(w2c)
    return image,(depth/maximum*65535).astype(np.uint16),(mask*255).astype(np.uint8),K,pose,maximum,{
        'raw_size_hw':[H,W],'principal_crop_bbox':[int(x) for x in bbox],
        'requested_resize_wh':target.tolist(),'processed_size_hw':[image.height,image.width],
        'raw_depth_min':float(np.min(bits.view(np.float16).astype(np.float32))),
        'raw_depth_max':float(np.max(bits.view(np.float16).astype(np.float32))),
        'raw_depth_positive_fraction':float(np.mean(bits.view(np.float16)>0)),
        'masked_processed_depth_positive_fraction':float(np.mean((depth>0)&(mask>.1)))}


def save_processed(root, annotation, values):
    image,depth,mask,K,pose,maximum,stats=values
    targets=[root/annotation[k]['path'] for k in ('image','depth','mask')]
    buf=io.BytesIO();image.save(buf,format='JPEG');exclusive_bytes(targets[0],buf.getvalue())
    for target,array in zip(targets[1:],[depth,mask]):
        ok,encoded=cv2.imencode('.png',array)
        if not ok: raise ValueError('PNG encoding failed')
        exclusive_bytes(target,encoded.tobytes())
    meta=targets[0].with_suffix('.npz')
    expected={'camera_intrinsics':K,'camera_pose':pose,'maximum_depth':maximum}
    if meta.exists():
        with np.load(meta,allow_pickle=False) as old:
            if set(old.files)!=set(expected) or any(not np.array_equal(old[k],v) for k,v in expected.items()):
                raise ValueError('Existing NPZ differs')
    else:
        meta.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(meta.parent).free<RESERVE+65536: raise RuntimeError('NPZ reserve')
        with meta.open('xb') as out: np.savez(out,**expected)
    return targets,meta


def main():
    torch.set_num_threads(2)
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=RawCache()
    category='apple'; scene=next(iter(cache.selected[category])); frames=cache.selected[category][scene][:10]
    parent=json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
    archive=ROOT/'archives'/f'{category}_000.zip'
    if sha(archive)!=parent['metadata_archives'][category]['archive_sha256']: raise ValueError('Metadata source changed')
    with zipfile.ZipFile(archive) as z:
        packed=z.read(category+'/frame_annotations.jgz')
        seq_packed=z.read(category+'/sequence_annotations.jgz')
        rows=[]
        for m in parent['metadata_archives'][category]['set_list_member_order']:
            rows.extend(json.loads(z.read(m))['test'])
    mapping=annotation_mapping(category,{scene:frames},rows)
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as g: raw=g.read(MAX_JSON+1)
    if len(raw)>MAX_JSON: raise ValueError('Annotation decode guard')
    wanted=set(mapping.values())
    annotations={}
    for a in json.loads(raw):
        key=(a['sequence_name'],a['frame_number'])
        if key in wanted:
            if key in annotations or mapping.get(a['image']['path'])!=key: raise ValueError('Annotation identity mismatch')
            annotations[key]=a
    if set(annotations)!=wanted: raise ValueError('Incomplete probe annotations')
    with gzip.GzipFile(fileobj=io.BytesIO(seq_packed)) as g: seq_raw=g.read(32*1024**2+1)
    if len(seq_raw)>32*1024**2: raise ValueError('Sequence decode guard')
    sequences=[s for s in json.loads(seq_raw) if s['sequence_name']==scene]
    if len(sequences)!=1 or sequences[0]['viewpoint_quality_score']<=.5: raise ValueError('Invalid selected sequence quality')
    records=[]
    raw_root=WORK/'reference_raw'
    crop,crop_path,geom_path=reference_modules()
    ns=preprocessing_namespace(crop)
    ordered=[annotations[mapping[f'{category}/{scene}/images/frame{f:06d}.jpg']] for f in frames]
    for a in ordered:
        paths={}; members=[]
        for kind,key in [('images','image'),('depths','depth'),('masks','mask')]:
            member=a[key]['path']; path,record=cache.fetch(member); paths[kind]=path
            target=raw_root/member;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=record['sha256']: raise ValueError('Reference raw copy differs')
            else: os.link(path,target)
            members.append({k:record[k] for k in ('sha256','member_crc_verified','full_archive_sha_verified','transport_bytes')})
            members[-1].update(path=member,archive_url=record['entry']['url'],etag=record['entry']['etag'],bytes=record['entry']['file_size'])
        values=process_frame(paths,a,ns,local_cropping)
        targets,meta=save_processed(WORK/'adapted_processed',a,values)
        records.append({'image_path':a['image']['path'],'annotation_frame_number':a['frame_number'],
                        'raw_members':members,**values[-1],
                        'adapted_processed_file_sha256':{str(p.relative_to(WORK/'adapted_processed')):sha(p) for p in targets},
                        'camera_pose_dtype':str(values[4].dtype),'maximum_depth':float(values[5])})
        print('RAW CRC/DECODE/PREPROCESS',a['image']['path'],flush=True)
    # This test-only fixture is explicitly ten ORIGINAL frames, not a benchmark selection.
    exclusive_bytes(raw_root/category/'frame_annotations.jgz',gzip.compress(json.dumps(ordered).encode(),mtime=0))
    exclusive_bytes(raw_root/category/'sequence_annotations.jgz',gzip.compress(json.dumps(sequences).encode(),mtime=0))
    exclusive_bytes(raw_root/category/'set_lists/probe_fewview_train.json',json.dumps({'test':[[scene,a['frame_number'],a['image']['path']] for a in ordered]}).encode())
    ref_root=WORK/'reference_processed'
    complete=WORK/'reference_complete.json'
    if not complete.exists():
        selected=ns['prepare_sequences'](category,str(raw_root),str(ref_root),512,'test',.5,50,42,False)
        save_identical(complete,{'frame_fixture_sha256':sha(raw_root/category/'frame_annotations.jgz'),
                                'selected':selected,'preprocessor_sha256':sha(ROOT/'references/preprocess_co3d_reference.py'),
                                'cropping_sha256':sha(crop_path),'geometry_sha256':sha(geom_path)})
    else:
        done=json.loads(complete.read_text())
        if (done['frame_fixture_sha256']!=sha(raw_root/category/'frame_annotations.jgz')
                or done['preprocessor_sha256']!=sha(ROOT/'references/preprocess_co3d_reference.py')
                or done['cropping_sha256']!=sha(crop_path) or done['geometry_sha256']!=sha(geom_path)):
            raise ValueError('Existing reference output fingerprint differs')
        selected=done['selected']
    if selected!={scene:frames}: raise ValueError('Reference changed fixture frame identities/order')
    for a,r in zip(ordered,records):
        for key in ('image','depth','mask'):
            if sha(ref_root/a[key]['path'])!=sha(WORK/'adapted_processed'/a[key]['path']):
                raise ValueError('Actual reference image/depth/mask bytes differ from adapter')
        rel=Path(a['image']['path']).with_suffix('.npz')
        with np.load(ref_root/rel,allow_pickle=False) as ref,np.load(WORK/'adapted_processed'/rel,allow_pickle=False) as adapted:
            if set(ref.files)!=set(adapted.files) or any(not np.array_equal(ref[k],adapted[k]) for k in ref.files):
                raise ValueError('Actual reference NPZ arrays differ from adapter')
        r['reference_image_depth_mask_bytes_equal']=True;r['reference_npz_arrays_equal']=True
    report={'status':'ten_real_frames_range_crc_decode_and_reference_preprocess_verified_not_benchmark_ready',
            'paper_mapping':'section 4.2 / Table 1 CO3D processed data wiring only',
            'category':category,'sequence':scene,'frame_count':len(records),'frame_numbers':frames,
            'candidate_manifest_sha256':sha(cache.manifest),'all_candidate_frame_pool_unchanged':True,
            'sampling_scope':'first ten candidate frames from first apple trajectory, explicit wiring probe only',
            'fingerprint':{'probe_sha256':sha(Path(__file__)),'cache_sha256':sha(Path('scripts/co3d_range_cache.py')),
                'preprocessor_sha256':sha(ROOT/'references/preprocess_co3d_reference.py'),
                'cropping_reference_sha256':sha(crop_path),'geometry_reference_sha256':sha(geom_path),
                'local_cropping_sha256':sha(Path('fast3r/dust3r/datasets/utils/cropping.py'))},
            'runtime':{'numpy':np.__version__,'opencv':cv2.__version__,'pillow':PIL.__version__,'torch':torch.__version__},
            'records':records,'source_zip_full_sha_verified':False,'formal_pose_metrics_available':False,
            'full_candidate_data_ready':False,'paper_selection_and_sampling_equivalence_verified':False,
            'network_bytes_first_successful_probe':cache.network_bytes,'source_zip_index_reuse_supported':True,
            'limits':{'raw_member_bytes':MAX_MEMBER,'cache_bytes':2*1024**3,'reserve_bytes':RESERVE,'decode_pixels':MAX_PIXELS},
            'note':'Member CRC + partial source identity is not full ZIP SHA. Ten frames do not validate all 399204. Reference run uses the same NumPy/PIL/OpenCV runtime; NPZ arrays compared, not ZIP timestamp bytes.'}
    result_path=Path('results/co3d_preprocess_probe_20261002.json')
    if result_path.exists():
        old=json.loads(result_path.read_text())
        report['network_bytes_first_successful_probe']=old['network_bytes_first_successful_probe']
    save_identical(result_path,report)
    print('Network bytes THIS verification process:',cache.network_bytes,flush=True)
    print('TEN-FRAME REFERENCE PREPROCESS PROBE COMPLETE',flush=True)


if __name__=='__main__': main()
