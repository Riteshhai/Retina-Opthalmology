"""Pinned EyeQ annotations and explicit original-image readiness.

EyeQ's three quality categories are retained. This external-only adapter never
turns them into DeepDRiD training labels or treats a filename audit as pixel or
patient independence. Author metadata and source photographs have separate terms.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import httpx
import numpy as np
import pandas as pd
from PIL import Image

from qsart.data.manifests import retinal_phash
from qsart.models.checkpoint import sha256_file

REVISION = '1dcfabb08cd31045de1a4a3ee82e48f6833697aa'
REPOSITORY = 'https://github.com/HzFu/EyeQ'
OFFICIAL_IMAGES = 'https://www.kaggle.com/c/diabetic-retinopathy-detection/data'
QUALITY_NAMES = {0:'Good',1:'Usable',2:'Reject'}
SOURCE_FILES = {
    'README.md': ('8d7c80347c34759058bae1580b7f17d9f6e6834446daf22cc441c7e58d1f7db4','dcf380756aca2cb7021c8ceaad49edbb7cbd18e6'),
    'data/Label_EyeQ_train.csv': ('34f39f8923f62a13fcaeccef7e5da7ec607b45805e7736042fa274b2731b9480','018cd040ce310fd93945a97405195e443807068b'),
    'data/Label_EyeQ_test.csv': ('209135bb0dff2d6203c445ee010cbdcb925edb5338a040dc6ae62c06649047b3','4e638e2fe1de11339d49005b2507172d66b6d45b'),
    'EyeQ_preprocess/EyeQ_process_main.py': ('27b52cc83044e47b28b304afb5fccd3b05166391713b0fc29a60ee71dba02b4f','4361e013df4325439a4893cc64c48b02ce2194e6'),
    'EyeQ_preprocess/fundus_prep.py': ('9681709449cf2ee57827683fa0c8884b72d607faaa8f7a7e659a2ab99154c6cf','a1dbc3e2d3136ce0f2f6f1cb6b23df512274808e'),
    'MCF_Net/dataloader/EyeQ_loader.py': ('c7049c6aedf7ed342725f7dc4d4e703c60c87b5f9654a391cd84e25f65bd372c','3d06835dee343784998d2429bcd2fab5f9dafa9c'),
    'MCF_Net/Main_EyeQuality.py': ('e5e1938bc8f168e8853841824edab90ede8579f0435d43404e53b9ffa4937f78','451f92799ac61220a061068275d42040c55a46b5'),
}
FILENAME = re.compile(r'(?P<subject>[1-9][0-9]*)_(?P<eye>left|right)\.jpeg\Z')
SCHEMA = ['Unnamed: 0','image','quality','DR_grade']
BINARY_VIEWS = {
    'reject_vs_nonreject': {'poor_codes':[2],'negative_codes':[0,1]},
    'usable_or_reject_vs_good': {'poor_codes':[1,2],'negative_codes':[0]},
}


def source_url(name):
    if name not in SOURCE_FILES:
        raise ValueError('Only the pinned author metadata/code files may be fetched')
    return f'https://raw.githubusercontent.com/HzFu/EyeQ/{REVISION}/{name}'


def verify_source_files(folder):
    folder = Path(folder)
    records = {}
    for name,(expected_sha,expected_blob) in SOURCE_FILES.items():
        path = folder/name
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if actual != expected_sha or blob != expected_blob:
            raise ValueError('Pinned EyeQ source bytes differ: '+name)
        records[name] = {'sha256':actual,'git_blob_sha1':blob,'bytes':len(data),'source_url':source_url(name)}
    return records


def fetch_source_files(folder):
    """Fetch a closed list of small public metadata files, never images/mirrors."""
    folder = Path(folder)
    with httpx.Client(timeout=45,follow_redirects=False) as client:
        for name,(expected_sha,expected_blob) in SOURCE_FILES.items():
            path = folder/name
            if path.exists():
                if sha256_file(path) != expected_sha:
                    raise ValueError('Preserve changed local source; do not overwrite: '+name)
                continue
            response = client.get(source_url(name))
            response.raise_for_status()
            data = response.content
            if len(data)>1_000_000 or hashlib.sha256(data).hexdigest()!=expected_sha:
                raise ValueError('Author source size/hash mismatch')
            if hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=expected_blob:
                raise ValueError('Author Git blob mismatch')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(data)
    return verify_source_files(folder)


def read_annotations(path, official_split, *, data_purpose='real_metadata'):
    if official_split not in {'train','test'} or data_purpose not in {'real_metadata','software_test'}:
        raise ValueError('Explicit author train/test split and supported data purpose required')
    path = Path(path)
    if data_purpose=='real_metadata' and sha256_file(path)!=SOURCE_FILES[f'data/Label_EyeQ_{official_split}.csv'][0]:
        raise ValueError('Require the exact pinned author annotation file for its original split')
    frame = pd.read_csv(path,dtype=str,keep_default_na=False)
    if list(frame.columns)!=SCHEMA or frame.empty:
        raise ValueError('Exact EyeQ CSV schema and nonempty source rows required')
    if frame.eq('').any().any() or frame['Unnamed: 0'].tolist()!=[str(i) for i in range(len(frame))]:
        raise ValueError('Complete labels and unchanged serialized source-row index required')
    if not frame.quality.str.fullmatch('[0-2]').all() or not frame.DR_grade.str.fullmatch('[0-4]').all():
        raise ValueError('EyeQ quality codes0..2 and independent source DR codes0..4 required')
    parsed = [FILENAME.fullmatch(name) for name in frame.image]
    if any(value is None for value in parsed):
        raise ValueError('Exact original EyePACS subject_eye.jpeg identifier required; no paths or processed aliases')
    if frame.image.duplicated().any():
        raise ValueError('Duplicate image identifier within author split')
    result = pd.DataFrame({
        'dataset_source':'EyeQ_EyePACS','source_official_split':official_split,
        'source_row':np.arange(len(frame)), 'original_image_name':frame.image,
        'source_subject_key':[value.group('subject') for value in parsed],
        'eye':[value.group('eye') for value in parsed],
        'eyeq_quality_code':frame.quality.astype(int),'source_dr_grade':frame.DR_grade.astype(int),
        'source_annotation_sha256':sha256_file(path), 'source_revision':REVISION,
        'split':'external_holdout','eligible_for_training':False,
        'annotation_status':'pinned_author_metadata_only' if data_purpose=='real_metadata' else 'software_fixture',
        'data_purpose':data_purpose})
    result['split_group_id'] = 'eyepacs_'+result.source_subject_key
    result['eyeq_quality_name'] = result.eyeq_quality_code.map(QUALITY_NAMES)
    return result


def load_annotations(folder, *, data_purpose='real_metadata'):
    parts = [read_annotations(Path(folder)/f'data/Label_EyeQ_{name}.csv',name,data_purpose=data_purpose) for name in ('train','test')]
    result = pd.concat(parts,ignore_index=True)
    if result.original_image_name.duplicated().any():
        raise ValueError('Image identifiers duplicate across original source splits')
    return result


def binary_view(frame, *, endpoint):
    """Explicit sensitivity view; no default harmonization or generic training label."""
    if endpoint not in BINARY_VIEWS or not frame.eyeq_quality_code.isin(QUALITY_NAMES).all():
        raise ValueError('Choose a named EyeQ binary sensitivity view explicitly')
    return frame.eyeq_quality_code.isin(BINARY_VIEWS[endpoint]['poor_codes'])


def annotation_profile(frame):
    summaries = {}
    for split,part in frame.groupby('source_official_split',sort=True):
        sizes = part.groupby('source_subject_key').size()
        paired = part.loc[part.source_subject_key.isin(sizes[sizes.eq(2)].index)]
        summaries[split] = {'images':len(part),'source_subject_keys':int(part.source_subject_key.nunique()),
            'views_per_subject_key':{str(k):int(v) for k,v in sizes.value_counts().sort_index().items()},
            'eye_counts':{name:int(part.eye.eq(name).sum()) for name in ('left','right')},
            'quality_counts':{str(k):int(part.eyeq_quality_code.eq(k).sum()) for k in range(3)},
            'dr_counts':{str(k):int(part.source_dr_grade.eq(k).sum()) for k in range(5)},
            'quality_by_dr':[[int((part.eyeq_quality_code.eq(q)&part.source_dr_grade.eq(g)).sum()) for g in range(5)] for q in range(3)],
            'quality_discordant_eye_pairs':int(paired.groupby('source_subject_key').eyeq_quality_code.nunique().gt(1).sum()),
            'dr_discordant_eye_pairs':int(paired.groupby('source_subject_key').source_dr_grade.nunique().gt(1).sum()),
            'binary_sensitivity_counts':{name:{'poor_positive':int(binary_view(part,endpoint=name).sum()),
                'negative':int((~binary_view(part,endpoint=name)).sum())} for name in BINARY_VIEWS}}
    train = set(frame.loc[frame.source_official_split.eq('train'),'source_subject_key'])
    test = set(frame.loc[frame.source_official_split.eq('test'),'source_subject_key'])
    return {'images_annotated':len(frame),'source_subject_keys':int(frame.source_subject_key.nunique()),
        'original_splits':summaries,'cross_split_subject_keys':len(train&test),
        'duplicate_image_identifiers':int(frame.original_image_name.duplicated().sum()),
        'source_grouping':'EyePACS filename subject key; preserve both eyes',
        'pixel_overlap_or_independence_verified':False,
        'quality_code_names':{str(k):v for k,v in QUALITY_NAMES.items()},
        'binary_harmonization_selected':False,'generic_training_quality_label_created':False}


def resolve_original(root, original_name, official_split):
    match = FILENAME.fullmatch(original_name)
    if match is None or official_split not in {'train','test'}:
        raise ValueError('Exact original image name and split required')
    root = Path(root).resolve()
    found = []
    for candidate in (root/original_name,root/official_split/original_name):
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('Image path escapes the declared original-image root')
        if resolved.is_file():found.append(resolved)
    if len(found)>1:
        raise ValueError('Ambiguous original image locations; never silently choose a copy')
    return found[0] if found else None


def inspect_originals(frame, root):
    """Private inventory only; no predictions, transforms, source confirmation or tuning."""
    root = Path(root).resolve()
    if not root.is_dir():raise ValueError('Existing original-image root required')
    records = []
    for row in frame.itertuples():
        path = resolve_original(root,row.original_image_name,row.source_official_split)
        record = {'original_image_name':row.original_image_name,'image_path':str(path) if path else None,
                  'source_official_split':row.source_official_split,'split_group_id':row.split_group_id,
                  'status':'missing','sha256':None,'phash':None,'width':None,'height':None,
                  'source_provenance':'unconfirmed','eligible_for_training':False,'eligible_for_model_evaluation':False}
        if path is not None:
            before = sha256_file(path)
            record['sha256'] = before
            try:
                with Image.open(path) as image:image.verify()
                with Image.open(path) as image:
                    image.load()
                    if image.format!='JPEG':raise ValueError('Original .jpeg name must contain an actual JPEG')
                    record.update(status='readable_original_candidate',sha256=before,width=image.width,height=image.height,
                                  phash=retinal_phash(image.convert('RGB')))
            except (OSError,ValueError,Image.DecompressionBombError) as exc:
                record['status']='invalid:'+type(exc).__name__
            if sha256_file(path)!=before:raise ValueError('Source bytes changed during read-only inventory')
        records.append(record)
    result = pd.DataFrame(records)
    good = result.loc[result.status.eq('readable_original_candidate')]
    summary = {'images_expected':len(frame),'images_readable':len(good),'images_missing':int(result.status.eq('missing').sum()),
        'images_invalid':int(result.status.str.startswith('invalid:').sum()),
        'exact_duplicate_groups':int(good.groupby('sha256').size().gt(1).sum()),
        'source_download_confirmed':False,'cross_source_pixel_overlap_checked':False,
        'label_harmonization_protocol_locked':False,'ready_for_model_evaluation':False,'model_inference_performed':False}
    return result,summary


def source_receipt(folder, records):
    path = Path(folder)/'source_receipt.json'
    expected = {'repository':REPOSITORY,'revision':REVISION,'files':records,'photographs_downloaded':False}
    if path.exists():
        saved = json.loads(path.read_text())
        if any(saved.get(k)!=v for k,v in expected.items()):raise ValueError('Preserve differing source receipt')
        return saved
    value = {**expected,'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    return value
