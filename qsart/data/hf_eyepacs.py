"""Pinned user-selected EyePACS mirror cache; no automatic training/validation.

The mirror contains cropped/resized photographs and DR grades, not new EyeQ
quality annotations. Original-name joins retain that transformation boundary.
Downloads are immutable, resumable and SHA256-verified; source code is never run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urljoin,urlsplit

import httpx
import pandas as pd

from qsart.data.eyeq import FILENAME, load_annotations

REPOSITORY='bumbledeep/eyepacs'
REVISION='3e94599d7e41d9992a8362cc0467eb397e11210f'
CARD_SHA='b92ffa7c33b1bc9faee655bf74876eabc501a4623697ef9283bde383bd63e3a0'
LABELS={0:'no_diabetic_retinopathy',1:'mild_retinopathy',2:'moderate_retinopathy',3:'severe_retinopathy',4:'proliferative_retinopathy'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_manifest(manifest):
    if (manifest.get('schema')!='qsart.hf_eyepacs_download.v1' or manifest.get('repository')!=REPOSITORY
        or manifest.get('revision')!=REVISION or manifest.get('card_sha256')!=CARD_SHA
        or manifest.get('expected_rows')!=35108 or manifest.get('total_bytes')!=6535643877):
        raise ValueError('Require the checked immutable user-selected mirror manifest')
    files=manifest.get('files',[])
    if len(files)!=14 or {r['path'] for r in files}!={f'data/train-{i:05}-of-00014.parquet' for i in range(14)}:
        raise ValueError('Exactly fourteen pinned Parquet shards required')
    for r in files:
        if type(r.get('bytes')) is not int or r['bytes']<1 or not re.fullmatch('[0-9a-f]{64}',r.get('sha256','')):
            raise ValueError('Positive exact file sizes and SHA256 values required')
    if sum(r['bytes'] for r in files)!=manifest['total_bytes']:raise ValueError('Pinned total byte count differs')
    return manifest


def checked_url(url):
    parsed=urlsplit(url);host=parsed.hostname or ''
    if (parsed.scheme!='https' or parsed.username is not None or parsed.password is not None
        or parsed.port not in (None,443) or not (host=='huggingface.co' or host.endswith('.huggingface.co') or host.endswith('.hf.co'))):
        raise ValueError('Dataset traffic requires public Hugging Face HTTPS/CDN endpoints')
    return url


def open_response(client,url,headers=None):
    for _ in range(7):
        checked_url(url)
        response=client.send(client.build_request('GET',url,headers=headers),stream=True,follow_redirects=False)
        if response.status_code in (301,302,303,307,308):
            location=response.headers.get('location');response.close()
            if not location:raise ValueError('Download redirect omitted location')
            url=urljoin(url,location);continue
        if response.status_code not in (200,206):
            code=response.status_code;response.close()
            raise ValueError(f'Pinned dataset endpoint returned HTTP {code}; no source or access substitution')
        return response
    raise ValueError('Excessive dataset redirects')


def cache_path(root,name):
    root=Path(root).resolve()
    if not re.fullmatch(r'data/train-[0-9]{5}-of-00014\.parquet',name):raise ValueError('Only pinned shard paths accepted')
    result=(root/name).resolve()
    if not result.is_relative_to(root):raise ValueError('Cache file escapes the selected root')
    return result


def download_record(record,root,*,client,progress=None,retries=12,chunk_bytes=32*1024*1024):
    """Resume only an identity-bound partial; a final file is never overwritten."""
    if type(retries) is not int or retries<1 or type(chunk_bytes) is not int or chunk_bytes<1:
        raise ValueError('Positive bounded chunk/retry settings required')
    path=cache_path(root,record['path']);expected=record['sha256'];size=record['bytes']
    if path.exists():
        if path.stat().st_size!=size or sha256(path)!=expected:raise ValueError('Existing cached file differs; preserve it')
        return {'path':record['path'],'sha256':expected,'bytes':size,'already_cached':True}
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.part');marker=partial.with_suffix(partial.suffix+'.json')
    identity={'repository':REPOSITORY,'revision':REVISION,**record}
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8'))!=identity:raise ValueError('Partial cache belongs to different source bytes')
    elif partial.exists():raise ValueError('Unbound partial file is preserved; use a new cache directory')
    else:
        with marker.open('x',encoding='utf-8') as stream:stream.write(json.dumps(identity,sort_keys=True)+'\n')
    url=f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{record["path"]}'
    failures=0;last_notice=partial.stat().st_size if partial.exists() else 0
    while True:
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>size:raise ValueError('Partial cache exceeds pinned file size')
        if offset==size:break
        end=min(offset+chunk_bytes,size)-1
        headers={'Accept-Encoding':'identity','Range':f'bytes={offset}-{end}'}
        response=None
        try:
            response=open_response(client,url,headers)
            if response.status_code==206:
                match=re.fullmatch(r'bytes ([0-9]+)-([0-9]+)/([0-9]+)',response.headers.get('content-range',''))
                if not match or int(match[1])!=offset or int(match[3])!=size or int(match[2])!=end:
                    raise ValueError('Server did not honor the exact resume range; partial bytes preserved')
            elif response.status_code==200 and offset==0:
                end=size-1  # A full initial response is safe if a server ignores Range.
            else:raise ValueError('Server did not honor the exact resume range; partial bytes preserved')
            if response.headers.get('content-length') and int(response.headers['content-length'])!=end-offset+1:
                raise ValueError('Response size differs from pinned file/range')
            downloaded=offset
            with partial.open('ab') as stream:
                for block in response.iter_raw(1024*1024):
                    if downloaded+len(block)>end+1:raise ValueError('Response exceeds pinned byte budget')
                    stream.write(block);downloaded+=len(block)
                    if progress and downloaded-last_notice>=64*1024*1024:
                        progress({'file':record['path'],'downloaded_bytes':downloaded,'expected_bytes':size});last_notice=downloaded
            if downloaded!=end+1:raise httpx.RemoteProtocolError('Incomplete chunk preserved for resume')
            failures=0
        except (httpx.HTTPError,OSError):
            failures+=1
            if failures>=retries:raise
        finally:
            if response is not None:response.close()
    if not partial.exists() or partial.stat().st_size!=size or sha256(partial)!=expected:
        raise ValueError('Downloaded checksum/size mismatch; partial cache preserved, never used')
    # Hard-link creation fails if another process created the final path; no overwrite.
    os.link(partial,path);partial.unlink()
    return {'path':record['path'],'sha256':expected,'bytes':size,'already_cached':False}


def read_metadata(cache,manifest):
    """Project names/DR columns only; never decode embedded photographs."""
    import pyarrow.parquet as pq
    rows=[]
    for record in manifest['files']:
        path=cache_path(cache,record['path'])
        if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise ValueError('Verify every pinned shard before reading metadata')
        file=pq.ParquetFile(path)
        for batch in file.iter_batches(columns=['image.path','label_code','label'],batch_size=1024):
            for row in batch.to_pylist():
                name=row['image']['path'];grade=row['label_code']
                if not isinstance(name,str) or FILENAME.fullmatch(name) is None or type(grade) is not int or LABELS.get(grade)!=row['label']:
                    raise ValueError('Original-name/DR schema differs; no guessed identity or quality labels')
                rows.append({'original_image_name':name,'mirror_dr_grade':grade,'parquet_shard':record['path']})
    frame=pd.DataFrame(rows)
    if len(frame)!=manifest['expected_rows'] or frame.original_image_name.duplicated().any():
        raise ValueError('Mirror row cardinality or image uniqueness differs')
    return frame


def annotation_join(frame,source_folder):
    source=load_annotations(source_folder)
    merged=source.merge(frame,on='original_image_name',how='left',validate='one_to_one',indicator=True)
    matched=merged['_merge'].eq('both')
    if not merged.loc[matched,'mirror_dr_grade'].eq(merged.loc[matched,'source_dr_grade']).all():
        raise ValueError('Mirror DR grades conflict with pinned EyeQ annotations')
    summary={'mirror_rows':len(frame),'eyeq_annotation_rows':len(source),'matched_eyeq_rows':int(matched.sum()),
        'missing_eyeq_rows':int((~matched).sum()),'dr_label_conflicts':0,
        'matched_quality_counts':{str(k):int((matched&merged.eyeq_quality_code.eq(k)).sum()) for k in range(3)},
        'missing_quality_counts':{str(k):int((~matched&merged.eyeq_quality_code.eq(k)).sum()) for k in range(3)},
        'image_representation':'user_selected_community_cropped_resized_mirror',
        'images_decoded':0,'model_inference_performed':False,'original_image_bytes_verified':False,
        'source_terms_independently_verified':False,'quality_annotations_revalidated_after_transform':False,
        'ready_for_model_evaluation':False,'eligible_for_training':False}
    return merged,summary
