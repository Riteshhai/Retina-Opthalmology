"""Plan or cache the pinned user-selected EyePACS mirror, then audit label joins.

Run explicitly before an experiment. API/dashboard startup never downloads data.
The cache is a transformed community release, not an original-EyeQ validation claim.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx

from qsart.data.hf_eyepacs import (CARD_SHA,REPOSITORY,REVISION,annotation_join,download_record,
                                  open_response,read_metadata,sha256,validate_manifest)


def preserve_json(path,value):
    content=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('Preserve existing audit/receipt; choose a new output directory')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(content)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true');parser.add_argument('--audit-cache',action='store_true')
    parser.add_argument('--cache',type=Path,default=Path('data/external/eyepacs_hf_v1'))
    parser.add_argument('--audit-output',type=Path,default=Path('artifacts/eyeq_hf_v1/cache_audit'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    cache=(root/args.cache).resolve();audit=(root/args.audit_output).resolve()
    if not cache.is_relative_to(root/'data') or not audit.is_relative_to(root/'artifacts'):
        raise ValueError('Keep the cache under ignored project data and audits under artifacts')
    manifest=validate_manifest(json.loads((root/'configs/eyepacs_hf_v1.json').read_text(encoding='utf-8')))
    plan={'repository':REPOSITORY,'revision':REVISION,'shards':14,'bytes':manifest['total_bytes'],
        'free_bytes':shutil.disk_usage(root).free,'download_requested':args.download,
        'mirror_is_original_eyeq':False,'startup_downloads_enabled':False}
    print(json.dumps({'plan':plan}),flush=True)
    if args.download:
        required=sum(r['bytes'] for r in manifest['files'] if not (cache/r['path']).exists())
        if plan['free_bytes']<required+2_000_000_000:raise ValueError('Insufficient cache space plus two-GB reserve')
        cache.mkdir(parents=True,exist_ok=True)
        with httpx.Client(timeout=httpx.Timeout(60,read=60),follow_redirects=False) as client:
            card=cache/'dataset_card.md'
            if not card.exists():
                response=open_response(client,f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/README.md')
                try:content=response.read()
                finally:response.close()
                import hashlib
                if hashlib.sha256(content).hexdigest()!=CARD_SHA:raise ValueError('Pinned dataset card differs')
                with card.open('xb') as stream:stream.write(content)
            if sha256(card)!=CARD_SHA:raise ValueError('Preserve changed cached card')
            def fetch(record):
                value=download_record(record,cache,client=client,progress=lambda item:print(json.dumps({'progress':item}),flush=True))
                print(json.dumps({'file_verified':value['path'],'cached':value['already_cached']}),flush=True)
                return value
            with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,manifest['files']))
        preserve_json(cache/'download_receipt.json',{'repository':REPOSITORY,'revision':REVISION,
            'card_sha256':CARD_SHA,'files':[{k:r[k] for k in ('path','sha256','bytes')} for r in records],
            'source_selected_by_user':True,'original_photo_source_verified':False,'model_inference_performed':False})
    if args.download or args.audit_cache:
        frame=read_metadata(cache,manifest);joined,summary=annotation_join(frame,root/'artifacts/eyeq_data_v1/source')
        preserve_json(audit/'summary.json',{'repository':REPOSITORY,'revision':REVISION,**summary})
        content=joined.to_csv(index=False,lineterminator='\n').encode();target=audit/'private_annotation_join.csv'
        if target.exists():
            if target.read_bytes()!=content:raise ValueError('Preserve existing private label join')
        else:
            with target.open('xb') as stream:stream.write(content)
        print(json.dumps({'cache_audit':summary}),flush=True)


if __name__=='__main__':main()
