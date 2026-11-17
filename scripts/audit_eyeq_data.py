"""Audit pinned public EyeQ annotations/access; optionally inventory local JPEGs privately.

No model prediction, temperature fit, image download, terms acceptance or training
occurs. Default validates source metadata only; --execute publishes aggregates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qsart.data.eyeq import (REVISION, annotation_profile, fetch_source_files,
    inspect_originals, load_annotations, source_receipt, verify_source_files)
from qsart.models.checkpoint import sha256_file
from qsart.training.quality_adaptation import canonical_hash, write_json
from scripts.run_quality_calibration_stress import candidates, preserve

STUDY='eyeq_data_v1'
CONFIG='configs/eyeq_data_v1.json'
PROTOCOL='docs/research/EYEQ_DATA_V1_PROTOCOL.md'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def ensure_aggregate(report):
    """Allow public provenance HTTPS links while rejecting identifiers/local paths."""
    json.dumps(report,allow_nan=False)
    hosts={'github.com','raw.githubusercontent.com','api.github.com','www.kaggle.com','kaggle.com','arxiv.org','creativecommons.org'}
    def check(value):
        if isinstance(value,dict):
            if {'image_id','image_path','subject_id','patient_id','original_image_name','source_subject_key'}&value.keys():
                raise ValueError('Public audit contains private identities')
            for item in value.values():check(item)
        elif isinstance(value,list):
            for item in value:check(item)
        elif isinstance(value,str):
            if '://' in value:
                parsed=urlsplit(value)
                if parsed.scheme!='https' or parsed.hostname not in hosts or parsed.username is not None or parsed.password is not None:
                    raise ValueError('Only verified public provenance HTTPS links are allowed')
            if value[:1] in {'/',chr(92)} or re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]',value):
                raise ValueError('Public audit contains a local path')
    check(report)


def frozen_prior_files(root):
    result=preserve(root,candidates(root))
    for name in subprocess.check_output(['git','ls-files','reports/quality_calibration_stress_v1'],cwd=root,text=True).splitlines():
        result[name]=sha256_file(root/name)
    previous=read(root/'reports/quality_calibration_stress_v1/evaluation.json')
    for run in previous['runs']:
        folder=f"artifacts/quality_calibration_stress_v1/models/{run['model_id']}"
        for name,expected in run['artifact_sha256'].items():
            if sha256_file(root/folder/name)!=expected:raise ValueError('Previous calibrated-candidate artifact changed')
            result[folder+'/'+name]=expected
        seal=folder+'/completion.json'
        if sha256_file(root/seal)!=run['completion_sha256']:raise ValueError('Previous scalar study seal changed')
        result[seal]=run['completion_sha256']
    return {name:value for name,value in result.items() if not name.startswith(f'reports/{STUDY}/')}


def source_provenance(root):
    names=('qsart/data/eyeq.py','qsart/data/manifests.py','qsart/preprocessing/fundus.py',
           'scripts/audit_eyeq_data.py',CONFIG,PROTOCOL)
    hashes={name:sha256_file(root/name) for name in names}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    return {'git_revision':revision,'working_tree_dirty':dirty,'source_file_sha256':hashes,'source_sha256':canonical_hash(hashes)}


def differences(profile):
    """Displayed references are evidence of disagreement, never relabeling rules."""
    result=[]
    for split,code,reference,value in [('train','1','pinned_README',1896),('train','1','paper_v4',1876),
        ('test','0','pinned_README_and_paper_v4',8470),('test','1','pinned_README_and_paper_v4',4559)]:
        actual=profile['original_splits'][split]['quality_counts'][code]
        result.append({'original_split':split,'quality_code':int(code),'reference':reference,
            'displayed_count':value,'pinned_csv_count':actual,'difference':actual-value,'labels_changed':False})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-metadata',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--image-root',type=Path)
    parser.add_argument('--inventory-output',type=Path,default=Path('artifacts/eyeq_data_v1/private_image_inventory'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/f'artifacts/{STUDY}/source'
    config=read(root/CONFIG)
    if config['source_revision']!=REVISION or config['binary_harmonization_selected'] is not False:
        raise ValueError('Require frozen source revision and no implicit harmonization')
    records=fetch_source_files(source) if args.fetch_metadata else verify_source_files(source)
    frame=load_annotations(source)
    profile=annotation_profile(frame)
    if args.image_root:
        if args.execute:raise ValueError('A private image inventory is separate from the frozen metadata-only report')
        inventory,summary=inspect_originals(frame,args.image_root)
        destination=(root/args.inventory_output).resolve()
        if destination.exists():raise ValueError('Preserve existing image inventory; choose a new output directory')
        if not destination.is_relative_to(root/'artifacts'):raise ValueError('Private image inventory must stay under project artifacts')
        if destination.is_relative_to(args.image_root.resolve()):raise ValueError('Keep inventory outputs outside the original-image root')
        destination.mkdir(parents=True)
        inventory.to_csv(destination/'images.csv',index=False)
        write_json(destination/'summary.json',summary)
        print(json.dumps(summary))
        return
    if not args.execute:
        print(json.dumps({'status':'metadata_readiness_only','annotation_profile':profile,
            'source_files_verified':len(records),'images_decoded':0,'model_inference_performed':False}))
        return
    provenance=source_provenance(root)
    if provenance['working_tree_dirty']:raise ValueError('Commit tested source/config/protocol before publishing the measured metadata audit')
    target=root/f'reports/{STUDY}/data_audit.json'
    if target.exists():raise ValueError('Completed metadata report exists; do not overwrite or relabel it')
    snapshot=frozen_prior_files(root)
    receipt=source_receipt(source,records)
    private=root/f'artifacts/{STUDY}'
    manifest=private/'external_annotations.csv'
    if manifest.exists():raise ValueError('Preserve previous external annotation manifest')
    frame.to_csv(manifest,index=False)
    subject_map=frame[['original_image_name','source_subject_key','split_group_id','source_official_split']]
    subject_map.to_csv(private/'source_group_lookup.csv',index=False)
    free=shutil.disk_usage(root).free
    report={'schema_version':'qsart.eyeq_annotation_access_audit.v1','experiment_id':STUDY,'dataset':'EyeQ_EyePACS',
        'partition':'external_candidate_unscored','scope':'external_quality_annotation_and_access_readiness_only',
        'source_commit':provenance['git_revision'],'source_file_sha256':provenance['source_file_sha256'],
        'author_source_revision':REVISION,'public_source_files':records,'source_receipt_sha256':sha256_file(source/'source_receipt.json'),
        'annotation_profile':profile,'count_reference_differences':differences(profile),
        'images_decoded':0,'images_downloaded':0,'model_inference_performed':False,
        'classifier_fitting_performed':False,'temperature_fitting_performed':False,'production_model_changed':False,
        'access_review':config,'private_manifest_sha256':sha256_file(manifest),
        'private_group_lookup_sha256':sha256_file(private/'source_group_lookup.csv'),
        'preserved_artifact_sha256':snapshot,'storage_snapshot':{'free_bytes_on_project_volume':free,
            'whole_official_archive_listing_bytes':88_290_000_000,
            'whole_archive_exceeds_current_free_space':free<88_290_000_000,
            'unpacked_or_selected_subset_size_verified':False},
        'readiness':{'metadata_verified':True,'source_photographs_verified':False,
            'download_terms_acceptance_recorded':False,'cross_source_pixel_overlap_checked':False,
            'binary_harmonization_protocol_locked':False,'ready_for_model_evaluation':False},
        'limitations':[
            'Metadata-only audit: no new images, model predictions, image quality accuracy or external validation.',
            'Both author splits are external-only candidates here; no EyeQ training/calibration is authorized by this manifest.',
            'Source subject keys preserve known eye pairs but do not prove pixel, patient or cross-dataset independence.',
            'Good/Usable/Reject are preserved; no middle-class collapse or DeepDRiD binary-label equivalence is assumed.',
            'Pinned CSV counts differ from displayed tables; source labels remain unchanged.',
            'The author README code license does not replace underlying EyePACS image terms or settle annotation permissions.',
            'Official indexed Kaggle page indicates sign-in/rule acceptance; direct terms text was not readable in this audit.',
            'Whole-archive size is an official indexed listing, not a verified EyeQ-subset or unpacked footprint.',
            'Photographs, accepted terms/source receipt, image overlap checks and a locked evaluation protocol are still required.',
            'Research prototype; useful validated triage and independent quality validation remain unestablished.']}
    ensure_aggregate(report)
    for name,expected in snapshot.items():
        if sha256_file(root/name)!=expected:raise ValueError('Preserved artifact changed: '+name)
    write_json(target,report)
    write_json(private/'execution.json',{'source_provenance':provenance,'source_receipt':receipt,
        'protocol_sha256':sha256_file(root/PROTOCOL),'config_sha256':sha256_file(root/CONFIG),
        'report_sha256':sha256_file(target)})
    print(json.dumps({'status':'metadata_audit_completed','images_annotated':len(frame),'source_subject_keys':profile['source_subject_keys'],
                      'images_decoded':0,'ready_for_model_evaluation':False,'report_sha256':sha256_file(target)}))


if __name__=='__main__':main()
