"""Execute the declared fixed-model quality study; default is input readiness only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import PIL
from PIL import features
import torch
import torchvision
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qsart.evaluation_quality import (CONDITIONS, CONDITION_PARAMETERS, HEADS, RESPONSES, STATES,
    ComponentBootstrap, QualityStudyConfig, prepare_quality_cache, responses, role_frame, run_fixed_model)
from qsart.models.checkpoint import sha256_file
from qsart.training.data import read_manifest, split_fingerprint
from qsart.training.quality_adaptation import MANIFEST_SHA256, canonical_hash, validate_frame, write_json
from scripts.run_quality_adaptation import ensure_aggregate, preserved_files as prior_preserved_files

STUDY = 'quality_calibration_stress_v1'
PROTOCOL = 'docs/research/QUALITY_CALIBRATION_STRESS_V1_PROTOCOL.md'
CONFIG = 'configs/quality_calibration_stress_v1.yaml'
PRIOR = 'reports/quality_adaptation_v1/evaluation_v1_1.json'
PRIOR_SHA256 = 'b21f0305d3d2c19392d91434b7b4dedaed65c9329029b84939143ca8497f2fd8'
LIMITATIONS = [
    'Reused DeepDRiD development selection chose the frozen model checkpoints; this is not fresh confirmation.',
    'The calibration role was excluded from these six models fitting/selection, but used in historical studies of older models.',
    'Temperature-fitting diagnostics are in-sample; selection diagnostics are separate from temperature fitting but development-reused.',
    'One scalar per head preserves class predictions and cannot repair discrimination or missing rare classes.',
    'Corrupted images have no new quality annotations. Stress outputs are score responses, never corrupted quality accuracy.',
    'Expected class indices use a normalized ordinal coding, not a physical or clinical defect scale.',
    'Input blur/exposure/JPEG after 224px preprocessing do not reproduce full acquisition defects; JPEG is codec-bound.',
    'Whole-component intervals condition on fixed models and fitted temperatures, without refitting, reselection or multiplicity correction.',
    'Raw adaptation response contrasts concern fixed weights; calibrated contrasts also reflect separately fitted arm temperatures.',
    'No production model/calibration/policy change, classifier fitting, conformal fitting or clinical utility claim follows.',
    'Consumed confirmation/IDRiD and reserved APTOS test are excluded from this study.',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_provenance(root):
    names = ('qsart/evaluation_quality.py','qsart/evaluation_robustness.py','qsart/calibration/temperature.py',
        'qsart/training/quality_adaptation.py','qsart/training/finetune.py','qsart/training/data.py',
        'qsart/models/networks.py','qsart/models/checkpoint.py','qsart/preprocessing/fundus.py',
        'scripts/run_quality_calibration_stress.py','scripts/run_quality_adaptation.py',CONFIG,PROTOCOL)
    hashes = {name:sha256_file(root/name) for name in names}
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    return {'git_revision':revision,'working_tree_dirty':dirty,'source_file_sha256':hashes,'source_sha256':canonical_hash(hashes)}


def population(root):
    manifest = root/'data/processed/deepdrid/development_v1.csv'
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise ValueError('Frozen development manifest changed')
    full = read_manifest(manifest)
    validate_frame(full)
    if full.split.value_counts().to_dict() != {'train':848,'calibration':168,'validation':168}:
        raise ValueError('Frozen development counts changed')
    selected = full.loc[full.split.isin(['calibration','validation'])].copy()
    role_frame(selected)
    groups = {name:int(part.split_group_id.nunique()) for name,part in selected.groupby('split')}
    patients = {name:int(part.subject_id.nunique()) for name,part in selected.groupby('split')}
    if groups != {'calibration':42,'validation':42} or patients != groups:
        raise ValueError('Frozen role populations changed')
    for row in selected.itertuples():
        if sha256_file(row.image_path) != row.sha256:
            raise ValueError('Development input image bytes changed')
    info = {'manifest_sha256':MANIFEST_SHA256,
        'role_counts':{name:len(part) for name,part in selected.groupby('split')},
        'role_patient_counts':patients,'role_component_counts':groups,
        'split_fingerprints':{name:split_fingerprint(part) for name,part in full.groupby('split')},
        'class_supports':{head:{name:[int(part[head].eq(i).sum()) for i in range(k)]
                              for name,part in selected.groupby('split')} for head,k in HEADS.items()},
        'training_images_opened':False,'calibration_pixels_predicted':True,
        'confirmation_images_opened':False,'idrid_images_opened':False,'aptos_images_opened':False,
        'historical_calibration_role_reuse_disclosed':True}
    return selected,info


def candidates(root):
    if sha256_file(root/PRIOR) != PRIOR_SHA256:
        raise ValueError('Frozen six-model source report changed')
    prior = read_json(root/PRIOR)
    result = []
    expected = [(seed,arm) for seed in (2026,2027,2028) for arm in ('frozen','adapted')]
    if [(r['seed'],r['arm']) for r in prior['runs']] != expected:
        raise ValueError('Frozen candidate order differs')
    for run in prior['runs']:
        seed,arm = run['seed'],run['arm']
        folder = root/f'checkpoints/quality_adaptation_v1/seed{seed}_{arm}'
        if sha256_file(folder/'completion.json') != run['completion_sha256']:
            raise ValueError('Historical model seal changed')
        for name,expected_sha in run['artifact_sha256'].items():
            if sha256_file(folder/name) != expected_sha:
                raise ValueError('Historical quality model artifact changed')
        result.append({'model_id':f'quality_seed{seed}_{arm}','training_seed':seed,'arm':arm,
            'model_version':run['model_version'],'selected_epoch':run['selected_epoch'],
            'checkpoint':str(folder/'best.pt'),'checkpoint_sha256':run['artifact_sha256']['best.pt'],
            'selection_reference':str(folder/'selection_predictions.csv'),
            'selection_reference_sha256':run['artifact_sha256']['selection_predictions.csv']})
    return result


def preserve(root, fixed):
    prior = prior_preserved_files(root)
    for name in subprocess.check_output(['git','ls-files','reports/quality_adaptation_v1'],cwd=root,text=True).splitlines():
        prior[name] = sha256_file(root/name)
    for row in read_json(root/PRIOR)['runs']:
        folder = f"checkpoints/quality_adaptation_v1/seed{row['seed']}_{row['arm']}"
        for name,expected in row['artifact_sha256'].items():
            prior[folder+'/'+name] = expected
        prior[folder+'/completion.json'] = row['completion_sha256']
    return {name:value for name,value in prior.items() if not name.startswith(f'reports/{STUDY}/')}


def summarize(root, frame, runtime, runs, config):
    validation = frame.loc[frame.split.eq('validation')]
    bootstrap = ComponentBootstrap(validation.split_group_id.to_numpy(),draws=config.bootstrap_draws,bootstrap_seed=config.bootstrap_seed)
    states = {run['model_id']:torch.load(root/f"artifacts/{STUDY}/models/{run['model_id']}/inference_state.pt",map_location='cpu',weights_only=True)
              for run in runs}
    contrasts = []
    for training_seed in (2026,2027,2028):
        paired = {arm:next(r for r in runs if r['training_seed']==training_seed and r['arm']==arm) for arm in ('frozen','adapted')}
        for state_name in STATES:
            for condition in CONDITIONS[1:]:
                deltas = {}
                for arm,run in paired.items():
                    state = states[run['model_id']]
                    temperatures = {h:1. for h in HEADS} if state_name=='raw' else run['temperatures']
                    deltas[arm] = responses(state['cells']['validation_'+condition],temperatures)-responses(state['cells']['validation_clean'],temperatures)
                contrasts.append({'training_seed':training_seed,'probability_state':state_name,'condition':condition,
                    'direction':'adapted_condition_minus_clean_minus_frozen_condition_minus_clean',
                    'metrics':bootstrap.intervals(deltas['adapted']-deltas['frozen'],RESPONSES)})
    summary = {}
    for arm in ('frozen','adapted'):
        arm_runs = [r for r in runs if r['arm']==arm]
        summary[arm] = {}
        for state_name in STATES:
            heads = {}
            for head in HEADS:
                heads[head] = {}
                for metric in ('accuracy','macro_f1','nll','brier','ece_10','poor_auroc'):
                    if metric=='poor_auroc' and head!='quality_grade':
                        continue
                    values = [r['clean_metrics']['validation'][state_name][head][metric] for r in arm_runs]
                    heads[head][metric] = {'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
            mean_nll = [np.mean([r['clean_metrics']['validation'][state_name][h]['nll'] for h in HEADS]) for r in arm_runs]
            summary[arm][state_name] = {'heads':heads,'mean_head_nll':{'mean':float(np.mean(mean_nll)),'sample_sd':float(np.std(mean_nll,ddof=1))}}
    report = {'schema_version':'qsart.quality_calibration_stress.v1','experiment_id':STUDY,
        'dataset':'DeepDRiD','partition':'calibration_and_validation','scope':'development_quality_temperature_and_input_stress',
        'source_commit':runtime['source_provenance']['git_revision'],'source_file_sha256':runtime['source_provenance']['source_file_sha256'],
        'protocol_sha256':runtime['protocol_sha256'],'config_sha256':runtime['config_sha256'],
        'population':runtime['population'],'primary_model_id':'quality_seed2026_adapted',
        'primary_calibration_metric':'selection_mean_four_head_nll_temperature_minus_raw',
        'primary_stress_conditions':['blur_2','jpeg_20'],'primary_stress_metric':'temperature_poor_probability_condition_minus_clean',
        'condition_parameters':CONDITION_PARAMETERS,'bootstrap':bootstrap.metadata,
        'runs':runs,'summary':summary,'response_contrasts':contrasts,
        'interval_counts':{'clean_calibration_changes':54,'nonclean_model_responses':288,'adaptation_response_contrasts':144,'zero_clean_response_controls':48},
        'input_cache':runtime['input_cache'],'package_versions':runtime['package_versions'],
        'preserved_artifact_sha256':runtime['preserved'],'prior_quality_report_sha256':PRIOR_SHA256,
        'classifier_fitting_performed':False,'temperature_fitting_performed':True,'conformal_threshold_fitting_performed':False,
        'production_model_changed':False,'corrupted_quality_ground_truth_available':False,'limitations':LIMITATIONS}
    ensure_aggregate(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    config = QualityStudyConfig(**yaml.safe_load((root/CONFIG).read_text(encoding='utf-8')))
    if asdict(config) != asdict(QualityStudyConfig()):
        raise ValueError('Require exact predeclared quality study settings')
    config.validate()
    frame,pop = population(root)
    fixed = candidates(root)
    if not args.execute:
        print(json.dumps({'status':'input_readiness_only','population':{**pop,'calibration_pixels_predicted':False},
            'candidates':[{k:v for k,v in row.items() if k not in {'checkpoint','selection_reference'}} for row in fixed],
            'temperature_fit_or_model_inference_performed':False,'config':asdict(config)}))
        return
    source = source_provenance(root)
    if source['working_tree_dirty']:
        raise ValueError('Commit tested source/protocol/config clean before fitting temperatures or stress inference')
    output = root/f'artifacts/{STUDY}'
    runtime_path = output/'runtime.json'
    if runtime_path.exists():
        if not args.resume:
            raise ValueError('Existing study requires explicit resume')
        runtime = read_json(runtime_path)
        if runtime['source_provenance'] != source or runtime['population'] != pop or runtime['candidates'] != fixed:
            raise ValueError('Resume source/population/frozen candidates changed')
    else:
        runtime = {'source_provenance':source,'population':pop,'candidates':fixed,
            'protocol_sha256':sha256_file(root/PROTOCOL),'config_sha256':sha256_file(root/CONFIG),
            'preserved':preserve(root,fixed), 'package_versions':{'torch':str(torch.__version__),'torchvision':torchvision.__version__,
                'numpy':np.__version__,'Pillow':PIL.__version__,'jpeg_codec':features.version_codec('jpg')}}
    cache,info = prepare_quality_cache(frame,root/'data/processed/deepdrid/development_v1.csv',output/'inputs',config)
    if 'input_cache' in runtime and runtime['input_cache'] != info:
        raise ValueError('Study input cache bytes changed')
    runtime['input_cache'] = info
    if not runtime_path.exists():
        write_json(runtime_path,runtime)
    started = time.perf_counter()
    binding = {'manifest_sha256':MANIFEST_SHA256,'runtime_binding_sha256':canonical_hash(runtime)}
    runs = []
    for candidate in fixed:
        folder = output/'models'/candidate['model_id']
        result = run_fixed_model(candidate,cache,frame,config,folder,binding,resume=args.resume)
        seal = read_json(folder/'completion.json')
        runs.append({**result,'artifact_sha256':seal['artifact_sha256'],'completion_sha256':sha256_file(folder/'completion.json')})
    for name,expected in runtime['preserved'].items():
        if sha256_file(root/name) != expected:
            raise ValueError('Historical artifact changed: '+name)
    for name,expected in source['source_file_sha256'].items():
        if sha256_file(root/name) != expected:
            raise ValueError('Scientific source changed during execution')
    report = summarize(root,frame,runtime,runs,config)
    target = root/f'reports/{STUDY}/evaluation.json'
    if target.exists() and read_json(target) != report:
        raise ValueError('Differing measured report exists; preserve original')
    if not target.exists():
        write_json(target,report)
    print(json.dumps({'event':'quality_calibration_stress_completed','report_sha256':sha256_file(target),
                      'new_execution_seconds':time.perf_counter()-started,'source_commit':source['git_revision']}),flush=True)


if __name__=='__main__':
    main()
