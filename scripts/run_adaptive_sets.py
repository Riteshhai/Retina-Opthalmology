"""Fixed APS/RAPS saved-prediction study; default verifies inputs without new sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qsart.data.eyeq import verify_source_files
from qsart.evaluation_adaptive import evaluate, probabilities, validate_saved_frames
from qsart.models.checkpoint import sha256_file
from qsart.training.quality_adaptation import write_json
from scripts.audit_eyeq_data import ensure_aggregate, frozen_prior_files

STUDY='adaptive_sets_v1'
SOURCE='reports/aptos_optimization_v2/evaluation.json'
CONFIG='configs/adaptive_sets_v1.json'
PROTOCOL='docs/research/ADAPTIVE_SETS_V1_PROTOCOL.md'
SCORES='artifacts/aptos_optimization_v2/scores'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_inputs(root):
    config=read(root/CONFIG)
    if (config['alphas']!=[.1,.05] or config['uniform_seeds']!=list(range(2026,2031))
        or config['model_training_seed']!=2026 or config['raps_k_reg']!=2 or config['raps_lambda']!=.1
        or config['bootstrap_draws']!=2000 or config['bootstrap_seed']!=20261109):
        raise ValueError('Require the declared fixed comparison, not outcome-driven reconfiguration')
    if sha256_file(root/SOURCE)!=config['source_report_sha256']:raise ValueError('Frozen source report changed')
    source=read(root/SOURCE)
    frames=[];bindings={SOURCE:sha256_file(root/SOURCE)}
    for split in ('calibration','validation'):
        name=f'{split}_scores.csv';path=root/SCORES/name
        if sha256_file(path)!=source['private_score_sha256'][name]:raise ValueError('Saved source logits changed')
        frame=pd.read_csv(path,float_precision='round_trip')
        frames.append(frame);bindings[f'{SCORES}/{name}']=sha256_file(path)
    calibration,selection=frames
    cp_mask,reps=validate_saved_frames(calibration,selection)
    if (len(calibration),len(selection),int(cp_mask.sum()),int(reps.sum()))!=(527,521,254,507):
        raise ValueError('Require the exact original development populations')
    runtime_name='artifacts/aptos_optimization_v2/runtime_development.csv'
    roles_name='artifacts/aptos_optimization_v2/runtime_calibration_roles.csv'
    for name,key in [(runtime_name,'runtime_manifest_sha256'),(roles_name,'runtime_calibration_roles_sha256')]:
        if sha256_file(root/name)!=source['input_bindings'][key]:raise ValueError('Source development roles changed')
        bindings[name]=sha256_file(root/name)
    runtime=pd.read_csv(root/runtime_name)
    for split,frame in [('calibration',calibration),('validation',selection)]:
        expected=runtime.loc[runtime.split.eq(split)]
        for name in ('image_id','split_group_id','dr_grade','is_group_representative'):
            if frame[name].astype(str).tolist()!=expected[name].astype(str).tolist():
                raise ValueError('Saved predictions do not align with source development identities/labels')
    roles=pd.read_csv(root/roles_name).set_index('image_id').loc[calibration.image_id]
    if roles.calibration_role.tolist()!=calibration.calibration_role.tolist():raise ValueError('Source role mismatch')
    checkpoint='checkpoints/aptos_optimization_v2/seed2026/best.pt'
    if sha256_file(root/checkpoint)!=source['runs'][0]['checkpoint_sha256']:raise ValueError('Fixed source checkpoint changed')
    bindings[checkpoint]=sha256_file(root/checkpoint)
    return config,source,calibration,selection,cp_mask,reps,bindings


def preserve(root, bindings):
    result={**frozen_prior_files(root),**bindings}
    for path in (root/'reports/eyeq_data_v1').glob('*'):
        if path.is_file():result[path.relative_to(root).as_posix()]=sha256_file(path)
    previous=read(root/'reports/eyeq_data_v1/data_audit.json')
    source=root/'artifacts/eyeq_data_v1/source'
    for name,record in verify_source_files(source).items():
        if record!=previous['public_source_files'][name]:raise ValueError('Preserve prior EyeQ metadata source')
        result[(source/name).relative_to(root).as_posix()]=record['sha256']
    expected_private={'source/source_receipt.json':previous['source_receipt_sha256'],
        'external_annotations.csv':previous['private_manifest_sha256'],
        'source_group_lookup.csv':previous['private_group_lookup_sha256'],
        'comparative_ddr/README.md':previous['access_review']['comparative_source']['readme_sha256'],
        'comparative_ddr/LICENSE':previous['access_review']['comparative_source']['code_license_sha256']}
    for name,expected in expected_private.items():
        path=root/'artifacts/eyeq_data_v1'/name
        if sha256_file(path)!=expected:raise ValueError('Preserve prior private EyeQ evidence')
        result[path.relative_to(root).as_posix()]=expected
    return result


def controls(comparisons, previous):
    result=[]
    for old in previous['comparisons']:
        row=next(x for x in comparisons if all(x[k]==old[k] for k in ('method','alpha','randomization_seed')))
        gap=abs(row['artifact']['qhat']-old['artifact']['qhat'])
        if gap>1e-12 or any(row[state]!=old[state] for state in ('raw','argmax_fallback')):
            raise ValueError('Historical LAC/ordinal control did not reproduce')
        result.append({'method':row['method'],'alpha':row['alpha'],'randomization_seed':row['randomization_seed'],
                       'threshold_absolute_delta':gap,'both_populations_and_empty_policies_identical':True})
    return result


def summarize(comparisons):
    rows=[]
    for alpha in (.1,.05):
        for method in ('lac','ordinal_greedy','ordinal_greedy_randomized','aps','raps'):
            candidates=[row for row in comparisons if row['alpha']==alpha and row['method']==method]
            for state in ('raw','argmax_fallback'):
                values=[row[state]['all_selection_images'] for row in candidates]
                row={'alpha':alpha,'method':method,'empty_policy':state,'uniform_replicates':len(values)}
                for key in ('marginal_coverage','mean_set_size','empty_set_count'):
                    numbers=[x[key] for x in values]
                    row[key]={'mean':float(np.mean(numbers)),'sample_standard_deviation':float(np.std(numbers,ddof=1)) if len(numbers)>1 else None}
                rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config,source,calibration,selection,cp_mask,reps,bindings=load_inputs(root)
    snapshot=preserve(root,bindings)
    if not args.execute:
        print(json.dumps({'status':'saved_input_readiness_verified','calibration_representatives':int(cp_mask.sum()),
            'selection_images':len(selection),'preserved_artifacts':len(snapshot),'new_prediction_sets_generated':False,
            'images_decoded':0,'model_inference_performed':False}));return
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('Commit tested source/config/protocol clean before new score outcomes')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    public=root/f'reports/{STUDY}/evaluation.json';private=root/f'artifacts/{STUDY}/prediction_sets.json'
    if public.exists() or private.exists():raise ValueError('Preserve completed or partial study outputs')
    names=['qsart/uncertainty/adaptive.py','qsart/uncertainty/conformal.py','qsart/uncertainty/randomized_ordinal.py',
        'qsart/evaluation_adaptive.py','scripts/run_adaptive_sets.py',CONFIG,PROTOCOL]
    provenance={name:sha256_file(root/name) for name in names}
    temperature=source['calibration']['temperature']
    columns=[f'logit_{k}' for k in range(5)]
    cp_p=probabilities(calibration.loc[cp_mask,columns].to_numpy(),temperature)
    p=probabilities(selection[columns].to_numpy(),temperature)
    comparisons,paired,sets=evaluate(cp_p,calibration.loc[cp_mask,'dr_grade'].to_numpy(int),p,
        selection.dr_grade.to_numpy(int),selection.split_group_id.to_numpy(),reps,config)
    baseline=controls(comparisons,source)
    if len(comparisons)!=34 or len(paired)!=40:raise ValueError('Incomplete declared comparison')
    report={'schema_version':'qsart.adaptive_sets_research.v1','specification':'MASTER_SPEC_v1.0',
        'experiment_id':STUDY,'dataset':'APTOS_2019','partition':'validation',
        'scope':'reused_development_fixed_aps_raps_score_comparison','model_version':source['model_version'],
        'source_commit':revision,'source_file_sha256':provenance,'source_report_sha256':bindings[SOURCE],
        'configuration':config,'population':{'temperature_images_previously_used':262,'conformal_representatives':254,
            'selection_images':521,'selection_fingerprint_representatives':507,'patient_identity_available':False,
            'historical_calibration_and_selection_reuse':True,'reserved_test_or_external_inputs_used':False},
        'fixed_temperature':temperature,'model_training_seed':2026,'comparisons':comparisons,
        'paired_comparisons':paired,'summary':summarize(comparisons),'baseline_reproduction':baseline,
        'bootstrap':{'draws':2000,'seed':config['bootstrap_seed'],'confidence_level':.95,
            'method':'paired_whole_fingerprint_component_percentile_fixed_sets','interval_count':160,
            'same_draws_across_all_contrasts':True,'refitting_or_reselection':False,'multiplicity_adjusted':False},
        'images_decoded':0,'model_inference_performed':False,'classifier_fitting_performed':False,
        'temperature_fitting_performed':False,'raps_parameter_tuning_performed':False,'production_model_changed':False,
        'preserved_artifact_sha256':snapshot,'private_sets_sha256':None,
        'limitations':['Reused development logits/calibration roles and consumed prior IDRiD outcomes; no fresh confirmation.',
            'Fixed RAPS k_reg=2/lambda=.1 matches initial author CFG, not final tuned procedure or full paper reproduction.',
            'One 224px source classifier and fixed temperature; five uniform seeds are not independent model-training replicates.',
            'Representatives/fingerprint components are not verified patient identities; label-dependent quarantine/stratified splits persist.',
            'Marginal/grade inclusion and bootstrap intervals are descriptive; no patient, conditional, external or clinical guarantee.',
            'Raw empties and argmax enlargement are separate; no method or parameter is selected for deployment.']}
    for name,expected in snapshot.items():
        if sha256_file(root/name)!=expected:raise ValueError('Preserved artifact changed')
    write_json(private,sets);report['private_sets_sha256']=sha256_file(private)
    ensure_aggregate(report);write_json(public,report)
    print(json.dumps({'status':'completed','comparisons':34,'paired_intervals':160,'historical_controls_reproduced':len(baseline),
                      'report_sha256':sha256_file(public),'images_decoded':0,'model_inference_performed':False}))


if __name__=='__main__':main()
