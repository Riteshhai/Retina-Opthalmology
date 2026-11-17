"""Bounded real fitting-role CPU/memory feasibility; no candidate checkpoint export."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qsart.models.checkpoint import load_checkpoint,sha256_file
from qsart.training.finetune import parameter_hash

ROOT=Path(__file__).resolve().parents[1]
SOURCE='reports/aptos_optimization_v2/evaluation.json'
MANIFEST='artifacts/aptos_optimization_v2/runtime_development.csv'
CHECKPOINT='checkpoints/aptos_optimization_v2/seed2026/best.pt'
PROTOCOL='docs/research/APTOS_384_FEASIBILITY_PROTOCOL.md'
PRIVATE=ROOT/'artifacts/aptos_384_feasibility_v1'
PUBLIC=ROOT/'reports/aptos_384_feasibility_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def choose_fitting(frame,count=32):
    if (frame.empty or not frame.split.eq('train').all() or frame.dr_grade.dtype.kind not in 'iu'
        or not frame.dr_grade.isin(range(5)).all() or frame.image_id.duplicated().any()):
        raise ValueError('Only complete unique fitting-role observations accepted')
    if not frame.eligible_for_training.astype(str).eq('True').all():raise ValueError('Training-ineligible input')
    reps=frame.loc[frame.is_group_representative.astype(str).eq('True')].copy()
    if reps.split_group_id.duplicated().any() or len(reps)<count:raise ValueError('Distinct fitting representatives required')
    order=reps.image_id.map(lambda value:hashlib.sha256(('aptos384-feasibility-v1:'+str(value)).encode()).hexdigest())
    return reps.loc[order.sort_values().index[:count]].copy()


def direct_rgb(path,size):
    if size not in (224,384):raise ValueError('Only declared 224/384 geometry accepted')
    with Image.open(path) as source:
        value=np.array(source.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),copy=True)
    return value


def worker(size):
    import torch
    torch.set_num_threads(2);torch.manual_seed(2026)
    selected=pd.read_csv(PRIVATE/'selected_inputs.csv')
    if not selected.split.eq('train').all() or len(selected)!=32:raise ValueError('Require exactly 32 frozen fitting observations')
    checkpoint=ROOT/CHECKPOINT;before=sha256_file(checkpoint)
    source=read(ROOT/SOURCE)
    if before!=source['runs'][0]['checkpoint_sha256']:raise ValueError('Original trained checkpoint changed')
    started=time.perf_counter();pixels=[]
    for row in selected.itertuples():
        if sha256_file(row.image_path)!=row.sha256:raise ValueError('Source image bytes changed')
        pixels.append(direct_rgb(row.image_path,size))
    images=torch.from_numpy(np.stack(pixels)).permute(0,3,1,2).float()/255
    images=(images-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    images=images.contiguous(memory_format=torch.channels_last)
    labels=torch.tensor(selected.dr_grade.to_numpy(),dtype=torch.long)
    preparation=time.perf_counter()-started
    model,metadata=load_checkpoint(checkpoint)
    if metadata['metadata']['backbone']!='efficientnet_b0':raise ValueError('Require the fixed genuine source architecture')
    model=model.to(memory_format=torch.channels_last);model.train();initial=parameter_hash(model)
    weights=torch.tensor(read(PRIVATE/'execution_inputs.json')['class_weights'],dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    losses=[];steps=0;started=time.perf_counter()
    for start in range(0,32,8):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(images[start:start+8])['dr_grade'],labels[start:start+8],weight=weights)
        if not torch.isfinite(loss):raise ValueError('Nonfinite temporary fitting loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite temporary fitting gradients')
        optimizer.step();steps+=1;losses.append(float(loss.detach()))
    elapsed=time.perf_counter()-started
    if parameter_hash(model)==initial or sha256_file(checkpoint)!=before:raise ValueError('Temporary/source parameter state invariant failed')
    print(json.dumps({'image_size':size,'fitting_images':32,'optimizer_steps':steps,'batch_size':8,'cpu_threads':2,
        'preparation_seconds':preparation,'four_update_seconds':elapsed,'all_losses_and_gradients_finite':True,
        'temporary_parameters_changed':True,'source_checkpoint_unchanged':True,'updated_weights_saved':False,
        'torch_version':torch.__version__,'source_checkpoint_sha256':before,'worker_process_id':os.getpid()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',type=int,choices=[224,384]);args=parser.parse_args()
    if args.worker:worker(args.worker);return
    source=read(ROOT/SOURCE)
    if sha256_file(ROOT/SOURCE)!='66c49f12bf630493c9dc38180a73daeeae6a222523067f4ec262971b45d2310e':
        raise ValueError('Fixed source report changed')
    if sha256_file(ROOT/MANIFEST)!=source['input_bindings']['runtime_manifest_sha256']:raise ValueError('Fitting source manifest changed')
    frame=pd.read_csv(ROOT/MANIFEST);fitting=frame.loc[frame.split.eq('train')];selected=choose_fitting(fitting)
    if not args.execute:print(json.dumps({'ready':True,'fitting_images':len(selected),'worker_sizes':[224,384],'new_updates_performed':False}));return
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit tested protocol/runner before real probe')
    if PRIVATE.exists() or PUBLIC.exists():raise ValueError('Preserve previous or partial probe outputs')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    PRIVATE.mkdir(parents=True);PUBLIC.mkdir(parents=True)
    selected.to_csv(PRIVATE/'selected_inputs.csv',index=False)
    counts=np.bincount(fitting.dr_grade.to_numpy(int),minlength=5)
    weights=(len(fitting)/(5*counts)).tolist()
    (PRIVATE/'execution_inputs.json').write_text(json.dumps({'class_weights':weights,'manifest_sha256':sha256_file(ROOT/MANIFEST)},indent=2)+'\n',encoding='utf-8')
    results=[]
    for size in (224,384):
        if psutil.virtual_memory().available<2_000_000_000:raise ValueError('Less than two GB currently available; preserve partial probe')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(size)]
        stdout=PRIVATE/f'worker_{size}.stdout.json';stderr=PRIVATE/f'worker_{size}.stderr.log'
        with stdout.open('wb') as out,stderr.open('wb') as err:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            observed=psutil.Process(process.pid);rss=peak=0;seen=set()
            while process.poll() is None:
                try:
                    processes=[observed,*observed.children(recursive=True)];total=0
                    for child in processes:
                        try:
                            memory=child.memory_info();seen.add(child.pid);total+=memory.rss
                            peak=max(peak,getattr(memory,'peak_wset',0))
                        except psutil.NoSuchProcess:continue
                    rss=max(rss,total)
                except psutil.NoSuchProcess:break
                time.sleep(.02)
            code=process.wait()
        if code:raise RuntimeError(f'Feasibility worker {size} failed; inspect preserved private error log')
        result=read(stdout)
        if result.pop('worker_process_id') not in seen:raise ValueError('Actual worker was not observed by memory sampler')
        result.update({'sampled_peak_process_tree_rss_bytes':rss,'windows_peak_single_process_working_set_bytes':peak or None,
            'memory_measurement_scope':'worker plus venv launcher descendants','actual_worker_observed':True,
            'memory_poll_interval_seconds':.02,'fresh_process':True})
        results.append(result);print(json.dumps({'completed_probe':result}),flush=True)
    names=['scripts/probe_aptos_384.py',PROTOCOL,'qsart/models/checkpoint.py','qsart/models/networks.py']
    report={'schema':'qsart.aptos_384_cpu_feasibility.v1','status':'completed','source_commit':revision,
        'source_file_sha256':{name:sha256_file(ROOT/name) for name in names},'source_report_sha256':sha256_file(ROOT/SOURCE),
        'private_input_manifest_sha256':sha256_file(PRIVATE/'selected_inputs.csv'),'runs':results,
        'classifier_updates_in_temporary_copies':8,'updated_checkpoints_saved':False,'production_model_changed':False,
        'selection_calibration_test_or_external_images_used':False,'shared_workstation_and_download_concurrency':True,
        'limitations':['Four updates per size on 32 source fitting images; no full training, convergence or performance comparison.',
            'Direct resize, active BatchNorm, existing trained initialization and no augmentation differ from the full paper experiment.',
            'Times are feasibility observations with concurrent workloads, not controlled speed benchmarks or precise full-run estimates.']}
    (PUBLIC/'feasibility.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'feasibility_completed','sizes':[224,384],'original_models_preserved':True}))


if __name__=='__main__':main()
