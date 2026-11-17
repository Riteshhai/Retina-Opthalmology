"""Fixed quality models: independent-role temperatures and unlabeled input stress.

Only clean annotations are classification truth. Corrupted inputs have response
summaries, never invented quality accuracy. No classifier or production state is
updated by this module.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, features, __version__ as pillow_version
import torch

from qsart.calibration.temperature import fit_temperature
from qsart.evaluation_robustness import CONDITIONS, CONDITION_PARAMETERS, apply_degradation, canonical_rgb
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.finetune import batchnorm_buffer_hash, normalize_cached, parameter_hash
from qsart.training.data import MEAN, STD
from qsart.training.quality_adaptation import HEADS, canonical_hash, metrics_from_logits, write_json

RESPONSES = ('poor_probability', 'expected_clarity', 'expected_field_definition', 'expected_artifact')
STATES = ('raw', 'temperature')


@dataclass
class QualityStudyConfig:
    image_size: int = 224
    batch_size: int = 16
    cpu_threads: int = 2
    cache_workers: int = 2
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 20261108

    def validate(self):
        if self.image_size != 224 or type(self.image_size) is not int:
            raise ValueError('Fixed 224px canonical inputs required')
        for name in ('batch_size', 'cpu_threads', 'cache_workers', 'bootstrap_draws'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive integer batch/thread/worker/draw settings required')
        if self.cache_workers > 2 or type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError('At most two workers and a nonnegative integer bootstrap seed required')


def role_frame(frame):
    required = {'image_path', 'split', 'subject_id', 'split_group_id', 'sha256', *HEADS}
    if (frame.empty or not required.issubset(frame) or frame[list(required)].isna().any().any() or
            set(frame.split) != {'calibration', 'validation'}):
        raise ValueError('Only complete nonempty calibration/validation roles accepted; no fitting/test/confirmation')
    for name in ('subject_id', 'split_group_id', 'sha256', 'image_path'):
        if frame.groupby(name).split.nunique().gt(1).any():
            raise ValueError('Calibration and selection roles overlap')
    if frame.image_path.astype(str).str.casefold().duplicated().any():
        raise ValueError('Duplicate source image paths')
    for name, classes in HEADS.items():
        if not pd.to_numeric(frame[name], errors='coerce').isin(range(classes)).all():
            raise ValueError('Complete integer clean quality labels required')
    for name in ('annotation_status', 'official_split'):
        if name in frame:
            expected = 'verified' if name == 'annotation_status' else 'official_train'
            if not frame[name].eq(expected).all():
                raise ValueError('Verified official-training development rows required')


def pixels_sha(values):
    array = values.numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_torch(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def prepare_quality_cache(frame, manifest, cache_dir, config):
    config.validate()
    role_frame(frame)  # Before opening images or creating output directories.
    content = []
    for row in frame.itertuples():
        actual = sha256_file(row.image_path)
        if actual != row.sha256:
            raise ValueError('Source image bytes changed since frozen manifest')
        content.append(f'{row.Index}:{row.split}:{actual}\n')
    metadata = {'schema': 'qsart.quality_calibration_stress_inputs.v1',
        'manifest_sha256': sha256_file(manifest), 'image_size': 224,
        'input_content_sha256': hashlib.sha256(''.join(content).encode()).hexdigest(),
        'manifest_rows': frame.index.astype(int).tolist(), 'roles': frame.split.tolist(),
        'condition_parameters': CONDITION_PARAMETERS, 'preprocessing': 'fundus_crop_rgb_v1_once_then_stress',
        'pillow_version':pillow_version,'jpeg_codec':features.version_codec('jpg'),
        'pixel_source_sha256':{name:sha256_file(Path(__file__).resolve().parent/name)
                              for name in ('evaluation_robustness.py','preprocessing/fundus.py')}}
    key = canonical_hash(metadata)
    path = Path(cache_dir)/f'quality_inputs_{key}.pt'
    if path.exists():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache.get('metadata') != metadata or cache.get('cache_key') != key:
            raise ValueError('Quality input cache binding mismatch')
    else:
        cells = {'calibration_clean': torch.empty((int(frame.split.eq('calibration').sum()),224,224,3),dtype=torch.uint8)}
        cells.update({f'validation_{name}': torch.empty((int(frame.split.eq('validation').sum()),224,224,3),dtype=torch.uint8)
                      for name in CONDITIONS})
        counters = {'calibration': 0, 'validation': 0}

        def decode(row):
            with Image.open(row.image_path) as handle:
                canonical = canonical_rgb(handle)
            names = ('clean',) if row.split == 'calibration' else CONDITIONS
            return row.split, {name: np.asarray(apply_degradation(canonical, name)).copy() for name in names}

        with ThreadPoolExecutor(max_workers=config.cache_workers) as pool:
            for count, (role, images) in enumerate(pool.map(decode, frame.itertuples()), 1):
                position = counters[role]
                for name, array in images.items():
                    cells[f'{role}_{name}'][position] = torch.from_numpy(array)
                counters[role] += 1
                if count % 50 == 0:
                    print(json.dumps({'event':'quality_cache_progress','images':count,'total':len(frame)}),flush=True)
        cache = {'metadata': metadata, 'cache_key': key, 'cells': cells,
                 'pixel_sha256': {name:pixels_sha(values) for name,values in cells.items()}}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch(cache, path)
    expected_cells = {'calibration_clean', *[f'validation_{name}' for name in CONDITIONS]}
    if set(cache['cells']) != expected_cells or set(cache['pixel_sha256']) != expected_cells:
        raise ValueError('Missing or extra quality cache conditions')
    for name, values in cache['cells'].items():
        role = 'calibration' if name.startswith('calibration_') else 'validation'
        expected = (int(frame.split.eq(role).sum()),224,224,3)
        if values.dtype != torch.uint8 or tuple(values.shape) != expected or pixels_sha(values) != cache['pixel_sha256'][name]:
            raise ValueError('Quality cached pixels/shape/hash mismatch')
    return cache, {'cache_key':key,'file_sha256':sha256_file(path),'pixel_sha256':cache['pixel_sha256'],
                   'shapes':{name:list(values.shape) for name,values in cache['cells'].items()}}


def predict_quality(model, pixels, config):
    config.validate()
    if model.training:
        raise ValueError('Quality evaluation requires eval mode')
    if pixels.dtype != torch.uint8 or tuple(pixels.shape[1:]) != (224,224,3) or not len(pixels):
        raise ValueError('Nonempty canonical uint8 RGB inputs required')
    before = (parameter_hash(model), batchnorm_buffer_hash(model))
    chunks = {name:[] for name in HEADS}
    with torch.inference_mode():
        for indices in torch.arange(len(pixels)).split(config.batch_size):
            output = model(normalize_cached(pixels[indices]).contiguous(memory_format=torch.channels_last))
            if set(output) != set(HEADS):
                raise ValueError('All four fixed quality heads required')
            for name, classes in HEADS.items():
                values = output[name].detach().cpu()
                if tuple(values.shape) != (len(indices),classes) or not torch.isfinite(values).all():
                    raise ValueError('Invalid quality inference logits')
                chunks[name].append(values)
    if before != (parameter_hash(model), batchnorm_buffer_hash(model)):
        raise ValueError('Quality inference changed model parameters or BatchNorm buffers')
    return {name:torch.cat(values) for name,values in chunks.items()}


def probabilities(logits, temperatures):
    if set(logits) != set(HEADS) or set(temperatures) != set(HEADS):
        raise ValueError('All four heads and positive finite temperatures required')
    result = {}
    for name, classes in HEADS.items():
        temperature = temperatures[name]
        if isinstance(temperature,bool) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Positive finite temperatures required')
        values = np.asarray(logits[name],dtype=np.float64)/temperature
        if values.ndim != 2 or values.shape[1] != classes or not np.isfinite(values).all():
            raise ValueError('Finite quality logits required')
        shifted = values-values.max(1,keepdims=True)
        p = np.exp(shifted)
        result[name] = p/p.sum(1,keepdims=True)
    return result


def fit_quality_temperatures(logits, frame):
    if frame.empty or not frame.split.eq('calibration').all():
        raise ValueError('Temperature fitting accepts the dedicated calibration role only')
    result = {}
    for name, classes in HEADS.items():
        if not pd.to_numeric(frame[name],errors='coerce').isin(range(classes)).all() or len(logits[name]) != len(frame):
            raise ValueError('Complete calibration labels/logits required')
        result[name] = fit_temperature(torch.as_tensor(logits[name]),torch.tensor(frame[name].to_numpy(dtype=np.int64,copy=True)))
    return result


def clean_metrics(logits, frame, temperatures):
    if frame.empty or not frame.split.isin(['calibration','validation']).all() or frame.split.nunique() != 1:
        raise ValueError('Clean classification metrics require one declared clean role')
    result = {}
    for name, classes in HEADS.items():
        values = np.asarray(logits[name],dtype=np.float64)/temperatures[name]
        y = frame[name].to_numpy(dtype=int)
        metrics = metrics_from_logits(values,y,classes)
        p = probabilities({head:logits[head] for head in HEADS},temperatures)[name]
        confidence = p.max(1)
        predicted = p.argmax(1)
        bins = np.minimum((confidence*10).astype(int),9)
        reliability = []
        for index in range(10):
            mask = bins==index
            reliability.append({'lower':index/10,'upper':(index+1)/10,'n':int(mask.sum()),
                'confidence':float(confidence[mask].mean()) if mask.any() else None,
                'accuracy':float((predicted[mask]==y[mask]).mean()) if mask.any() else None})
        result[name] = {**metrics,'reliability_bins':reliability}
    return result


def responses(logits, temperatures):
    p = probabilities(logits, temperatures)
    return np.column_stack((p['quality_grade'][:,0], p['clarity_class']@np.arange(5)/4,
                            p['field_definition_class']@np.arange(5)/4,p['artifact_class']@np.arange(6)/5))


class ComponentBootstrap:
    """One common whole-component stream; training seeds have separate fields."""
    def __init__(self, groups, *, draws=2000, bootstrap_seed=20261108):
        ids = np.asarray(groups)
        if ids.ndim != 1 or len(ids) < 2 or pd.isna(ids).any() or any(not str(value).strip() for value in ids):
            raise ValueError('Complete nonempty component identities required')
        if type(draws) is not int or draws < 1 or type(bootstrap_seed) is not int or bootstrap_seed < 0:
            raise ValueError('Positive draws and nonnegative bootstrap seed required')
        unique, self.inverse = np.unique(ids.astype(str),return_inverse=True)
        if len(unique) < 2:
            raise ValueError('At least two components required')
        picked = np.random.default_rng(bootstrap_seed).integers(0,len(unique),size=(draws,len(unique)))
        multiplicity = np.zeros((draws,len(unique)),dtype=np.int64)
        np.add.at(multiplicity,(np.repeat(np.arange(draws),len(unique)),picked.ravel()),1)
        self.multiplicity = multiplicity
        self.counts = np.bincount(self.inverse,minlength=len(unique))
        self.metadata = {'unit':'whole_split_group_id','components':len(unique),'draws':draws,
            'bootstrap_seed':bootstrap_seed,'component_draw_sha256':hashlib.sha256(picked.tobytes()).hexdigest(),
            'confidence_level':.95,'multiplicity_adjusted':False,'refitting_or_reselection':False}

    def intervals(self, values, names):
        values = np.asarray(values,dtype=float)
        if values.ndim == 1:
            values = values[:,None]
        if values.shape != (len(self.inverse),len(names)) or not np.isfinite(values).all() or len(set(names)) != len(names):
            raise ValueError('Aligned finite outcome columns and unique names required')
        sums = np.zeros((len(self.counts),len(names)))
        np.add.at(sums,self.inverse,values)
        sampled = self.multiplicity@sums/(self.multiplicity@self.counts)[:,None]
        bounds = np.quantile(sampled,[.025,.975],axis=0)
        return {name:{'difference':float(values[:,index].mean()),'lower_95':float(bounds[0,index]),
                      'upper_95':float(bounds[1,index])} for index,name in enumerate(names)}


def calibration_changes(logits, frame, temperatures, bootstrap):
    raw = probabilities(logits,{name:1. for name in HEADS})
    scaled = probabilities(logits,temperatures)
    nll_columns, brier_columns = [], []
    for name, classes in HEADS.items():
        y = frame[name].to_numpy(dtype=int)
        losses = []
        for t in (1.,temperatures[name]):
            z = np.asarray(logits[name],dtype=np.float64)/t
            z -= z.max(1,keepdims=True)
            losses.append(np.log(np.exp(z).sum(1))-z[np.arange(len(y)),y])
        nll_columns.append(losses[1]-losses[0])
        onehot = np.eye(classes)[y]
        brier_columns.append(np.square(scaled[name]-onehot).sum(1)-np.square(raw[name]-onehot).sum(1))
    values = np.column_stack([*nll_columns,*brier_columns,np.mean(nll_columns,axis=0)])
    names = [*[name+'_nll' for name in HEADS],*[name+'_brier' for name in HEADS],'mean_head_nll']
    return bootstrap.intervals(values,names)


def run_fixed_model(candidate, cache, frame, config, folder, binding, *, resume=False, allow_software_test=False):
    """Atomic inference cells and a separate one-time scalar-fit state."""
    role_frame(frame)
    config.validate()
    folder = Path(folder)
    digest = canonical_hash({'candidate':candidate,'binding':binding,'config':asdict(config)})
    if folder.exists() and not resume:
        raise ValueError('Existing model study output requires explicit resume')
    if (folder/'completion.json').exists():
        seal = json.loads((folder/'completion.json').read_text())
        if seal['binding_sha256'] != digest:
            raise ValueError('Completed model source/input bindings changed')
        for name, expected in seal['artifact_sha256'].items():
            if sha256_file(folder/name) != expected:
                raise ValueError('Completed quality study artifact changed')
        return json.loads((folder/'result.json').read_text())
    path = Path(candidate['checkpoint'])
    if sha256_file(path) != candidate['checkpoint_sha256']:
        raise ValueError('Frozen quality checkpoint changed')
    if sha256_file(candidate['selection_reference']) != candidate['selection_reference_sha256']:
        raise ValueError('Frozen selection reference changed')
    torch.set_num_threads(config.cpu_threads)
    model, checkpoint = load_checkpoint(path,allow_software_test=allow_software_test)
    meta = checkpoint['metadata']
    purpose = 'software_test' if allow_software_test else 'real_research'
    if (meta['heads'] != HEADS or meta.get('poor_class_indices') != [0] or meta['image_size'] != 224 or
            meta['task'] != 'quality' or meta['data_purpose'] != purpose or
            meta.get('normalization_mean') != MEAN or meta.get('normalization_std') != STD or
            meta.get('preprocessing') != 'fundus_crop_rgb_v1' or
            meta['model_version'] != candidate['model_version'] or meta['manifest_sha256'] != binding['manifest_sha256']):
        raise ValueError('Frozen quality model metadata differs from protocol')
    model.to(memory_format=torch.channels_last).eval().requires_grad_(False)
    torch.set_num_threads(config.cpu_threads)
    folder.mkdir(parents=True,exist_ok=True)
    state_path = folder/'inference_state.pt'
    if state_path.exists():
        state = torch.load(state_path,map_location='cpu',weights_only=True)
        if state['binding_sha256'] != digest:
            raise ValueError('Partial model source/input bindings changed')
        for cell,logits in state['cells'].items():
            if state['cell_sha256'].get(cell) != canonical_hash({name:pixels_sha(v) for name,v in logits.items()}):
                raise ValueError('Partial inference cell bytes changed')
        if state['temperatures'] is not None and state.get('temperature_sha256') != canonical_hash(state['temperatures']):
            raise ValueError('Partial fitted temperatures changed')
    else:
        state = {'binding_sha256':digest,'cells':{},'cell_sha256':{},'temperatures':None}
    # Clean selection first provides a regression check, never a transform fit.
    order = ['validation_clean','calibration_clean',*[f'validation_{c}' for c in CONDITIONS if c!='clean']]
    for cell in order:
        if cell not in state['cells']:
            state['cells'][cell] = predict_quality(model,cache['cells'][cell],config)
            state['cell_sha256'][cell] = canonical_hash({name:pixels_sha(v) for name,v in state['cells'][cell].items()})
            atomic_torch(state,state_path)
        if cell == 'validation_clean':
            reference = pd.read_csv(candidate['selection_reference'],float_precision='round_trip')
            validation = frame.loc[frame.split.eq('validation')]
            if not np.array_equal(reference.manifest_row.to_numpy(),validation.index.to_numpy()):
                raise ValueError('Selection regression identities differ')
            deltas = {}
            for name, classes in HEADS.items():
                old = reference[[f'{name}_logit{i}' for i in range(classes)]].to_numpy(float)
                actual = state['cells'][cell][name].numpy()
                deltas[name] = float(np.max(np.abs(actual-old)))
                if deltas[name]>2e-4 or not np.array_equal(actual.argmax(1),old.argmax(1)):
                    raise ValueError('Frozen selection-logit regression failed')
            state['clean_regression'] = {'all_argmax_identical':True,'maximum_logit_delta':deltas}
        if cell == 'calibration_clean' and state['temperatures'] is None:
            state['temperatures'] = fit_quality_temperatures(state['cells'][cell],frame.loc[frame.split.eq('calibration')])
            state['temperature_sha256'] = canonical_hash(state['temperatures'])
            atomic_torch(state,state_path)
        print(json.dumps({'event':'quality_study_cell','model_id':candidate['model_id'],'cell':cell}),flush=True)
    temperatures = state['temperatures']
    raw_t = {name:1. for name in HEADS}
    result = {name:candidate[name] for name in ('model_id','training_seed','arm','model_version','checkpoint_sha256','selected_epoch')}
    result.update(data_purpose=purpose,temperatures=temperatures,clean_regression=state['clean_regression'],clean_metrics={},
                  classifier_parameters_updated=False,batchnorm_buffers_updated=False)
    all_rows = []
    for role in ('calibration','validation'):
        part = frame.loc[frame.split.eq(role)]
        logits = state['cells'][role+'_clean']
        result['clean_metrics'][role] = {label:clean_metrics(logits,part,t) for label,t in (('raw',raw_t),('temperature',temperatures))}
        for name in HEADS:
            before = result['clean_metrics'][role]['raw'][name]
            after = result['clean_metrics'][role]['temperature'][name]
            if before['confusion_matrix'] != after['confusion_matrix']:
                raise ValueError('Positive temperature changed clean class predictions')
        conditions = ('clean',) if role=='calibration' else CONDITIONS
        for condition in conditions:
            logits = state['cells'][role+'_'+condition]
            rows = {'manifest_row':part.index.to_numpy(),'role':role,'condition':condition}
            for name,classes in HEADS.items():
                for index in range(classes):
                    rows[f'{name}_logit{index}'] = logits[name][:,index].numpy()
            all_rows.append(pd.DataFrame(rows))
    validation = frame.loc[frame.split.eq('validation')]
    bootstrap = ComponentBootstrap(validation.split_group_id.to_numpy(),draws=config.bootstrap_draws,bootstrap_seed=config.bootstrap_seed)
    result['calibration_changes'] = calibration_changes(state['cells']['validation_clean'],validation,temperatures,bootstrap)
    result['stress'] = []
    for condition in CONDITIONS:
        row = {'condition':condition,'images':len(validation),'corrupted_quality_ground_truth_available':False}
        for label,t in (('raw',raw_t),('temperature',temperatures)):
            current = responses(state['cells']['validation_'+condition],t)
            clean = responses(state['cells']['validation_clean'],t)
            delta = current-clean
            row[label] = {'levels':{name:float(current[:,i].mean()) for i,name in enumerate(RESPONSES)},
                          'changes':bootstrap.intervals(delta,RESPONSES),
                          'fraction_poor_score_increased':float((delta[:,0]>0).mean()),
                          'fraction_clarity_score_decreased':float((delta[:,1]<0).mean())}
        result['stress'].append(row)
    if sha256_file(path) != candidate['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint file changed during evaluation')
    pd.concat(all_rows,ignore_index=True).to_csv(folder/'raw_logits.csv',index=False,float_format='%.17g')
    write_json(folder/'temperatures.json',{'schema':'qsart.quality_probability_temperature.v1',
        'checkpoint_sha256':candidate['checkpoint_sha256'],'role':'calibration','heads':temperatures,
        'data_purpose':purpose,'production_use':False,'source_binding_sha256':digest})
    write_json(folder/'result.json',result)
    names = ('inference_state.pt','raw_logits.csv','temperatures.json','result.json')
    write_json(folder/'completion.json',{'binding_sha256':digest,'artifact_sha256':{n:sha256_file(folder/n) for n in names}})
    return result
