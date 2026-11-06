"""Generated fixtures test role isolation, transformations and exact resume."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from qsart import evaluation_quality as eq
from qsart.models.checkpoint import sha256_file
from qsart.training.data import MEAN, STD, image_to_tensor


@pytest.fixture
def generated(tmp_path):
    torch.set_num_threads(1)
    records = []
    for i in range(8):
        path = tmp_path/f'generated_{i}.png'
        array = np.random.default_rng(i).integers(0,256,(43,59,3),dtype=np.uint8)
        array[:,:5] = 0
        Image.fromarray(array).save(path)
        records.append({'image_path':str(path),'split':'calibration' if i<4 else 'validation',
            'subject_id':f'software_subject_{i//2}','split_group_id':f'software_component_{i//2}',
            'sha256':sha256_file(path),'annotation_status':'verified','official_split':'official_train',
            **{head:i%classes for head,classes in eq.HEADS.items()}})
    frame = pd.DataFrame(records)
    manifest = tmp_path/'generated_manifest.csv'
    frame.to_csv(manifest,index=False)
    config = eq.QualityStudyConfig(batch_size=2,cpu_threads=1,cache_workers=1,bootstrap_draws=50)
    return manifest,frame,config


@pytest.mark.parametrize('role',['train','test','holdout','confirmation','external_holdout'])
def test_forbidden_pixels_rejected_before_file_access(generated,tmp_path,role):
    manifest,frame,config = generated
    frame = frame.copy()
    frame.loc[:1,'split'] = role
    frame.image_path = [f'missing_generated_{i}.png' for i in range(len(frame))]
    with pytest.raises(ValueError,match='calibration/validation roles'):
        eq.prepare_quality_cache(frame,manifest,tmp_path/'never_written',config)
    assert not (tmp_path/'never_written').exists()
    with pytest.raises(ValueError,match='dedicated calibration role'):
        eq.fit_quality_temperatures({},frame.loc[frame.split.eq('validation')])


def test_cache_exact_clean_geometry_and_immutable_source_bytes(generated,tmp_path):
    manifest,frame,config = generated
    cached,info = eq.prepare_quality_cache(frame,manifest,tmp_path/'cache',config)
    hit,hitinfo = eq.prepare_quality_cache(frame,manifest,tmp_path/'cache',config)
    assert info==hitinfo and set(cached['cells'])=={'calibration_clean',*[f'validation_{c}' for c in eq.CONDITIONS]}
    for role in ('calibration','validation'):
        row = frame.loc[frame.split.eq(role)].iloc[0]
        with Image.open(row.image_path) as image:
            expected = image_to_tensor(image,224)
        observed = eq.normalize_cached(hit['cells'][role+'_clean'][:1])[0]
        assert torch.equal(expected,observed)
    for row in frame.itertuples():assert sha256_file(row.image_path)==row.sha256
    cachefile = next((tmp_path/'cache').glob('*.pt'))
    cached['cells']['validation_jpeg_20'][0,0,0,0] ^= 1
    torch.save(cached,cachefile)
    with pytest.raises(ValueError,match='pixels/shape/hash'):
        eq.prepare_quality_cache(frame,manifest,tmp_path/'cache',config)


def test_complete_labels_and_separate_known_groups_required(generated):
    _,frame,_ = generated
    frame = frame.copy()
    frame.loc[0,'artifact_class'] = np.nan
    with pytest.raises(ValueError,match='complete nonempty'):
        eq.role_frame(frame)
    frame.loc[0,'artifact_class'] = 0
    frame.loc[4,'split_group_id'] = frame.loc[0,'split_group_id']
    with pytest.raises(ValueError,match='roles overlap'):
        eq.role_frame(frame)


def test_positive_temperatures_preserve_predictions_and_fit_calibration_only(generated):
    _,frame,_ = generated
    calibration = frame.loc[frame.split.eq('calibration')]
    logits = {head:torch.from_numpy(np.random.default_rng(classes).normal(0,5,(4,classes)).astype(np.float32))
              for head,classes in eq.HEADS.items()}
    temperatures = eq.fit_quality_temperatures(logits,calibration)
    before = eq.clean_metrics(logits,calibration,{head:1. for head in eq.HEADS})
    after = eq.clean_metrics(logits,calibration,temperatures)
    for head in eq.HEADS:
        assert .05<=temperatures[head]<=20
        assert after[head]['nll']<=before[head]['nll']+1e-12
        assert before[head]['confusion_matrix']==after[head]['confusion_matrix']
        assert before[head]['accuracy']==after[head]['accuracy']
        assert sum(b['n'] for b in after[head]['reliability_bins'])==4
    altered = calibration.copy();altered['split']='blur_2'
    with pytest.raises(ValueError,match='declared clean role'):
        eq.clean_metrics(logits,altered,temperatures)
    with pytest.raises(ValueError,match='Positive finite'):
        eq.probabilities(logits,{head:0. for head in eq.HEADS})


def test_component_intervals_match_explicit_all_view_draws_and_keep_rng_fields_distinct():
    groups = np.array(['a','a','b','c','c','c'])
    values = np.array([[1.,2.],[3.,2.],[5.,4.],[7.,4.],[9.,6.],[11.,8.]])
    bootstrap = eq.ComponentBootstrap(groups,draws=200,bootstrap_seed=123)
    observed = bootstrap.intervals(values,['first','second'])
    rng = np.random.default_rng(123)
    explicit = []
    for _ in range(200):
        picked = rng.integers(0,3,3)
        rows = np.concatenate([np.flatnonzero(groups==np.unique(groups)[i]) for i in picked])
        explicit.append(values[rows].mean(0))
    expected = np.quantile(explicit,[.025,.975],axis=0)
    for index,name in enumerate(['first','second']):
        assert observed[name]['difference']==values[:,index].mean()
        assert np.allclose([observed[name]['lower_95'],observed[name]['upper_95']],expected[:,index])
    assert bootstrap.metadata['bootstrap_seed']==123 and 'training_seed' not in bootstrap.metadata
    assert all(v==0 for row in bootstrap.intervals(np.zeros((6,1)),['zero']).values() for v in row.values())


class GeneratedQuality(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3,3,1),nn.BatchNorm2d(3),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.heads = nn.ModuleDict({name:nn.Linear(3,k) for name,k in eq.HEADS.items()})

    def forward(self,images):
        x = self.backbone(images)
        return {name:layer(x) for name,layer in self.heads.items()}


def test_fixed_inference_never_changes_weights_or_buffers(generated):
    _,_,config = generated
    model = GeneratedQuality().eval()
    before = deepcopy(model.state_dict())
    pixels = torch.randint(0,256,(3,224,224,3),dtype=torch.uint8)
    values = eq.predict_quality(model,pixels,config)
    assert all(v.shape==(3,eq.HEADS[h]) for h,v in values.items())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    model.train()
    with pytest.raises(ValueError,match='eval mode'):
        eq.predict_quality(model,pixels,config)


def test_partial_model_resume_preserves_cells_and_temperatures(generated,tmp_path,monkeypatch):
    """Dependency-injected generated fixture, explicitly barred from production evidence."""
    manifest,frame,config = generated
    cached,_ = eq.prepare_quality_cache(frame,manifest,tmp_path/'cache',config)
    torch.manual_seed(4)
    model = GeneratedQuality().eval()
    values = eq.predict_quality(model,cached['cells']['validation_clean'],config)
    reference = pd.DataFrame({'manifest_row':frame.loc[frame.split.eq('validation')].index})
    for head,k in eq.HEADS.items():
        for i in range(k):reference[f'{head}_logit{i}']=values[head][:,i].numpy()
    reference_path = tmp_path/'generated_reference.csv';reference.to_csv(reference_path,index=False,float_format='%.17g')
    checkpoint_path = tmp_path/'injected_software_checkpoint.pt';checkpoint_path.write_bytes(b'Explicit dependency-injected software fixture')
    candidate = {'model_id':'generated_fixture','training_seed':2026,'arm':'adapted','model_version':'software_fixture',
        'selected_epoch':1,'checkpoint':str(checkpoint_path),'checkpoint_sha256':sha256_file(checkpoint_path),
        'selection_reference':str(reference_path),'selection_reference_sha256':sha256_file(reference_path)}
    metadata = {'heads':eq.HEADS,'poor_class_indices':[0],'image_size':224,'task':'quality','data_purpose':'software_test',
        'normalization_mean':MEAN,'normalization_std':STD,'preprocessing':'fundus_crop_rgb_v1',
        'model_version':'software_fixture','manifest_sha256':sha256_file(manifest)}
    def injected_loader(path,*,allow_software_test):
        assert allow_software_test is True
        return deepcopy(model),{'metadata':metadata}
    monkeypatch.setattr(eq,'load_checkpoint',injected_loader)
    atomic = eq.atomic_torch
    def interrupt(state,path):
        atomic(state,path)
        if 'validation_blur_1' in state['cells']:raise InterruptedError('simulated saved-cell interruption')
    monkeypatch.setattr(eq,'atomic_torch',interrupt)
    with pytest.raises(InterruptedError):
        eq.run_fixed_model(candidate,cached,frame,config,tmp_path/'run',{'manifest_sha256':sha256_file(manifest)},allow_software_test=True)
    first = torch.load(tmp_path/'run/inference_state.pt',weights_only=True)
    assert first['temperatures'] is not None
    monkeypatch.setattr(eq,'atomic_torch',atomic)
    monkeypatch.setattr(eq,'fit_quality_temperatures',lambda *a: (_ for _ in ()).throw(AssertionError('must not fit twice')))
    result = eq.run_fixed_model(candidate,cached,frame,config,tmp_path/'run',{'manifest_sha256':sha256_file(manifest)},resume=True,allow_software_test=True)
    last = torch.load(tmp_path/'run/inference_state.pt',weights_only=True)
    assert first['temperatures']==last['temperatures'] and result['data_purpose']=='software_test'
    assert first['cell_sha256']['validation_clean']==last['cell_sha256']['validation_clean']
    assert all('accuracy' not in row for row in result['stress'])
    assert result['stress'][0]['temperature']['changes']['poor_probability']['difference']==0
    (tmp_path/'run/raw_logits.csv').write_text('tampered fixture')
    with pytest.raises(ValueError,match='artifact changed'):
        eq.run_fixed_model(candidate,cached,frame,config,tmp_path/'run',{'manifest_sha256':sha256_file(manifest)},resume=True,allow_software_test=True)


def test_study_config_validation_and_missing_inputs_are_fail_closed():
    for cfg in (replace(eq.QualityStudyConfig(),bootstrap_seed=True),replace(eq.QualityStudyConfig(),cache_workers=3)):
        with pytest.raises(ValueError):cfg.validate()
    from scripts.run_quality_calibration_stress import ensure_aggregate
    with pytest.raises(ValueError):ensure_aggregate({'private':{'image_path':'generated'}})
    assert json.loads(json.dumps({'training_seed':2026,'bootstrap_seed':20261108}))['training_seed']==2026
