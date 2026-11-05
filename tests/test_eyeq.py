"""Generated CSV/JPEG fixtures verify metadata and image-inventory contracts."""
from copy import deepcopy
import hashlib
import json

import httpx
import pandas as pd
from PIL import Image
import pytest

from qsart.data import eyeq
from qsart.training.data import read_manifest


def csv(path,names,qualities=None,grades=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    n=len(names)
    pd.DataFrame({'Unnamed: 0':range(n),'image':names,'quality':qualities or [i%3 for i in range(n)],
                  'DR_grade':grades or [i%5 for i in range(n)]}).to_csv(path,index=False)
    return path


def release(tmp_path):
    csv(tmp_path/'data/Label_EyeQ_train.csv',['1_left.jpeg','1_right.jpeg','2_left.jpeg'],[0,1,2],[4,0,3])
    csv(tmp_path/'data/Label_EyeQ_test.csv',['3_left.jpeg','3_right.jpeg','4_right.jpeg'],[2,0,1],[2,1,0])
    return eyeq.load_annotations(tmp_path,data_purpose='software_test')


def test_three_class_labels_stay_separate_from_dr_and_training(tmp_path):
    frame=release(tmp_path)
    assert frame.eyeq_quality_code.tolist()==[0,1,2,2,0,1]
    assert frame.eyeq_quality_name.tolist()==['Good','Usable','Reject','Reject','Good','Usable']
    assert frame.source_dr_grade.tolist()==[4,0,3,2,1,0]
    assert frame.split.eq('external_holdout').all() and not frame.eligible_for_training.any()
    assert not {'quality_grade','dr_grade','clarity_class','artifact_class'}.intersection(frame.columns)
    assert frame.annotation_status.eq('software_fixture').all()
    path=tmp_path/'generated_external_manifest.csv';frame.to_csv(path,index=False)
    with pytest.raises(ValueError):read_manifest(path)
    profile=eyeq.annotation_profile(frame)
    assert profile['images_annotated']==6 and profile['source_subject_keys']==4
    assert profile['cross_split_subject_keys']==0 and profile['binary_harmonization_selected'] is False
    assert profile['original_splits']['train']['quality_discordant_eye_pairs']==1
    assert profile['original_splits']['train']['dr_discordant_eye_pairs']==1


def test_real_release_hash_and_original_partition_are_mandatory(tmp_path):
    path=csv(tmp_path/'fixture.csv',['1_left.jpeg'])
    with pytest.raises(ValueError,match='exact pinned author'):eyeq.read_annotations(path,'train')
    with pytest.raises(ValueError,match='Explicit author'):eyeq.read_annotations(path,'validation',data_purpose='software_test')


@pytest.mark.parametrize('name',['../1_left.jpeg','C:\\1_left.jpeg','1_left.png','1_left.JPG','01_left.jpeg','1_middle.jpeg',' 1_left.jpeg'])
def test_processed_aliases_paths_and_noncanonical_ids_fail_closed(tmp_path,name):
    path=csv(tmp_path/'fixture.csv',[name])
    with pytest.raises(ValueError,match='original EyePACS'):eyeq.read_annotations(path,'train',data_purpose='software_test')
    with pytest.raises(ValueError,match='Exact original'):eyeq.resolve_original(tmp_path,name,'train')


@pytest.mark.parametrize('column,value',[('quality','3'),('quality','1.0'),('quality',''),('DR_grade','5'),('Unnamed: 0','2')])
def test_source_schema_index_and_codes_cannot_silently_change(tmp_path,column,value):
    path=csv(tmp_path/'fixture.csv',['1_left.jpeg'])
    frame=pd.read_csv(path,dtype=str,keep_default_na=False);frame.loc[0,column]=value;frame.to_csv(path,index=False)
    with pytest.raises(ValueError):eyeq.read_annotations(path,'train',data_purpose='software_test')


def test_duplicate_images_rejected_but_shared_subject_keys_reported_without_relabeling(tmp_path):
    csv(tmp_path/'data/Label_EyeQ_train.csv',['1_left.jpeg'])
    csv(tmp_path/'data/Label_EyeQ_test.csv',['1_right.jpeg'])
    frame=eyeq.load_annotations(tmp_path,data_purpose='software_test')
    assert eyeq.annotation_profile(frame)['cross_split_subject_keys']==1
    assert frame.source_official_split.tolist()==['train','test']
    csv(tmp_path/'data/Label_EyeQ_test.csv',['1_left.jpeg'])
    with pytest.raises(ValueError,match='duplicate across'):eyeq.load_annotations(tmp_path,data_purpose='software_test')


def test_binary_sensitivity_view_requires_explicit_choice_and_never_rewrites_source(tmp_path):
    frame=release(tmp_path);before=frame.copy(deep=True)
    assert eyeq.binary_view(frame,endpoint='reject_vs_nonreject').tolist()==[False,False,True,True,False,False]
    assert eyeq.binary_view(frame,endpoint='usable_or_reject_vs_good').tolist()==[False,True,True,True,False,True]
    with pytest.raises(ValueError,match='named EyeQ'):eyeq.binary_view(frame,endpoint=None)
    pd.testing.assert_frame_equal(frame,before)


def test_private_original_inventory_preserves_bytes_and_never_enables_evaluation(tmp_path):
    frame=release(tmp_path/'labels');root=tmp_path/'originals';(root/'train').mkdir(parents=True)
    Image.new('RGB',(75,57),'red').save(root/'train/1_left.jpeg',format='JPEG')
    (root/'train/1_right.jpeg').write_bytes((root/'train/1_left.jpeg').read_bytes())
    (root/'train/2_left.jpeg').write_bytes(b'generated corrupt software fixture')
    (root/'test').mkdir();Image.new('RGB',(75,57),'blue').save(root/'test/3_left.jpeg',format='PNG')
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    inventory,summary=eyeq.inspect_originals(frame,root)
    assert summary['images_readable']==2 and summary['images_invalid']==2 and summary['images_missing']==2
    assert summary['exact_duplicate_groups']==1 and summary['ready_for_model_evaluation'] is False
    assert not inventory.eligible_for_training.any() and not inventory.eligible_for_model_evaluation.any()
    assert all(path.read_bytes()==data for path,data in before.items())
    Image.new('RGB',(75,57),'green').save(root/'1_left.jpeg',format='JPEG')
    with pytest.raises(ValueError,match='Ambiguous'):eyeq.resolve_original(root,'1_left.jpeg','train')


def test_no_network_photo_routes_and_no_replacement_of_changed_metadata(tmp_path,monkeypatch):
    payload=b'Explicit generated metadata fixture\n'
    sha=hashlib.sha256(payload).hexdigest();blob=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    monkeypatch.setattr(eyeq,'SOURCE_FILES',{'README.md':(sha,blob)})
    calls=[];client_class=httpx.Client
    def reply(request):
        calls.append(str(request.url));return httpx.Response(200,content=payload)
    monkeypatch.setattr(eyeq.httpx,'Client',lambda **kwargs:client_class(**kwargs,transport=httpx.MockTransport(reply)))
    assert eyeq.fetch_source_files(tmp_path)['README.md']['sha256']==sha
    assert len(calls)==1 and calls[0].endswith('/README.md')
    with pytest.raises(ValueError,match='Only the pinned'):eyeq.source_url('original_img/1_left.jpeg')
    (tmp_path/'README.md').write_bytes(b'changed local fixture')
    with pytest.raises(ValueError,match='do not overwrite'):eyeq.fetch_source_files(tmp_path)
    assert (tmp_path/'README.md').read_bytes()==b'changed local fixture'


def test_source_receipt_cannot_be_silently_replaced(tmp_path):
    records={'generated_metadata_fixture':{'sha256':'a'*64}}
    first=eyeq.source_receipt(tmp_path,records)
    assert eyeq.source_receipt(tmp_path,records)==first and first['photographs_downloaded'] is False
    changed=deepcopy(records);changed['generated_metadata_fixture']['sha256']='b'*64
    with pytest.raises(ValueError,match='differing source receipt'):eyeq.source_receipt(tmp_path,changed)
    assert json.loads((tmp_path/'source_receipt.json').read_text())==first
