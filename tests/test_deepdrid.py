import pandas as pd
import pytest
from qsart.deepdrid import normalize_labels, audit


def _example(path, patient):
    d=pd.DataFrame({
       'patient_id':[patient]*4,
       'image_id':[f'{patient}_l1',f'{patient}_l2',f'{patient}_r1',f'{patient}_r2'],
       'Overall quality':[1,0,0,1],
       'left_eye_DR_Level':[2,2,float('nan'),float('nan')],
       'right_eye_DR_Level':[float('nan'),float('nan'),4,4],
       'patient_DR_Level':[4]*4,
       'Clarity':[10,5,6,10], 'Field definition':[8]*4, 'Artifact':[0,6,8,0]})
    d.to_csv(path,index=False)


def test_eye_grade_not_patient_grade(tmp_path):
    f=tmp_path/'train.csv'; _example(f,1)
    d=normalize_labels(f,'train')
    assert d.grade.tolist()==[2,2,4,4]
    assert d.patient_id.nunique()==1


def test_source_patient_overlap_rejected(tmp_path):
    a,b=tmp_path/'a.csv',tmp_path/'b.csv'
    _example(a,1); _example(b,1)
    with pytest.raises(ValueError,match='overlap'):
        audit(a,b)


def test_crosstab_patient_disjoint(tmp_path):
    a,b=tmp_path/'a.csv',tmp_path/'b.csv'
    _example(a,1); _example(b,2)
    report=audit(a,b)
    assert report['train']['images']==4
    assert report['train']['patients']==1
    assert report['train']['quality_vs_eye_grade']['1']['2']==1
