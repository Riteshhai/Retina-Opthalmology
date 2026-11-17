"""Generated selection/geometry fixtures establish the fitting-only probe boundary."""
import hashlib

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.probe_aptos_384 import choose_fitting,direct_rgb


def fixture():
    return pd.DataFrame({'image_id':['fixture'+str(i) for i in range(40)],'split_group_id':['g'+str(i) for i in range(40)],
        'split':'train','dr_grade':np.arange(40)%5,'eligible_for_training':True,'is_group_representative':True})


def test_fitting_selection_is_deterministic_and_rejects_heldout_roles():
    frame=fixture();first=choose_fitting(frame)
    assert first.image_id.tolist()==choose_fitting(frame.sample(frac=1,random_state=7)).image_id.tolist()
    assert len(first)==32 and first.split_group_id.nunique()==32
    for field,value in [('split','test'),('split','validation'),('split','calibration'),('eligible_for_training',False),('dr_grade',9)]:
        bad=frame.copy();bad.loc[0,field]=value
        with pytest.raises(ValueError):choose_fitting(bad)


def test_direct_resize_is_rgb_exact_geometry_and_preserves_original_bytes(tmp_path):
    p=tmp_path/'generated.png';Image.new('RGBA',(29,51),(32,88,144,255)).save(p)
    before=hashlib.sha256(p.read_bytes()).hexdigest()
    for size in (224,384):
        value=direct_rgb(p,size)
        assert value.shape==(size,size,3) and value.dtype==np.uint8
        np.testing.assert_array_equal(value[0,0],[32,88,144])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==before
    with pytest.raises(ValueError):direct_rgb(p,512)
