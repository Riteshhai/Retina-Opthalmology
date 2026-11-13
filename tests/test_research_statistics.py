import numpy as np
import pytest
from qsart.research_statistics import cluster_mean_intervals


def test_paired_difference_retains_patient_clusters():
    a = np.array([1., 1., 0., 0.])
    b = a+.25
    interval = cluster_mean_intervals(b-a, ['p1', 'p1', 'p2', 'p2'], repeats=100)[0]
    assert interval == {'mean': .25, 'lower': .25, 'upper': .25}
    assert cluster_mean_intervals(a, ['p1']*4)[0]['lower'] is None
    with pytest.raises(ValueError):
        cluster_mean_intervals([np.nan], ['p1'])
