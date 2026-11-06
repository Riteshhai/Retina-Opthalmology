import numpy as np
from qsart.evaluation import grade_metrics, grouped_accuracy_ci


def test_group_bootstrap_and_missing_class_metrics():
    p = np.array([[.9, .025, .025, .025, .025], [.025, .025, .9, .025, .025]])
    m = grade_metrics([0, 2], p)
    assert m['accuracy'] == 1
    assert m['per_class'][4]['support'] == 0
    ci = grouped_accuracy_ci([0, 2], p, ['a', 'b'], repeats=30)
    assert ci['lower'] == ci['upper'] == 1
