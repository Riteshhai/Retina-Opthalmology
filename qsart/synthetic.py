"""Seeded mock MODEL SCORES for verifying software only, NOT clinical results."""
import numpy as np
import pandas as pd

def make_mock(n=1800, seed=2026, prefix='mock'):
    g = np.random.default_rng(seed)
    grade = g.choice(5, n, p=[.51,.19,.19,.08,.03])
    q = np.clip(g.beta(5,2, size=n) - .085*grade + g.normal(0,.10,n), 0, 1)
    p_rdr = np.clip(.13 + .20*grade + .24*(1-q) + g.normal(0,.22,n), 0, 1)
    p_severe = np.clip(.03 + .16*grade + .13*(1-q) + g.normal(0,.20,n),0,1)
    uncertainty = np.clip(.20 + .46*(1-q) + .18*(1-np.abs(p_rdr-.5)*2) + g.normal(0,.15,n),0,1)
    return pd.DataFrame({'image_id':[f'{prefix}_img_{i:05d}' for i in range(n)],
                         'patient_id':[f'{prefix}_patient_{i//2:05d}' for i in range(n)],
                         'grade':grade, 'quality':q.round(5),
                         'p_rdr':p_rdr.round(5), 'p_severe':p_severe.round(5),
                         'uncertainty':uncertainty.round(5)})
