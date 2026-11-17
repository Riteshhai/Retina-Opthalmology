"""Offline triage policy research. NOT a medical device or a diagnostic model."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

ACTIONS = ('routine', 'recapture', 'review')
MANDATORY = ('image_id', 'patient_id', 'grade', 'quality', 'p_rdr', 'p_severe', 'uncertainty')

@dataclass(frozen=True)
class Thresholds:
    quality_min: float = .50
    refer_risk: float = .48
    suspect_risk: float = .24
    severe_risk: float = .40
    uncertainty_max: float = .62
    def as_dict(self): return asdict(self)


def validate(df):
    missing = sorted(set(MANDATORY) - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df.empty: raise ValueError('No rows in input')
    if df.image_id.isna().any() or df.patient_id.isna().any():
        raise ValueError('Missing IDs')
    if df.image_id.astype(str).duplicated().any():
        raise ValueError('Duplicate image_id in input')
    for col in ('quality', 'p_rdr', 'p_severe', 'uncertainty'):
        values = pd.to_numeric(df[col], errors='coerce')
        if values.isna().any() or not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f'{col} must be finite values in [0,1]')
    grades = pd.to_numeric(df.grade, errors='coerce')
    if grades.isna().any() or not grades.isin([0,1,2,3,4]).all():
        raise ValueError('grade must be integer 0..4')
    return df


def decide(df, t=Thresholds(), method='qsart'):
    """Return retrospective policy routes. quality=estimated gradability [0,1].

    p_rdr=P(grade>=2), p_severe=P(grade>=3), uncertainty in [0,1].
    These are model SCORE INPUTS, never inferred in this package.
    """
    validate(df)
    q = df.quality.to_numpy(dtype=float)
    r = df.p_rdr.to_numpy(dtype=float)
    s = df.p_severe.to_numpy(dtype=float)
    u = df.uncertainty.to_numpy(dtype=float)
    low = q < t.quality_min
    if method == 'qsart':
        # Conservative rules: a low-quality image never gets the routine outcome.
        # A suspicious low-quality image goes to review instead of recapture.
        review = (s >= t.severe_risk) | (r >= t.refer_risk) | (low & (r >= t.suspect_risk)) | (~low & (u > t.uncertainty_max))
        recapture = low & ~review
    elif method == 'quality_first':
        recapture = low
        review = ~low & (r >= t.refer_risk)
    elif method == 'confidence_only':
        recapture = np.zeros(len(df), dtype=bool)
        review = (r >= t.refer_risk) | (u > t.uncertainty_max)
    elif method == 'risk_only':
        recapture = np.zeros(len(df), dtype=bool)
        review = r >= t.refer_risk
    else:
        raise ValueError(f'Unknown method: {method}')
    return np.select([review, recapture], ['review', 'recapture'], default='routine')


def metrics(df, actions):
    validate(df)
    a = np.asarray(actions)
    if len(a) != len(df) or not np.isin(a, ACTIONS).all():
        raise ValueError('Invalid actions')
    grade = df.grade.to_numpy(dtype=int)
    rdr = grade >= 2
    severe = grade >= 3
    routine, review, recapture = (a == 'routine'), (a == 'review'), (a == 'recapture')
    missed = int(np.sum(rdr & routine))
    clearance = max(int(routine.sum()), 1)
    positives = max(int(rdr.sum()), 1)
    return {
        'n_images': int(len(a)), 'n_patients': int(df.patient_id.nunique()),
        'n_rdr': int(rdr.sum()), 'n_severe': int(severe.sum()),
        'routine_rate': float(routine.mean()),
        'review_rate': float(review.mean()),
        'recapture_rate': float(recapture.mean()),
        'missed_rdr': missed,
        'missed_rdr_per_1000_images': 1000 * missed / len(a),
        'missed_rdr_per_1000_routine': 1000 * missed / clearance,
        'missed_severe': int(np.sum(severe & routine)),
        'severe_deferred_to_recapture': int(np.sum(severe & recapture)),
        'rdr_deferred_to_recapture': int(np.sum(rdr & recapture)),
        'rdr_review_fraction': float(np.sum(rdr & review) / positives),
        'rdr_recapture_fraction': float(np.sum(rdr & recapture) / positives),
        'rdr_routine_fraction': float(missed / positives),
        'low_quality_routine_count': int(np.sum((df.quality.to_numpy() < .5) & routine)),
        'low_quality_missed_rdr': int(np.sum((df.quality.to_numpy() < .5) & rdr & routine)),
    }


def tune(calibration, *, max_review=.32, max_recapture=.24, min_routine=.30, method='qsart'):
    """Pick one deterministic feasible policy from a fixed grid using CALIBRATION only.

    Does not guarantee prospective safety or data-shift coverage. Raises if infeasible.
    """
    validate(calibration)
    if any(not (0 <= x <= 1) for x in (max_review, max_recapture, min_routine)):
        raise ValueError('Budget fractions must be in [0,1]')
    best = None
    # Coarse grid is intentionally small for CPU and auditability.
    for quality_min in (.35, .50, .65):
        for refer_risk in (.30, .40, .50, .60):
            for suspect_risk in (.12, .20, .28, .36):
                if suspect_risk >= refer_risk: continue
                for uncertainty_max in (.50, .65, .80):
                    t = Thresholds(quality_min, refer_risk, suspect_risk, .40, uncertainty_max)
                    m = metrics(calibration, decide(calibration, t, method=method))
                    if (m['review_rate'] > max_review + 1e-10 or
                        m['recapture_rate'] > max_recapture + 1e-10 or
                        m['routine_rate'] < min_routine - 1e-10):
                        continue
                    # Multiobjective proxy: avoid both clearing referable disease AND
                    # deferring severe disease to recapture without human review.
                    # This is NOT a clinical utility function.
                    key = (m['missed_rdr_per_1000_images'],
                           m['missed_severe'] + m['severe_deferred_to_recapture'],
                           m['rdr_deferred_to_recapture'],
                           m['review_rate'], -m['routine_rate'])
                    if best is None or key < best[0]: best = (key, t, m)
    if best is None:
        raise ValueError('No feasible thresholds under budgets; report infeasibility. Do not relax silently.')
    return best[1], best[2]
