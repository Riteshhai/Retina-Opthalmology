"""Patient-cluster intervals conditional on the locked cohort's assigned actions."""
import numpy as np


def cluster_mean_intervals(outcomes, subject_ids, repeats=1000, seed=2026):
    x, ids = np.asarray(outcomes, dtype=float), np.asarray(subject_ids)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) != len(ids) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite outcome rows and corresponding subject identifiers required')
    if repeats < 20:
        raise ValueError('Use at least 20 bootstrap replicates')
    unique = np.unique(ids)
    if len(unique) < 2:
        return [{'mean': float(v), 'lower': None, 'upper': None} for v in x.mean(0)]
    sums = np.stack([x[ids == subject].sum(0) for subject in unique])
    counts = np.array([(ids == subject).sum() for subject in unique])
    sampled = np.random.default_rng(seed).integers(0, len(unique), size=(repeats, len(unique)))
    values = sums[sampled].sum(1)/counts[sampled].sum(1)[:, None]
    bounds = np.quantile(values, [.025, .975], axis=0)
    return [{'mean': float(mean), 'lower': float(low), 'upper': float(high)}
            for mean, low, high in zip(x.mean(0), bounds[0], bounds[1])]


def policy_outcomes(grades, probabilities, actions):
    from qsart.triage import Action
    grades, p = np.asarray(grades, dtype=int), np.asarray(probabilities)
    auto = np.array([a == Action.AUTO_SCREEN for a in actions])
    recapture = np.array([a == Action.RECAPTURE for a in actions])
    predicted = p.argmax(1)
    return np.column_stack(((grades >= 2) & (predicted < 2) & auto,
                            (grades >= 3) & recapture,
                            np.array([1., 1., 3., 6., 10.])[grades]*(predicted != grades)*auto,
                            auto.astype(float)))


OUTCOME_NAMES = ('missed_referable_auto_per_image', 'severe_recapture_per_image',
                 'weighted_auto_grade_error_per_image', 'automated_coverage')
