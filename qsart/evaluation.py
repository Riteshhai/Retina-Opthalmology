"""Held-out reporting; all policy decisions use predictions alone."""
from dataclasses import replace
import math
import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from qsart.triage import (METHODS, BudgetInfeasible, PolicyConfig, allocate,
                         screening_metrics)


def grade_metrics(grades, probabilities, referable_grade=2):
    g, p = np.asarray(grades, dtype=int), np.asarray(probabilities, dtype=float)
    pred = p.argmax(1)
    precision, recall, f1, count = precision_recall_fscore_support(
        g, pred, labels=np.arange(5), zero_division=0)
    positive = g >= referable_grade
    risk = p[:, referable_grade:].sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        kappa = float(cohen_kappa_score(g, pred, weights='quadratic', labels=np.arange(5)))
    confidence = p.max(1)
    correct = pred == g
    ece, bins = 0., []
    for i in range(10):
        mask = (confidence >= i/10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
        n = int(mask.sum())
        accuracy = float(correct[mask].mean()) if n else None
        mean = float(confidence[mask].mean()) if n else None
        if n:
            ece += n/len(g)*abs(accuracy-mean)
        bins.append({'count': n, 'accuracy': accuracy, 'confidence': mean})
    return {'n': len(g), 'accuracy': float(accuracy_score(g, pred)),
            'macro_f1': float(f1_score(g, pred, labels=np.arange(5), average='macro', zero_division=0)),
            'quadratic_weighted_kappa': kappa if math.isfinite(kappa) else None,
            'confusion_matrix': confusion_matrix(g, pred, labels=np.arange(5)).tolist(),
            'per_class': [{'grade': i, 'precision': float(precision[i]), 'recall': float(recall[i]),
                           'f1': float(f1[i]), 'support': int(count[i])} for i in range(5)],
            'referable_auroc': float(roc_auc_score(positive, risk)) if len(set(positive)) == 2 else None,
            'referable_auprc': float(average_precision_score(positive, risk)) if positive.any() else None,
            'ece': float(ece), 'reliability_bins': bins,
            'brier': float(((p-np.eye(5)[g])**2).sum(1).mean())}


def grouped_accuracy_ci(grades, probabilities, subject_ids, seed=2026, repeats=1000):
    """Cluster bootstrap; retains all views of sampled subjects, with replacement."""
    g, p, ids = np.asarray(grades), np.asarray(probabilities), np.asarray(subject_ids)
    unique = np.unique(ids)
    if len(unique) < 2:
        return {'lower': None, 'upper': None, 'groups': len(unique)}
    groups = {subject: np.flatnonzero(ids == subject) for subject in unique}
    correct = p.argmax(1) == g
    rng = np.random.default_rng(seed)
    values = [correct[np.concatenate([groups[k] for k in rng.choice(unique, len(unique), replace=True)])].mean()
              for _ in range(repeats)]
    lo, hi = np.quantile(values, [.025, .975])
    return {'lower': float(lo), 'upper': float(hi), 'groups': len(unique),
            'repeats': repeats, 'seed': seed, 'method': 'patient_cluster_percentile_bootstrap'}


def evaluate_budgets(grades, probabilities, quality, uncertainty):
    rows = []
    for review_budget in (.2, .4, .6):
        for recapture_budget in (.1, .2, .4):
            config = PolicyConfig(review_budget, recapture_budget)
            for method in METHODS:
                row = {'method': method, 'review_budget': review_budget,
                       'recapture_budget': recapture_budget}
                try:
                    actions = allocate(probabilities, quality, uncertainty, config, method)
                    row.update(status='evaluated', **screening_metrics(grades, probabilities, actions, config))
                except BudgetInfeasible:
                    row.update(status='infeasible', reason='Required quality deferrals exceed capacities')
                rows.append(row)
    return rows


def ablations(grades, probabilities, quality, uncertainty):
    config = PolicyConfig(.4, .2)
    results = []
    for name in ('full', 'without_quality', 'without_severity_weighting', 'without_separate_budgets'):
        c = replace(config, severity_costs=(1.,)*5) if name == 'without_severity_weighting' else config
        q = np.ones(len(quality)) if name == 'without_quality' else quality
        method = 'combined_budget' if name == 'without_separate_budgets' else 'dual_budget'
        row = {'ablation': name}
        try:
            actions = allocate(probabilities, q, uncertainty, c, method)
            # Alter policy weights, retain identical evaluation weights across ablations.
            row.update(status='evaluated', **screening_metrics(grades, probabilities, actions, config))
            if name == 'without_separate_budgets':
                row['constraint_scope'] = 'combined_deferral_slots_only; separate caps intentionally removed'
        except BudgetInfeasible:
            row['status'] = 'infeasible'
        results.append(row)
    return results
