"""Supplementary descriptive natural-defect strata from already frozen predictions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from qsart.models.checkpoint import sha256_file
from qsart.research_statistics import cluster_mean_intervals
from qsart.training.data import read_manifest


def main():
    report_path = Path('reports/confirmation_v1/evaluation.json')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    manifest = Path('data/processed/deepdrid/confirmation_v1.csv')
    private = Path('artifacts/confirmation_v1/predictions.partial.json')
    saved = json.loads(private.read_text(encoding='utf-8'))
    if (sha256_file(manifest) != report['confirmation_manifest_sha256']
            or saved['selection_lock_sha256'] != report['selection_lock_sha256']
            or saved['completed_images'] != report['images']):
        raise ValueError('Frozen prediction, manifest or selection identity differs')
    frame = read_manifest(manifest)
    probabilities = np.array([row['grade_probabilities'] for row in saved['results']])
    if probabilities.shape != (len(frame), 5):
        raise ValueError('Frozen prediction shape differs from the cohort')
    truth, predicted = frame.dr_grade.to_numpy(dtype=int), probabilities.argmax(1)
    if confusion_matrix(truth, predicted, labels=range(5)).tolist() != report['grading']['confusion_matrix']:
        raise ValueError('Frozen predictions do not reproduce the published confusion matrix')
    error = predicted != truth
    entropy = np.array([row['uncertainty']['entropy'] for row in saved['results']])
    strata = []
    for column in ('quality_grade', 'clarity', 'field_definition', 'artifact'):
        for value in sorted(frame[column].dropna().unique()):
            for grade in [None, *range(5)]:
                mask = frame[column].eq(value).to_numpy(copy=True)
                if grade is not None:
                    mask &= truth == grade
                if not mask.any():
                    continue
                true, pred = truth[mask], predicted[mask]
                referable, severe = true >= 2, true >= 3
                missed = referable & (pred < 2)
                ci = cluster_mean_intervals((~error[mask]).astype(float), frame.loc[mask, 'subject_id'])[0]
                strata.append({'natural_annotation': column, 'source_code': int(value),
                               'reference_dr_grade': grade, 'images': int(mask.sum()),
                               'patients': int(frame.loc[mask, 'subject_id'].nunique()),
                               'accuracy': float(np.mean(pred == true)),
                               'accuracy_ci95_lower': ci['lower'], 'accuracy_ci95_upper': ci['upper'],
                               'true_referable': int(referable.sum()),
                               'missed_referable': int(missed.sum()),
                               'referable_false_negative_rate': float(missed.sum()/referable.sum())
                               if referable.any() else None,
                               'true_severe': int(severe.sum()),
                               'severe_predicted_nonreferable': int((severe & (pred < 2)).sum()),
                               'mean_entropy': float(entropy[mask].mean()),
                               'sparse_stratum': int(mask.sum()) < 20})
    output = report_path.parent
    pd.DataFrame(strata).to_csv(output/'natural_defect_strata.csv', index=False)
    summary = {
        'scope': 'Supplementary descriptive analysis after locked confirmation; no tuning or causal claim',
        'confirmation_report_sha256': sha256_file(report_path),
        'private_prediction_file_sha256': sha256_file(private),
        'images': len(frame), 'patients': frame.subject_id.nunique(),
        'entropy_error_detection_auroc': float(roc_auc_score(error, entropy)),
        'strata_rows': len(strata),
        'interval_method': '1000 patient-cluster resamples per observed stratum, seed 2026',
        'limitations': ['Natural defects are associated with disease/acquisition; not randomized damage.',
                        'Conditional subgroup intervals ignore model-fitting uncertainty and multiplicity.',
                        'Small strata are flagged; absent strata are not estimated.',
                        'Source codes retain their original meaning; artifact direction differs from clarity.',
                        'False negatives here use unselected first-pass grades; no deferred outcomes are assumed.']}
    (output/'natural_defect_analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
