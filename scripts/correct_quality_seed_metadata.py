"""Append a versioned metadata correction, preserving the original measured report.

No classifier fitting, predictions, checkpoint selection, metric or interval
computation occurs here. The three paired rows follow the declared training-seed
order; v1 accidentally overwrote those labels with the bootstrap RNG seed.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.models.checkpoint import sha256_file
from qsart.training.quality_adaptation import SEEDS, write_json

ORIGINAL_SHA256 = '18e184b5abb4c67a53a17a9887b9b4321f34a904f46e714a925c2b46fdcff188'


def corrected_report(original):
    result = deepcopy(original)
    if (original.get('schema_version') != 'qsart.quality_adaptation.v1' or
            original.get('source_commit') != '2d4243c977a0669776a5dc356d6d6e3166732a12' or
            [(r['seed'], r['arm']) for r in original['runs']] != [(s,a) for s in SEEDS for a in ('frozen','adapted')] or
            len(original['paired_changes']) != 3 or any(row['seed'] != 20261107 for row in original['paired_changes'])):
        raise ValueError('Original report does not match the known metadata-only defect')
    result['schema_version'] = 'qsart.quality_adaptation.v1.1'
    result['metadata_correction'] = {'record': 'metadata_correction.json',
        'reason': 'Separate paired-comparison training seed from bootstrap RNG seed',
        'original_report_sha256': ORIGINAL_SHA256, 'new_fitting_or_metric_recomputation': False}
    for seed, row in zip(SEEDS, result['paired_changes'], strict=True):
        row['bootstrap_seed'] = row['seed']
        row['seed'] = seed
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/'reports/quality_adaptation_v1'
    original_path = folder/'evaluation.json'
    target = folder/'evaluation_v1_1.json'
    receipt = folder/'metadata_correction.json'
    if sha256_file(original_path) != ORIGINAL_SHA256 or target.exists() or receipt.exists():
        raise ValueError('Require exact original report and absent correction outputs')
    original = json.loads(original_path.read_text(encoding='utf-8'))
    result = corrected_report(original)
    write_json(target, result)
    write_json(receipt, {'status': 'metadata_only_correction',
        'original_report': 'evaluation.json', 'original_report_sha256': ORIGINAL_SHA256,
        'corrected_report': 'evaluation_v1_1.json', 'corrected_report_sha256': sha256_file(target),
        'changed_fields': ['schema_version', 'metadata_correction', 'paired_changes[].seed', 'paired_changes[].bootstrap_seed'],
        'training_seeds': list(SEEDS), 'bootstrap_seed': 20261107,
        'models_epochs_predictions_metrics_intervals_unchanged': True,
        'new_fitting_or_metric_recomputation': False, 'correction_script_sha256': sha256_file(__file__),
        'postexecution_source_file_sha256': {'qsart/training/quality_adaptation.py': sha256_file(root/'qsart/training/quality_adaptation.py')}})
    assert sha256_file(original_path) == ORIGINAL_SHA256
    print(json.dumps({'status': 'metadata_corrected', 'corrected_report_sha256': sha256_file(target)}))


if __name__ == '__main__':
    main()
