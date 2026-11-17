"""Select on development validation, calibrate separately, then lock confirmation inputs."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.calibration.temperature import write_calibration
from qsart.data.development import check_confirmation_independence, confirmation_image_overlap
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    p.add_argument('--confirmation', type=Path, default=Path('data/processed/deepdrid/confirmation_v1.csv'))
    p.add_argument('--root', type=Path, default=Path('checkpoints/development_v1'))
    args = p.parse_args()
    destination = args.root/'selected'
    if (destination/'selection_lock.json').exists():
        p.error('Selection is already locked. Do not overwrite it after inspecting confirmation outcomes.')
    development, confirmation = read_manifest(args.development), read_manifest(args.confirmation)
    inspected = set(read_manifest('data/processed/deepdrid/pilot_manifest.csv').query("split == 'holdout'").subject_id)
    check_confirmation_independence(confirmation, development, inspected)
    overlap = confirmation_image_overlap(confirmation, development)
    private = Path('artifacts/development_v1'); private.mkdir(parents=True, exist_ok=True)
    (private/'confirmation_overlap_audit.json').write_text(json.dumps(overlap, indent=2), encoding='utf-8')
    if not overlap['passed']:
        p.error('Unresolved cross-cohort image fingerprint matches; confirmation remains closed.')
    manifest_hash = sha256_file(args.development)
    candidates = []
    for name in ('dr_mobilenet', 'dr_resnet18'):
        path = args.root/name/'best.pt'
        _, checkpoint = load_checkpoint(path)
        metadata = checkpoint['metadata']
        if metadata['manifest_sha256'] != manifest_hash or metadata['task'] != 'dr':
            p.error('Candidate manifest/task differs from frozen development definition')
        metrics = metadata['selection_metrics']['dr_grade']
        if not all(math.isfinite(metrics[k]) for k in ('macro_f1', 'nll')):
            p.error('Invalid development-selection metrics')
        candidates.append({'name': name, 'path': str(path), 'backbone': metadata['backbone'],
                           'metrics': metrics, 'selected_epoch': checkpoint['epoch']+1,
                           'checkpoint_sha256': sha256_file(path)})
    winner = sorted(candidates, key=lambda item: (-item['metrics']['macro_f1'], item['metrics']['nll'], item['name']))[0]
    quality_path = args.root/'quality_mobilenet/best.pt'
    _, quality = load_checkpoint(quality_path)
    if quality['metadata']['manifest_sha256'] != manifest_hash or quality['metadata']['task'] != 'quality':
        p.error('Quality checkpoint is not from this development experiment')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner['path'], destination/'dr.pt')
    shutil.copy2(quality_path, destination/'quality.pt')
    write_calibration(destination/'dr.pt', destination/'quality.pt', args.development,
                       destination/'calibration.json', cpu_threads=2, batch_size=16)
    lock = {'experiment': 'development_v1', 'selection_rule': 'highest validation macro-F1, then lowest unweighted NLL',
            'development_manifest_sha256': manifest_hash,
            'confirmation_manifest_sha256': sha256_file(args.confirmation),
            'protocol_sha256': sha256_file('docs/research/DEVELOPMENT_V1_PROTOCOL.md'),
            'triage_source_sha256': sha256_file('qsart/triage.py'),
            'evaluation_source_sha256': sha256_file('scripts/evaluate_confirmation.py'),
            'evaluation_dependency_sha256': {str(path).replace('\\', '/'): sha256_file(path)
                                            for path in sorted(Path('qsart').rglob('*.py'))},
            'candidates': candidates, 'selected_candidate': winner['name'],
            'quality_selection_metrics': quality['metadata']['selection_metrics'],
            'dr_checkpoint_sha256': sha256_file(destination/'dr.pt'),
            'quality_checkpoint_sha256': sha256_file(destination/'quality.pt'),
            'calibration_sha256': sha256_file(destination/'calibration.json'),
            'classical_quality_sha256': sha256_file(args.root/'classical_quality.json'),
            'confirmation_predictions_inspected': False,
            'confirmation_images': len(confirmation), 'confirmation_patients': confirmation.subject_id.nunique(),
            'cross_cohort_fingerprint_audit_passed': True,
            'note': 'Internal same-domain confirmation; not external validation or clinical approval.'}
    (destination/'selection_lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    report = Path('reports/development_v1'); report.mkdir(parents=True, exist_ok=True)
    (report/'selection_lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    print(json.dumps(lock, indent=2))


if __name__ == '__main__':
    main()
