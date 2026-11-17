"""Fit a new independent-temperature CP sidecar; preserve original model outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader

from qsart.calibration.temperature import calibration_metrics, fit_temperature
from qsart.inference.conformal_sidecar import content_hash, temperature_probabilities
from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import ManifestDataset, read_manifest, split_fingerprint
from qsart.uncertainty.conformal import fit_conformal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    p.add_argument('--selected', type=Path, default=Path('checkpoints/development_v1/selected'))
    p.add_argument('--output', type=Path, default=Path('checkpoints/spec_v1/conformal.json'))
    args = p.parse_args()
    if args.output.exists():
        p.error('Sidecar already exists; use a new declared experiment rather than overwriting')
    frame = read_manifest(args.manifest)
    if not set(frame.split).issubset({'train', 'validation', 'calibration'}):
        p.error('Only development partitions are accepted')
    dr_path, quality_path = args.selected/'dr.pt', args.selected/'quality.pt'
    calibration_path = args.selected/'calibration.json'
    model, checkpoint = load_checkpoint(dr_path)
    metadata = checkpoint['metadata']
    if (metadata['data_purpose'] != 'real_research' or metadata['task'] != 'dr'
            or metadata['manifest_sha256'] != sha256_file(args.manifest)):
        p.error('Genuine selected development model and unchanged manifest required')
    calibration = frame[frame.split == 'calibration'].copy()
    if split_fingerprint(calibration) != metadata['split_fingerprints']['calibration']:
        p.error('Dedicated calibration partition differs from training provenance')
    groups = sorted(calibration.subject_id.unique(), key=lambda value: (
        hashlib.sha256(f'spec-v1-conformal-20261020:{value}'.encode()).hexdigest(), str(value)))
    if len(groups) != 42 or len(calibration) != 168:
        p.error('This protocol requires the frozen 42-patient calibration group')
    temp = calibration[calibration.subject_id.isin(groups[:21])].reset_index(drop=True)
    cp = calibration[calibration.subject_id.isin(groups[21:])].reset_index(drop=True)
    assert set(temp.subject_id).isdisjoint(cp.subject_id)
    torch.set_num_threads(2)

    def infer(part):
        dataset = ManifestDataset(part, {'dr_grade': 5}, metadata['image_size'],
                                  mean=metadata['normalization_mean'], std=metadata['normalization_std'])
        first, flipped, labels = [], [], []
        with torch.inference_mode():
            for images, targets in DataLoader(dataset, batch_size=16, num_workers=0):
                original = model(images)['dr_grade']
                second = model(images.flip(-1))['dr_grade']
                first.append(original); flipped.append((original+second)/2)
                labels.append(targets['dr_grade'])
        return {'first_pass': torch.cat(first), 'all_flip': torch.cat(flipped)}, torch.cat(labels)

    fitting_logits, fitting_labels = infer(temp)
    print('Independent temperature logits: 84 images / 21 patients', flush=True)
    cp_logits, cp_labels = infer(cp)
    temperatures = {path: fit_temperature(logits, fitting_labels) for path, logits in fitting_logits.items()}
    temp_hash = content_hash(temperatures)
    binding = {'dr_checkpoint_sha256': sha256_file(dr_path),
               'quality_checkpoint_sha256': sha256_file(quality_path),
               'base_calibration_sha256': sha256_file(calibration_path),
               'temperature_sha256': temp_hash, 'manifest_sha256': sha256_file(args.manifest),
               'split_sha256': split_fingerprint(cp),
               'temperature_fit_split_sha256': split_fingerprint(temp), 'data_purpose': 'real_research'}
    models = {}
    for path, logits in cp_logits.items():
        probabilities = temperature_probabilities(logits.softmax(1).numpy(), temperatures[path])
        models[path] = {str(alpha): {method: fit_conformal(probabilities, cp_labels.numpy(),
                       alpha=alpha, method=method, groups=cp.subject_id.to_numpy(),
                       metadata={**binding, 'probability_path': path})
                       for method in ('lac', 'ordinal_greedy')} for alpha in (.1, .05)}
    artifact = {**binding, 'format_version': 1, 'experiment': 'spec_v1_alignment',
                'model_version': metadata['model_version'], 'independent_temperature': temperatures,
                'temperature_images': len(temp), 'temperature_patients': temp.subject_id.nunique(),
                'conformal_images': len(cp), 'conformal_patients': cp.subject_id.nunique(),
                'patient_groups_disjoint': True, 'models': models,
                'protocol_sha256': sha256_file('docs/research/SPEC_V1_ALIGNMENT_PROTOCOL.md'),
                'source_sha256': {name: sha256_file(name) for name in (
                    'scripts/calibrate_conformal.py', 'qsart/uncertainty/conformal.py',
                    'qsart/inference/conformal_sidecar.py')},
                'temperature_fit_diagnostics': {path: calibration_metrics(logits, fitting_labels, temperatures[path])
                                                for path, logits in fitting_logits.items()},
                'limitations': ['Development-only addition after v1 confirmation was consumed.',
                                'Original stratified sampling and conditional TTA do not justify a blanket coverage guarantee.',
                                'Ordinal score is an independently implemented deterministic paper-inspired variant.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    private = Path('artifacts/spec_v1'); private.mkdir(parents=True, exist_ok=True)
    (private/'conformal_partition.json').write_text(json.dumps({
        'temperature_subjects': list(map(str, groups[:21])), 'conformal_subjects': list(map(str, groups[21:]))}))
    public = Path('reports/spec_v1'); public.mkdir(parents=True, exist_ok=True)
    (public/'conformal_calibration.json').write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'fitted', 'temperature_images': len(temp), 'conformal_images': len(cp),
                      'conformal_patients': cp.subject_id.nunique(), 'independent_temperatures': temperatures,
                      'artifact_sha256': sha256_file(args.output)}, indent=2))


if __name__ == '__main__':
    main()
