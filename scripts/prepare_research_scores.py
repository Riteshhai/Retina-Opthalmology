"""Cache real selection-only inference profiles and aggregate v2 diagnostics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from qsart.evaluation import grade_metrics
from qsart.inference.conformal_sidecar import ConformalSidecar
from qsart.inference.engine import InferenceEngine
from qsart.models.checkpoint import sha256_file
from qsart.quality_baselines import binary_quality_metrics
from qsart.training.data import read_manifest
from qsart.triage_v2 import BudgetOverloaded, METHODS, PolicyConfig, allocate_rows, screening_metrics
from qsart.uncertainty.conformal import coverage_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    p.add_argument('--selected', type=Path, default=Path('checkpoints/development_v1/selected'))
    p.add_argument('--conformal', type=Path, default=Path('checkpoints/spec_v1/conformal_r1.json'))
    p.add_argument('--output', type=Path, default=Path('checkpoints/spec_v1/development_scores.json'))
    args = p.parse_args()
    if args.output.exists():
        p.error('Development profile already exists; use a new experiment output')
    frame = read_manifest(args.manifest)
    if not set(frame.split).issubset({'train', 'validation', 'calibration'}):
        p.error('Only development data are allowed in simulator preparation')
    selected = frame[frame.split == 'validation'].reset_index(drop=True)
    if selected.empty:
        p.error('Selection partition is empty')
    for row in selected.itertuples():
        if sha256_file(row.image_path) != row.sha256:
            p.error('A real selection image differs from its frozen SHA256')
    dr, quality, calibration = (args.selected/name for name in ('dr.pt', 'quality.pt', 'calibration.json'))
    engine = InferenceEngine(dr, quality, calibration)
    cp = ConformalSidecar(args.conformal, dr_checkpoint=dr, quality_checkpoint=quality, calibration=calibration)
    if sha256_file(args.manifest) != cp.bundle['manifest_sha256']:
        p.error('Conformal and selection manifest identities differ')
    profiles = {}
    for name, budget, policy in [('single', 0., 'none'), ('budget25', .25, 'budget'), ('all', 1., 'all')]:
        rows = []
        for start in range(0, len(selected), 8):
            images = []
            for path in selected.iloc[start:start+8].image_path:
                with Image.open(path) as image:
                    images.append(image.convert('RGB').copy())
            predicted = engine.predict_images(images, second_pass_budget=budget,
                                               second_pass_policy=policy, include_uncalibrated=True)
            cp.enrich(predicted)
            for row in predicted:
                row.pop('uncalibrated', None)
            rows.extend(predicted)
            print(f'{name}: {len(rows)}/{len(selected)} real selection images', flush=True)
        profiles[name] = {'rows': rows, 'requested_second_pass_budget': budget,
                          'realized_second_pass_fraction': np.mean([row['second_pass_activated'] for row in rows])}
    cache = {'dataset_id': 'development_selection_v2', 'scope': 'development_exploration',
             'partition': 'validation', 'dataset': 'DeepDRiD', 'model_version': engine.model_version,
             'dr_checkpoint_sha256': sha256_file(dr), 'quality_checkpoint_sha256': sha256_file(quality),
             'base_calibration_sha256': sha256_file(calibration), 'conformal_sha256': cp.sha256,
             'manifest_sha256': sha256_file(args.manifest),
             'true_grades': selected.dr_grade.astype(int).tolist(), 'profiles': profiles,
             'activation_scope': 'Fixed eight-image inference request chunks; simulation never reruns inference'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cache, indent=2, allow_nan=False), encoding='utf-8')
    rows = profiles['single']['rows']
    probabilities = np.array([row['grade_probabilities'] for row in rows])
    q = np.array([row['quality']['probabilities'] for row in rows])
    truth = selected.dr_grade.to_numpy(dtype=int)
    conformal = {}
    for name, profile in profiles.items():
        conformal[name] = {method: coverage_metrics(
            [row['conformal'][field] for row in profile['rows']], truth,
            groups=selected.subject_id.to_numpy())
            for method, field in [('ordinal_greedy', 'prediction_set'), ('lac', 'baseline_lac_set')]}
    budgets = []
    for profile, data in profiles.items():
        for review in (.2, .4, .6, .8, 1.):
            for recapture in (.1, .2, .4):
                config = PolicyConfig(review_budget=review, recapture_budget=recapture)
                for method in METHODS:
                    result = {'profile': profile, 'review_budget': review, 'recapture_budget': recapture,
                              'method': method}
                    try:
                        actions = allocate_rows(data['rows'], config, method)
                        result.update(status='evaluated', **screening_metrics(truth,
                            [row['grade_probabilities'] for row in data['rows']], actions, config))
                    except BudgetOverloaded as exc:
                        result.update(status='overloaded', diagnostics=exc.diagnostics)
                    budgets.append(result)
    report = {'experiment': 'spec_v1_alignment', 'spec_version': '1.0', 'policy_version': '2.0',
              'dataset': 'DeepDRiD', 'partition': 'validation', 'scope': 'development_exploration',
              'images': len(selected), 'patients': selected.subject_id.nunique(),
              'model_info': engine.info(), 'conformal_artifact_sha256': cp.sha256,
              'cache_sha256': sha256_file(args.output),
              'protocol_sha256': sha256_file('docs/research/SPEC_V1_ALIGNMENT_PROTOCOL.md'),
              'grading': grade_metrics(truth, probabilities),
              'quality': binary_quality_metrics(selected.quality_grade.to_numpy(dtype=int), q),
              'conformal': conformal, 'budgets': budgets,
              'source_sha256': {name: sha256_file(name) for name in (
                  'scripts/prepare_research_scores.py', 'qsart/triage_v2.py',
                  'qsart/uncertainty/conformal.py', 'qsart/inference/conformal_sidecar.py')},
              'limitations': ['Selection data informed model selection; these are development diagnostics only.',
                              'No new independent confirmation or external validation is claimed.',
                              'Current AUTO_NONREFERABLE metrics do not relabel historical AUTO_SCREEN results.',
                              'Conformal patient-max sets can require more review than budgets allow.',
                              'Conditional-TTA coverage is empirical, without a formal routing-conditional guarantee.']}
    public = Path('reports/spec_v1'); public.mkdir(parents=True, exist_ok=True)
    (public/'development_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'images': len(selected), 'patients': selected.subject_id.nunique(),
        'conformal': conformal, 'policy_scenarios': len(budgets),
        'overloaded': sum(row['status'] == 'overloaded' for row in budgets)}, indent=2))


if __name__ == '__main__':
    main()
