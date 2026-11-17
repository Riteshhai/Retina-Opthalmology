"""Review frozen IDRiD pairs and reserve label-independent sensitivity populations."""
from collections import Counter
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from qsart.data.idrid_provenance import idrid_source_confirmation
from qsart.data.idrid_review import compare_images, freeze_populations
from qsart.data.manifests import sha256_file
from scripts.inspect_datasets import default_aptos_root
from scripts.prepare_idrid import default_idrid_root, load_references

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SHA = 'b31891fc14c5c52ab1e638821a5e014762e20d2add6a936ab18d093ef165f88a'
SOURCES = ('scripts/review_idrid.py', 'qsart/data/idrid_review.py', 'qsart/data/idrid_provenance.py',
           'qsart/data/idrid.py', 'qsart/data/aptos.py', 'configs/idrid_data_v1.json',
           'qsart/data/manifests.py', 'qsart/preprocessing/fundus.py', 'scripts/inspect_datasets.py',
           'scripts/prepare_idrid.py', 'docs/research/IDRID_REVIEW_V1_PROTOCOL.md')


def run(source_commit):
    begin = time.perf_counter()
    private, output = ROOT/'artifacts/idrid_review_v1', ROOT/'reports/idrid_review_v1/data_review.json'
    if private.exists() or output.exists():
        raise ValueError('Preserve completed/partial review outputs; choose a new declared identity')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if revision != source_commit or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Pair review requires the declared clean committed source')
    sources = {}
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError('Review source does not match committed bytes')
        sources[name] = sha256_file(ROOT/name)
    audit_path = ROOT/'reports/idrid_data_v1/data_audit.json'
    if sha256_file(audit_path) != AUDIT_SHA:
        raise ValueError('Original frozen IDRiD data audit changed')
    audit = json.loads(audit_path.read_text())
    confirmation = idrid_source_confirmation(audit_path)
    if confirmation.get('source_provenance_status') != 'official_download_user_confirmed':
        raise ValueError('A valid bound user source-confirmation receipt is required')
    original = ROOT/'data/manifests/idrid_v1'
    preserved = json.loads((ROOT/'artifacts/idrid_data_v1/pre_execution_hashes.json').read_text())
    preserved['reports/idrid_data_v1/data_audit.json'] = AUDIT_SHA
    for name, digest in audit['private_output_sha256'].items():
        path = original/name
        if sha256_file(path) != digest:
            raise ValueError('Frozen private IDRiD audit inputs changed')
        preserved[str(path.relative_to(ROOT)).replace('\\', '/')] = digest
    preserved['reports/idrid_data_v1/source_confirmation.json'] = sha256_file(audit_path.parent/'source_confirmation.json')
    for name, digest in preserved.items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError('Preserved original input or research artifact changed')
    frame = pd.read_csv(original/'external_manifest.csv', dtype={'image_id': str, 'sha256': str, 'phash': str})
    internal = pd.read_csv(original/'internal_candidates.csv')
    cross = pd.read_csv(original/'cross_source_candidates.csv')
    if len(frame) != 516 or len(internal) != 8 or len(cross) != 3:
        raise ValueError('The frozen candidate/cohort counts differ from the declared review')
    config = json.loads((ROOT/'configs/idrid_data_v1.json').read_text())
    references, bindings = load_references(config)
    if bindings != audit['reference_manifest_bindings']:
        raise ValueError('Review reference pool differs from the frozen data audit')
    reference_rows = references.set_index(['dataset_source', 'image_id'])
    external_rows = frame.set_index('image_id')
    idrid_root, aptos_root = default_idrid_root(), default_aptos_root()

    def external(identity):
        row = external_rows.loc[identity]
        path = Path(row.image_path).resolve()
        if not path.is_relative_to(idrid_root):
            raise ValueError('Bound IDRiD path is outside the configured source')
        return path, row.sha256, int(row.dr_grade), int(row.dme_risk)

    pairs = []
    for row in internal.itertuples():
        first, second = external(row.first_image_id), external(row.second_image_id)
        proof = compare_images(first[0], second[0], first[1], second[1])
        pairs.append({'family': 'internal', 'first_image_id': row.first_image_id,
            'second_dataset': 'IDRiD', 'second_image_id': row.second_image_id,
            'dr_labels_disagree': first[2] != second[2], 'dme_labels_disagree': first[3] != second[3], **proof})
    for row in cross.itertuples():
        if row.reference_dataset != 'APTOS_2019':
            raise ValueError('Unexpected cross-source candidate family')
        first = external(row.external_image_id)
        reference = reference_rows.loc[(row.reference_dataset, row.reference_image_id)]
        path = (aptos_root/'train_images'/f'{row.reference_image_id}.png').resolve()
        if not path.is_relative_to(aptos_root):
            raise ValueError('APTOS pair path outside configured source')
        proof = compare_images(first[0], path, first[1], reference.sha256)
        pairs.append({'family': 'cross_source', 'first_image_id': row.external_image_id,
            'second_dataset': row.reference_dataset, 'second_image_id': row.reference_image_id,
            'dr_labels_disagree': None, 'dme_labels_disagree': None, **proof})
    population, summary = freeze_populations(frame, cross)
    evidence = {}
    for family in ('internal', 'cross_source'):
        part = [row for row in pairs if row['family'] == family]
        evidence[family] = {'pairs': len(part),
            'evidence_categories': dict(Counter(row['evidence_category'] for row in part)),
            'decoded_rgb_exact_pairs': sum(row['decoded_rgb_exact'] for row in part),
            'canonical_224_exact_pairs': sum(row['canonical_224_rgb_exact'] for row in part)}
    evidence['internal']['dr_disagreeing_pairs'] = sum(row['dr_labels_disagree'] for row in pairs if row['family'] == 'internal')
    evidence['internal']['dme_disagreeing_pairs'] = sum(row['dme_labels_disagree'] for row in pairs if row['family'] == 'internal')
    if any(sha256_file(ROOT/name) != digest for name, digest in preserved.items()):
        raise ValueError('Original artifact changed during review')
    private.mkdir(parents=True)
    population.to_csv(private/'external_population.csv', index=False, lineterminator='\n')
    pd.DataFrame(pairs).to_csv(private/'pair_evidence.csv', index=False, lineterminator='\n')
    report = {'schema_version': 1, 'experiment_id': 'idrid_review_v1', 'dataset': 'IDRiD',
        'partition': 'external_holdout', 'scope': 'external_overlap_review_and_population_freeze',
        'images': len(frame), 'summary': {**summary, 'pixel_evidence': evidence},
        'source_provenance_status': confirmation['source_provenance_status'],
        'source_confirmation': confirmation['source_confirmation'],
        'source_commit': revision, 'execution_source_sha256': sources,
        'input_bindings': {'data_audit_sha256': AUDIT_SHA,
            'private_audit_outputs': audit['private_output_sha256'], 'reference_manifests': bindings,
            'source_confirmation_sha256': preserved['reports/idrid_data_v1/source_confirmation.json']},
        'private_output_sha256': {name: sha256_file(private/name) for name in ('external_population.csv', 'pair_evidence.csv')},
        'original_artifacts_preserved': len(preserved), 'model_inference_performed': False,
        'external_model_evaluation_completed': False, 'elapsed_seconds': time.perf_counter()-begin,
        'limitations': ['Pixel-identical evidence establishes same displayed image content, not patient identity or the correct clinical grade.',
            'Canonical pixel statistics are descriptive and never establish original-pixel identity or inclusion decisions.',
            'All original cross-source candidates are conservatively flagged regardless of pixel similarity or labels.',
            'No source label is changed; retained representatives of label-conflicting components remain ambiguous.',
            'Populations are sensitivity views under a fixed rule, not independent-patient or complete decontamination guarantees.',
            'Source origin is user-confirmed; official archive checksum authentication has not been performed.',
            'No classifier, temperature, conformal threshold, policy fitting or external performance was computed.']}
    serialized = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if any(token in serialized for token in ('"image_id"', '"image_path"', str(ROOT), 'C:\\Users\\')):
        raise ValueError('Private identities or paths in review aggregates')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(serialized)
    print(json.dumps({'status': 'data_review_completed_no_external_predictions', 'summary': report['summary']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    run(args.source_commit)
