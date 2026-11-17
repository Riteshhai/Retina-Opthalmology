"""Reopen manifest image paths and enforce readiness; exits nonzero on failure."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from qsart.data.manifests import inspect_image, sha256_file, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path, default=Path('data/processed/deepdrid/validation.json'))
    parser.add_argument('--allow-incomplete', action='store_true', help='Write audit without returning nonzero')
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, dtype={'dhash': str, 'phash': str, 'sha256': str})
    # Recompute state instead of trusting a stale readability/hash column.
    rows = [inspect_image(Path(value), 'found' if Path(value).is_file() else 'missing')
            for value in frame.image_path]
    for field in rows[0] if rows else []:
        frame[field] = [row[field] for row in rows]
    report = validate_manifest(frame, strict=False)
    report['manifest_sha256'] = sha256_file(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {'near_duplicate_candidates', 'exact_duplicate_groups'}}, indent=2))
    return 0 if report['ready_for_training'] or args.allow_incomplete else 2


if __name__ == '__main__':
    raise SystemExit(main())
