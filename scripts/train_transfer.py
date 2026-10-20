"""CPU transfer training on development data only; no confirmation predictions."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from qsart.data.development import defect_mapping_report
from qsart.training.transfer import TransferConfig, train_transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['dr', 'quality'], default='dr')
    parser.add_argument('--backbone', choices=['mobilenet_v3_small', 'resnet18'])
    parser.add_argument('--config', type=Path, default=Path('configs/development_v1.yaml'))
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/development_v1/features'))
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--quality-defects', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding='utf-8')) or {}
    if not isinstance(settings, dict) or set(settings)-{f.name for f in fields(TransferConfig)}:
        parser.error('Unknown or invalid transfer configuration fields')
    if args.backbone:
        settings['backbone'] = args.backbone
    heads, mappings = {'quality_grade': 2}, {}
    if args.quality_defects:
        mappings = defect_mapping_report()
        heads.update({name: mapping['classes'] for name, mapping in mappings.items()})
    result = train_transfer(args.manifest, args.output, args.cache_dir, task=args.task,
                            config=TransferConfig(**settings), quality_heads=heads,
                            label_mappings=mappings, allow_download=args.allow_download,
                            resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
