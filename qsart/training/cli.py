"""Shared supervised-training command-line options."""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import yaml

from .trainer import TrainingConfig, train


def training_main(task: str):
    parser = argparse.ArgumentParser(description=f"Train supervised {task} model; never generates training images")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--backbone", choices=["resnet18", "mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--pretrained", action="store_true", help="Explicit one-time ImageNet weights download/cache")
    parser.add_argument("--resume")
    parser.add_argument("--quality-heads", default='{"quality_grade":2}',
                        help="JSON mapping of genuine manifest label columns to class counts")
    parser.add_argument("--poor-class-indices", default="0", help="Comma-separated poor overall-quality classes")
    parser.add_argument("--software-test-data", action="store_true", help="Mark generated fixtures; production inference rejects this")
    args = parser.parse_args()
    config_file = Path(args.config)
    settings = yaml.safe_load(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    allowed = {field.name for field in fields(TrainingConfig)}
    if set(settings or {}) - allowed:
        parser.error(f"Unknown training configuration keys: {set(settings) - allowed}")
    settings = settings or {}
    for name in ("backbone", "epochs", "image_size", "batch_size", "cpu_threads", "seed", "device"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    if args.pretrained:
        settings["pretrained"] = True
    result = train(args.manifest, args.output, task=task, config=TrainingConfig(**settings),
                   quality_heads=json.loads(args.quality_heads),
                   poor_class_indices=[int(item) for item in args.poor_class_indices.split(",")],
                   resume=args.resume, data_purpose="software_test" if args.software_test_data else "real_research")
    print(json.dumps(result, indent=2))
