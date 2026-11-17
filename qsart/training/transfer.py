"""Frozen ImageNet transfer learning with provenance-checked CPU feature caches.

No final confirmation/test data are accepted by this development trainer.
The exported state dictionaries remain compatible with RetinalClassifier.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from qsart.models import build_model
from qsart.models.checkpoint import FORMAT_VERSION, load_checkpoint, sha256_file
from qsart.training.data import MEAN, STD, image_to_tensor, read_manifest, split_fingerprint
from qsart.training.trainer import set_seed

CACHE_VERSION = 1
HEAD_RNG_VERSION = "torch_seed_before_linear_heads_v1"


@dataclass
class TransferConfig:
    backbone: str = "mobilenet_v3_small"
    image_size: int = 224
    extraction_batch_size: int = 16
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    patience: int = 15
    seed: int = 2026
    cpu_threads: int = 2
    num_workers: int = 2
    pretrained: bool = True
    class_weights: bool = True
    horizontal_flip: bool = True

    def validate(self):
        if self.backbone not in {"mobilenet_v3_small", "resnet18", "efficientnet_b0"}:
            raise ValueError("Unsupported transfer backbone")
        if self.image_size < 32 or min(self.extraction_batch_size, self.batch_size, self.epochs,
                                       self.patience, self.cpu_threads) < 1:
            raise ValueError("Invalid transfer size, epochs, batch or thread settings")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid transfer loader or optimizer settings")
        if not all(math.isfinite(x) for x in (self.learning_rate, self.weight_decay)):
            raise ValueError("Transfer optimizer settings must be finite")


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    sources = ["qsart/training/transfer.py", "qsart/training/data.py", "qsart/training/trainer.py",
               "qsart/models/networks.py", "qsart/models/checkpoint.py", "qsart/preprocessing/fundus.py",
               "scripts/train_transfer.py"]
    hashes = {name: sha256_file(root / name) for name in sources if (root / name).is_file()}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                             stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "uncommitted", True
    return {"git_revision": revision, "working_tree_dirty": dirty,
            "source_file_sha256": hashes, "source_sha256": _canonical_hash(hashes)}


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def pretrained_spec(backbone: str) -> dict:
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        raise ValueError("Unsupported transfer backbone")
    filename = weights.url.rsplit("/", 1)[-1]
    return {"enum": str(weights), "url": weights.url, "filename": filename,
            "published_sha256_prefix": filename.rsplit("-", 1)[-1].split(".")[0],
            "code_license": "torchvision BSD-3-Clause; pretrained data/weight terms require separate review"}


def initialize_transfer(config: TransferConfig, cache_dir: str | Path, *, allow_download: bool = False,
                        data_purpose: str = "real_research") -> tuple[nn.Module, dict]:
    config.validate()
    if data_purpose not in {"real_research", "software_test"}:
        raise ValueError("Unsupported data purpose")
    if not config.pretrained and data_purpose != "software_test":
        raise ValueError("Real transfer runs require legitimate pretrained weights")
    torch.hub.set_dir(str(Path(cache_dir).resolve() / "torchhub"))
    information = {"pretrained": False, "data_purpose": data_purpose}
    if config.pretrained:
        information = {**pretrained_spec(config.backbone), "pretrained": True}
        weight_file = Path(torch.hub.get_dir()) / "checkpoints" / information["filename"]
        if not weight_file.is_file() and not allow_download:
            raise ValueError("Official pretrained weights missing; explicitly allow the one-time download")
        if weight_file.is_file() and not sha256_file(weight_file).startswith(information["published_sha256_prefix"]):
            raise ValueError("Pretrained weight file hash does not match the published URL prefix")
    model = build_model(config.backbone, pretrained=config.pretrained)
    if config.pretrained:
        information["file_sha256"] = sha256_file(weight_file)
        if not information["file_sha256"].startswith(information["published_sha256_prefix"]):
            raise ValueError("Downloaded pretrained weight checksum mismatch")
    model.backbone.requires_grad_(False)
    model.eval()  # In particular, freeze pretrained BatchNorm running statistics.
    information["backbone_state_sha256"] = _state_hash(model.backbone)
    return model, information


class _FeatureImages(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.paths = frame.image_path.tolist()
        self.train_flags = frame.split.eq("train").tolist()
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as handle:
            image = handle.convert("RGB")
        return image_to_tensor(image, self.image_size), index, self.train_flags[index]


def _worker_init(_):
    torch.set_num_threads(1)


def _input_content_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples():
        actual = sha256_file(row.image_path)
        expected = getattr(row, "sha256", None)
        if expected is not None and pd.notna(expected) and str(expected) != actual:
            raise ValueError("Input image changed since its verified manifest was produced")
        digest.update(f"{row.Index}:{row.split}:{actual}\n".encode())
    return digest.hexdigest()


def extract_features(model: nn.Module, frame: pd.DataFrame, manifest: str | Path,
                     config: TransferConfig, cache_dir: str | Path, pretrained: dict,
                     provenance: dict) -> tuple[dict, dict]:
    """Verify current image bytes before reading any reusable feature cache."""
    if not frame.split.isin(["train", "validation"]).all():
        raise ValueError("Feature cache accepts train and selection-validation records only")
    selected = frame.reset_index(drop=False).rename(columns={"index": "manifest_row"})
    content_hash = _input_content_hash(frame)
    metadata = {
        "format_version": CACHE_VERSION, "manifest_sha256": sha256_file(manifest),
        "input_content_sha256": content_hash, "source_sha256": provenance["source_sha256"],
        "backbone": config.backbone, "pretrained": pretrained,
        "transform": {"name": "fundus_crop_rgb_v1", "size": config.image_size,
                      "normalization_mean": MEAN, "normalization_std": STD,
                      "train_horizontal_flip": config.horizontal_flip},
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "split_fingerprints": {name: split_fingerprint(part) for name, part in frame.groupby("split")},
        "manifest_rows": selected.manifest_row.astype(int).tolist(),
    }
    cache_hash = _canonical_hash(metadata)
    cache_path = Path(cache_dir) / f"features_{config.backbone}_{cache_hash}.pt"
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("metadata") != metadata or cached.get("cache_key") != cache_hash:
            raise ValueError("Feature-cache provenance mismatch")
        for key in ("features", "flip_features"):
            values = cached.get(key)
            if not isinstance(values, torch.Tensor) or values.ndim != 2 or len(values) != len(frame):
                raise ValueError("Invalid feature-cache tensor shape")
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite cached features")
        if cached["features"].shape != cached["flip_features"].shape:
            raise ValueError("Feature and flip dimensions differ")
        print(json.dumps({"event": "feature_cache_reused", "backbone": config.backbone,
                          "records": len(frame), "cache_key": cache_hash}), flush=True)
        return cached, {"cache_key": cache_hash, "cache_file_sha256": sha256_file(cache_path),
                        "reused": True, "extraction_seconds": 0.0, "metadata": metadata}
    model.backbone.eval()
    before_state = _state_hash(model.backbone)
    loader = DataLoader(_FeatureImages(frame, config.image_size), batch_size=config.extraction_batch_size,
                        num_workers=config.num_workers, shuffle=False, worker_init_fn=_worker_init)
    features, flips = None, None
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (images, indices, train_flags) in enumerate(loader):
            values = model.backbone(images).detach().cpu()
            if features is None:
                features = torch.empty(len(frame), values.shape[1], dtype=torch.float32)
                flips = torch.zeros_like(features)
            features[indices] = values
            mask = train_flags.bool()
            if config.horizontal_flip and mask.any():
                flips[indices[mask]] = model.backbone(images[mask].flip(-1)).detach().cpu()
            if batch_number % 10 == 0:
                print(json.dumps({"event": "feature_extraction", "backbone": config.backbone,
                                  "processed": min((batch_number + 1) * config.extraction_batch_size, len(frame)),
                                  "total": len(frame)}), flush=True)
    if features is None or not torch.isfinite(features).all() or not torch.isfinite(flips).all():
        raise ValueError("Feature extraction yielded no finite features")
    if before_state != _state_hash(model.backbone):
        raise ValueError("Frozen backbone parameters or BatchNorm statistics changed")
    cached = {"metadata": metadata, "cache_key": cache_hash,
              "features": features, "flip_features": flips}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(cached, temporary)
    temporary.replace(cache_path)
    elapsed = time.perf_counter() - start
    return cached, {"cache_key": cache_hash, "cache_file_sha256": sha256_file(cache_path),
                    "reused": False, "extraction_seconds": elapsed, "metadata": metadata}


def _head_metrics(logits: torch.Tensor, targets: torch.Tensor, classes: int) -> dict:
    mask = targets >= 0
    if int(mask.sum()) == 0:
        raise ValueError("An output head has no selection-validation labels")
    logits, targets = logits[mask], targets[mask]
    predicted = logits.argmax(1)
    return {"n": len(targets), "accuracy": float(accuracy_score(targets, predicted)),
            "macro_f1": float(f1_score(targets, predicted, labels=list(range(classes)),
                                        average="macro", zero_division=0)),
            "nll": float(F.cross_entropy(logits, targets)),
            "confusion_matrix": confusion_matrix(targets, predicted, labels=list(range(classes))).tolist()}


def fit_cached_heads(model: nn.Module, cache: dict, frame: pd.DataFrame, output: str | Path,
                      config: TransferConfig, *, task: str, heads: dict[str, int],
                      base_metadata: dict, resume: str | Path | None = None) -> dict:
    """Only linear heads are optimized; original/flip features are fixed inputs."""
    primary = "dr_grade" if task == "dr" else "quality_grade"
    if primary not in heads or (task == "dr" and heads != {"dr_grade": 5}):
        raise ValueError("Invalid primary transfer task head")
    features = cache["features"]
    if len(features) != len(frame):
        raise ValueError("Feature rows and manifest rows differ")
    targets = {}
    for name, count in heads.items():
        if name not in frame:
            raise ValueError(f"Missing genuine mapped label column: {name}")
        values = pd.to_numeric(frame[name], errors="raise")
        present = values.dropna()
        if not ((present >= 0) & (present < count) & (present % 1 == 0)).all():
            raise ValueError(f"Invalid class range in {name}")
        targets[name] = torch.tensor(values.fillna(-100).to_numpy(), dtype=torch.long)
    train_mask = torch.tensor(frame.split.eq("train").to_numpy())
    validation_mask = torch.tensor(frame.split.eq("validation").to_numpy())
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Separate nonempty train and selection-validation partitions are required")
    train_features = features[train_mask]
    train_targets = {name: values[train_mask] for name, values in targets.items()}
    original_train_count = len(train_features)
    if config.horizontal_flip:
        train_features = torch.cat((train_features, cache["flip_features"][train_mask]))
        train_targets = {name: torch.cat((values, values)) for name, values in train_targets.items()}
    validation_features = features[validation_mask]
    validation_targets = {name: values[validation_mask] for name, values in targets.items()}
    # Feature extraction creates a DataLoader iterator, which advances Torch's
    # global RNG on a cache miss. Head initialization must not depend on that
    # incidental cache state. Resume restores its saved RNG after initialization.
    torch.manual_seed(config.seed)
    model.heads = nn.ModuleDict({name: nn.Linear(features.shape[1], count) for name, count in heads.items()})
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criteria = {}
    class_counts = {}
    for name, classes in heads.items():
        values = targets[name][train_mask]
        counts = torch.bincount(values[values >= 0], minlength=classes).float()
        if counts.sum() == 0:
            raise ValueError(f"No training labels for {name}")
        weights = torch.ones(classes)
        nonzero = counts > 0
        weights[nonzero] = counts.sum() / (nonzero.sum() * counts[nonzero])
        criteria[name] = nn.CrossEntropyLoss(weight=weights if config.class_weights else None)
        class_counts[name] = counts.int().tolist()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    history, steps, start_epoch, stale = [], 0, 0, 0
    best_score = (-math.inf, -math.inf)
    if resume:
        _, old = load_checkpoint(resume, allow_software_test=base_metadata["data_purpose"] == "software_test")
        previous = old["metadata"]
        for name in ("manifest_sha256", "feature_cache_key", "backbone", "heads", "data_purpose", "source_sha256"):
            expected = heads if name == "heads" else base_metadata[name]
            if previous.get(name) != expected:
                raise ValueError(f"Resume transfer provenance differs: {name}")
        old_config = previous["training_config"]
        if any(old_config.get(name) != value for name, value in asdict(config).items()
               if name not in {"epochs", "num_workers", "cpu_threads", "extraction_batch_size"}):
            raise ValueError("Resume optimizer/augmentation configuration differs")
        model.load_state_dict(old["model_state"])
        optimizer.load_state_dict(old["optimizer_state"])
        scheduler.load_state_dict(old["scheduler_state"])
        start_epoch, steps, stale = old["epoch"] + 1, previous["optimizer_steps"], old["stale_epochs"]
        history, best_score = old["history"], tuple(old["best_selection_score"])
        generator.set_state(old["loader_rng_state"])
        torch.set_rng_state(old["torch_rng_state"])
        old_best = Path(resume).parent / "best.pt"
        if not (output / "best.pt").is_file():
            if not old_best.is_file():
                raise ValueError("Resume needs its validation-selected sibling best.pt")
            shutil.copy2(old_best, output / "best.pt")
    backbone_before = _state_hash(model.backbone)
    for epoch in range(start_epoch, config.epochs):
        model.backbone.eval()
        model.heads.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        losses = []
        for indices in permutation.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            output_logits = {name: head(train_features[indices]) for name, head in model.heads.items()}
            batch_losses = []
            for name in heads:
                values = train_targets[name][indices]
                mask = values >= 0
                if mask.any():
                    batch_losses.append(criteria[name](output_logits[name][mask], values[mask]))
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite transfer head loss")
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        model.heads.eval()
        with torch.inference_mode():
            validation_logits = {name: head(validation_features) for name, head in model.heads.items()}
            metrics = {name: _head_metrics(validation_logits[name], validation_targets[name], count)
                       for name, count in heads.items()}
        selected = metrics[primary]
        score = (selected["macro_f1"], -selected["nll"])
        scheduler.step(selected["nll"])
        improved = score > best_score
        best_score, stale = (score, 0) if improved else (best_score, stale + 1)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "validation_macro_f1": selected["macro_f1"], "validation_nll": selected["nll"],
               "validation_accuracy": selected["accuracy"], "learning_rate": optimizer.param_groups[0]["lr"]}
        for name, values in metrics.items():
            for metric in ("accuracy", "macro_f1", "nll"):
                row[f"validation_{name}_{metric}"] = values[metric]
        history.append(row)
        metadata = {**base_metadata, "trained": True, "optimizer_steps": steps, "task": task,
                    "heads": heads, "image_size": config.image_size, "normalization_mean": MEAN,
                    "normalization_std": STD, "preprocessing": "fundus_crop_rgb_v1", "seed": config.seed,
                    "head_rng_version": HEAD_RNG_VERSION, "head_initialization_seed": config.seed,
                    "training_config": asdict(config), "training_strategy": "frozen_imagenet_cached_features",
                    "backbone_frozen": True, "class_counts": class_counts,
                    "original_train_images": original_train_count, "train_feature_views": len(train_features),
                    "selection_metric": "macro_f1_then_unweighted_nll", "selection_primary_head": primary,
                    "model_version": f"development-v1-{task}-{config.backbone}-seed{config.seed}-epoch{epoch + 1}",
                    "selection_metrics": metrics, "epoch": epoch + 1}
        artifact = {"format_version": FORMAT_VERSION, "metadata": metadata,
                    "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch, "stale_epochs": stale, "best_selection_score": list(best_score),
                    "history": history, "loader_rng_state": generator.get_state(),
                    "torch_rng_state": torch.get_rng_state()}
        torch.save(artifact, output / "last.pt")
        if improved:
            torch.save(artifact, output / "best.pt")
            (output / "selection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            predictions = pd.DataFrame({"manifest_row": frame.index[validation_mask.numpy()]})
            for name, logits in validation_logits.items():
                predictions[f"true_{name}"] = validation_targets[name].numpy()
                predictions[f"predicted_{name}"] = logits.argmax(1).numpy()
                for index, values in enumerate(logits.softmax(1).T):
                    predictions[f"{name}_p{index}"] = values.numpy()
            predictions.to_csv(output / "selection_predictions.csv", index=False)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(json.dumps({"event": "head_epoch", "task": task, "backbone": config.backbone, **row,
                          "selected": improved}), flush=True)
        if stale >= config.patience:
            break
    if _state_hash(model.backbone) != backbone_before:
        raise ValueError("Head training modified the frozen backbone")
    if not history or not (output / "best.pt").is_file():
        raise ValueError("No validation-selected trained transfer checkpoint was produced")
    _, selected_checkpoint = load_checkpoint(output / "best.pt",
                                              allow_software_test=base_metadata["data_purpose"] == "software_test")
    selected_metadata = selected_checkpoint["metadata"]
    result = {"best_checkpoint": str(output / "best.pt"), "epochs_executed": len(history),
              "total_optimizer_steps": steps, "selected_epoch": selected_checkpoint["epoch"] + 1,
              "selected_metrics": selected_metadata["selection_metrics"],
              "data_purpose": base_metadata["data_purpose"], "backbone_unchanged": True}
    (output / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_transfer(manifest: str | Path, output: str | Path, cache_dir: str | Path, *, task: str = "dr",
                    config: TransferConfig | None = None, quality_heads: dict[str, int] | None = None,
                    label_mappings: dict | None = None, poor_class_indices: list[int] | None = None,
                    allow_download: bool = False, resume: str | Path | None = None,
                    data_purpose: str = "real_research") -> dict:
    config = config or TransferConfig()
    config.validate()
    if task not in {"dr", "quality"}:
        raise ValueError("Unsupported transfer task")
    set_seed(config.seed, config.cpu_threads)
    frame = read_manifest(manifest)
    if not frame.split.isin(["train", "validation", "calibration"]).all():
        raise ValueError("Development transfer manifest must exclude all holdout/test/confirmation records")
    if "official_split" in frame and not frame.official_split.eq("official_train").all():
        raise ValueError("Development transfer accepts official-training source records only")
    selected = frame.loc[frame.split.isin(["train", "validation"])].copy()
    provenance = source_provenance()
    model, pretrained = initialize_transfer(config, cache_dir, allow_download=allow_download,
                                            data_purpose=data_purpose)
    features, cache_info = extract_features(model, selected, manifest, config, cache_dir, pretrained, provenance)
    heads = {"dr_grade": 5} if task == "dr" else dict(quality_heads or {"quality_grade": 2})
    poor_class_indices = [0] if poor_class_indices is None else poor_class_indices
    if task == "quality" and ("quality_grade" not in heads or not poor_class_indices or any(
            index < 0 or index >= heads["quality_grade"] for index in poor_class_indices)):
        raise ValueError("Quality transfer needs overall labels and a valid poor-class mapping")
    base_metadata = {
        "data_purpose": data_purpose, "backbone": config.backbone, "manifest_sha256": sha256_file(manifest),
        "split_fingerprints": {name: split_fingerprint(part) for name, part in frame.groupby("split")},
        "split_counts": {name: len(part) for name, part in frame.groupby("split")},
        "code_version": provenance["git_revision"], "code_provenance": provenance,
        "source_sha256": provenance["source_sha256"], "config_sha256": _canonical_hash(asdict(config)),
        "feature_cache_key": cache_info["cache_key"], "feature_cache_provenance": cache_info,
        "pretrained_provenance": pretrained, "poor_class_indices": poor_class_indices if task == "quality" else [],
        "label_mappings": label_mappings or {}, "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "confirmation_used": False,
    }
    return fit_cached_heads(model, features, selected, output, config, task=task, heads=heads,
                            base_metadata=base_metadata, resume=resume)
