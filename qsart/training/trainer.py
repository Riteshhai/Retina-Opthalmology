"""Real optimization with validation selection; synthetic inputs are test-only."""
from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from qsart.models import build_model
from qsart.models.checkpoint import FORMAT_VERSION, load_checkpoint, sha256_file
from .data import MEAN, STD, ManifestDataset, read_manifest, split_fingerprint


@dataclass
class TrainingConfig:
    backbone: str = "mobilenet_v3_small"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 7
    seed: int = 42
    cpu_threads: int = 2
    device: str = "cpu"
    pretrained: bool = False
    class_weights: bool = True
    augment: bool = True
    mixed_precision: bool = False

    def validate(self):
        if min(self.image_size, self.batch_size, self.epochs, self.patience, self.cpu_threads) <= 0:
            raise ValueError("Image/batch/epoch/patience/thread settings must be positive")
        if self.image_size < 32 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid image size or optimizer settings")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if self.mixed_precision and self.device != "cuda":
            raise ValueError("Mixed precision is supported only on CUDA")


def set_seed(seed: int, threads: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def _code_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def train(manifest: str | Path, output: str | Path, *, task: str = "dr",
          config: TrainingConfig | None = None, quality_heads: dict[str, int] | None = None,
          poor_class_indices: list[int] | None = None, resume: str | Path | None = None,
          data_purpose: str = "real_research") -> dict:
    config = config or TrainingConfig()
    config.validate()
    if task not in {"dr", "quality"} or data_purpose not in {"real_research", "software_test"}:
        raise ValueError("Unsupported task or data purpose")
    set_seed(config.seed, config.cpu_threads)
    manifest, output = Path(manifest), Path(output)
    frame = read_manifest(manifest)
    heads = {"dr_grade": 5} if task == "dr" else dict(quality_heads or {"quality_grade": 2})
    if task == "quality" and "quality_grade" not in heads:
        raise ValueError("Quality training requires an overall quality_grade head")
    poor_class_indices = [0] if poor_class_indices is None else poor_class_indices
    if task == "quality" and (not poor_class_indices or any(
            item < 0 or item >= heads["quality_grade"] for item in poor_class_indices)):
        raise ValueError("Specify valid poor quality class indices")
    train_frame = frame.loc[frame.split == "train"]
    validation_frame = frame.loc[frame.split == "validation"]
    datasets = {"train": ManifestDataset(train_frame, heads, config.image_size, config.augment),
                "validation": ManifestDataset(validation_frame, heads, config.image_size)}
    # A missing head in either partition cannot be selected or validated honestly.
    for split, dataset in datasets.items():
        for head in heads:
            if dataset.frame[head].notna().sum() == 0:
                raise ValueError(f"No {head} labels in {split}")
    # BatchNorm cannot learn from a 1-item final batch at small spatial resolutions.
    if len(datasets["train"]) < 2 or config.batch_size < 2:
        raise ValueError("Training requires at least two labeled images and batch_size >= 2")
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True,
                            num_workers=0, drop_last=len(datasets["train"]) % config.batch_size == 1,
                            generator=generator),
        "validation": DataLoader(datasets["validation"], batch_size=config.batch_size, num_workers=0),
    }
    model = build_model(config.backbone, heads, config.pretrained).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criteria = {}
    for head, classes in heads.items():
        counts = datasets["train"].frame[head].value_counts().reindex(range(classes), fill_value=0)
        # Classes absent from training retain unit weight for honest validation loss.
        weights = np.ones(classes, dtype=np.float32)
        positive = counts.to_numpy() > 0
        weights[positive] = counts.sum() / (positive.sum() * counts.to_numpy()[positive])
        criteria[head] = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, device=config.device) if config.class_weights else None)
    output.mkdir(parents=True, exist_ok=True)
    start_epoch, steps, best, stale = 0, 0, float("inf"), 0
    history = []
    manifest_hash = sha256_file(manifest)
    fingerprints = {name: split_fingerprint(part) for name, part in frame.groupby("split")}
    if resume:
        _, previous = load_checkpoint(resume, allow_software_test=data_purpose == "software_test")
        metadata = previous["metadata"]
        if metadata["manifest_sha256"] != manifest_hash or metadata["heads"] != heads:
            raise ValueError("Resume manifest or head definitions differ")
        if metadata["backbone"] != config.backbone or metadata["image_size"] != config.image_size:
            raise ValueError("Resume model configuration differs")
        if metadata["data_purpose"] != data_purpose or metadata["task"] != task:
            raise ValueError("Resume purpose or supervised task differs")
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        scheduler.load_state_dict(previous["scheduler_state"])
        start_epoch, steps = previous["epoch"] + 1, metadata["optimizer_steps"]
        best, stale = previous["best_validation_loss"], previous.get("stale_epochs", 0)
        history = previous.get("history", [])
        if "torch_rng_state" in previous:
            torch.set_rng_state(previous["torch_rng_state"])
            generator.set_state(previous["loader_rng_state"])
    amp = config.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if resume and "scaler_state" in previous:
        scaler.load_state_dict(previous["scaler_state"])
        if config.device == "cuda" and "cuda_rng_states" in previous:
            torch.cuda.set_rng_state_all(previous["cuda_rng_states"])
    for epoch in range(start_epoch, config.epochs):
        epoch_metrics = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"]}
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            total_loss, total_count = 0.0, 0
            correct = {head: 0 for head in heads}
            labeled = {head: 0 for head in heads}
            for images, labels in loader:
                images = images.to(config.device)
                labels = {name: values.to(config.device) for name, values in labels.items()}
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), torch.autocast(
                        device_type=config.device, enabled=amp):
                    logits = model(images)
                    losses = []
                    for head in heads:
                        mask = labels[head] >= 0
                        if mask.any():
                            losses.append(criteria[head](logits[head][mask], labels[head][mask]))
                            correct[head] += int((logits[head][mask].argmax(1) == labels[head][mask]).sum())
                            labeled[head] += int(mask.sum())
                    loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training/validation loss")
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    steps += 1
                total_loss += float(loss.detach()) * len(images)
                total_count += len(images)
            epoch_metrics[f"{split}_loss"] = total_loss / total_count
            for head in heads:
                epoch_metrics[f"{split}_{head}_accuracy"] = correct[head] / labeled[head]
        validation_loss = epoch_metrics["validation_loss"]
        scheduler.step(validation_loss)
        history.append(epoch_metrics)
        improved = validation_loss < best
        best, stale = (validation_loss, 0) if improved else (best, stale + 1)
        metadata = {
            "trained": True, "data_purpose": data_purpose, "optimizer_steps": steps,
            "backbone": config.backbone, "heads": heads, "task": task,
            "image_size": config.image_size, "normalization_mean": MEAN, "normalization_std": STD,
            "preprocessing": "fundus_crop_rgb_v1", "manifest_sha256": manifest_hash,
            "split_fingerprints": fingerprints, "split_counts": {
                name: len(data) for name, data in datasets.items()},
            "model_version": f"{task}-{config.backbone}-seed{config.seed}-epoch{epoch + 1}",
            "seed": config.seed, "code_version": _code_version(), "torch_version": str(torch.__version__),
            "poor_class_indices": poor_class_indices if task == "quality" else [],
            "training_config": asdict(config), "selection_metric": "validation_loss",
        }
        checkpoint = {
            "format_version": FORMAT_VERSION, "metadata": metadata,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch, "best_validation_loss": best, "stale_epochs": stale, "history": history,
            "torch_rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state(),
        }
        if config.device == "cuda":
            checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if stale >= config.patience:
            break
    if not history:
        raise ValueError("No epochs executed; increase epochs when resuming")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
        ax.plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="validation")
        ax.set(xlabel="Epoch", ylabel="Supervised cross-entropy", title=f"{task.upper()} training curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "training_curves.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    return {"best_checkpoint": str(output / "best.pt"), "epochs_executed": len(history),
            "optimizer_steps": steps, "best_validation_loss": best, "data_purpose": data_purpose}
