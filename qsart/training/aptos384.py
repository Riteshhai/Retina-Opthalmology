"""Declared 384px development training with bounded input memory and microbatches.

This research path uses direct RGB resizing. Its checkpoints are explicitly
ineligible for the crop-based production inference engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import psutil
import torch
from torch.nn import functional as F

from qsart.models.checkpoint import FORMAT_VERSION, sha256_file
from qsart.training.data import MEAN, STD
from qsart.training.finetune import batchnorm_buffer_hash, parameter_hash
from qsart.training.transfer import _head_metrics

PREPROCESSING = "direct_rgb_bilinear_square_no_crop_v1"
STRATEGY = "imagenet_full_parameter_active_bn_microbatch_v1"


@dataclass(frozen=True)
class Config:
    image_size: int = 384
    epochs: int = 15
    patience: int = 4
    microbatch_size: int = 4
    optimizer_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 2026
    cpu_threads: int = 4
    max_training_seconds: int = 10800
    minimum_available_bytes: int = 268435456
    maximum_process_rss_bytes: int = 4294967296

    def validate(self):
        for name in (
            "image_size",
            "epochs",
            "patience",
            "microbatch_size",
            "optimizer_batch_size",
            "cpu_threads",
            "max_training_seconds",
            "minimum_available_bytes",
            "maximum_process_rss_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer training/resource settings required")
        if self.image_size < 32 or self.optimizer_batch_size % self.microbatch_size:
            raise ValueError("Size >=32 and integral optimizer/microbatch ratio required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("Supported nonnegative training seed required")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (self.learning_rate, self.weight_decay)
        ):
            raise ValueError("Finite optimizer settings required")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Positive learning rate and nonnegative weight decay required")


def validate_frame(frame, scope):
    allowed = {"fit_selection": {"train", "validation"}, "posthoc": {"validation", "calibration"}}
    if (
        scope not in allowed
        or frame.empty
        or not {"image_path", "split", "dr_grade", "sha256"}.issubset(frame)
    ):
        raise ValueError("Explicit development scope and complete image/label/hash records required")
    if not frame.split.isin(allowed[scope]).all():
        raise ValueError("Forbidden partition; no reserved-test, confirmation or external images")
    if scope == "fit_selection" and set(frame.split) != {"train", "validation"}:
        raise ValueError("Fitting cache requires both fitting and selection roles")
    labels = pd.to_numeric(frame.dr_grade, errors="coerce")
    if labels.isna().any() or not labels.isin(range(5)).all():
        raise ValueError("Complete integer DR grades 0..4 required")
    if "official_split" in frame and not frame.official_split.eq("official_train").all():
        raise ValueError("Only verified official-training APTOS source accepted")
    if "eligible_for_training" in frame and not frame.eligible_for_training.astype(str).eq("True").all():
        raise ValueError("Training-ineligible image record")
    if not frame.index.is_unique or frame.image_path.astype(str).str.casefold().duplicated().any():
        raise ValueError("Unique row indices and image paths required")
    for name in ("split_group_id", "sha256"):
        if name in frame and (frame[name].isna().any() or frame.groupby(name).split.nunique().gt(1).any()):
            raise ValueError("Missing or cross-partition fingerprint component")


def direct_rgb(path, size):
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def prepare_store(frame, manifest, folder, *, size=384, scope="fit_selection"):
    """Store exact uint8 RGB bytes on disk; never allocate a cohort-sized tensor."""
    validate_frame(frame, scope)
    if type(size) is not int or size < 32:
        raise ValueError("Image size >=32 required")
    content = hashlib.sha256()
    for row in frame.itertuples():
        actual = sha256_file(row.image_path)
        if actual != row.sha256:
            raise ValueError("Original image SHA256 changed")
        content.update(f"{row.Index}:{row.split}:{actual}\n".encode())
    metadata = {
        "schema": "qsart.disk_rgb.v1",
        "scope": scope,
        "image_size": size,
        "preprocessing": PREPROCESSING,
        "manifest_sha256": sha256_file(manifest),
        "input_content_sha256": content.hexdigest(),
        "manifest_rows": frame.index.astype(int).tolist(),
        "partitions": frame.split.tolist(),
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    entry = folder / key
    path, info_path = entry / "pixels.rgb", entry / "metadata.json"
    started = time.perf_counter()
    reused = entry.exists()
    if reused:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info["metadata"] != metadata or sha256_file(path) != info["file_sha256"]:
            raise ValueError("Disk RGB cache provenance/checksum mismatch")
    else:
        temporary = folder / f"{key}.building.{uuid.uuid4().hex}"
        temporary.mkdir()
        staged_pixels = temporary / path.name
        with staged_pixels.open("xb") as stream:
            for index, row in enumerate(frame.itertuples()):
                stream.write(direct_rgb(row.image_path, size).tobytes())
                if (index + 1) % 200 == 0:
                    print(
                        json.dumps({"event": "direct_rgb_store", "images": index + 1, "total": len(frame)}),
                        flush=True,
                    )
        info = {"metadata": metadata, "file_sha256": sha256_file(staged_pixels)}
        with (temporary / info_path.name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(info, indent=2) + "\n")
        temporary.rename(entry)
    if path.stat().st_size != len(frame) * size * size * 3:
        raise ValueError("Disk RGB cache length differs from exact shape")
    return ImageStore(path, metadata), {
        **info,
        "reused": reused,
        "preparation_or_verification_seconds": time.perf_counter() - started,
        "bytes": path.stat().st_size,
    }


class ImageStore:
    def __init__(self, path, metadata):
        self.path, self.metadata = Path(path), metadata
        self.size = metadata["image_size"]
        self.rows = metadata["manifest_rows"]

    def image(self, position):
        if not 0 <= int(position) < len(self.rows):
            raise ValueError("Pixel position outside the declared cache")
        count = self.size * self.size * 3
        with self.path.open("rb") as stream:
            stream.seek(int(position) * count)
            pixels = stream.read(count)
        if len(pixels) != count:
            raise ValueError("Truncated cached image bytes")
        return Image.frombytes("RGB", (self.size, self.size), pixels)


def augment(image, generator):
    """Independent H/V flips, nearest ±15° rotation, shuffled .1 color jitter."""
    if torch.rand((), generator=generator) < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if torch.rand((), generator=generator) < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    angle = 30 * float(torch.rand((), generator=generator)) - 15
    image = image.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))
    factors = 0.9 + 0.2 * torch.rand(3, generator=generator)
    kinds = (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color)
    for index in torch.randperm(3, generator=generator).tolist():
        image = kinds[index](image).enhance(float(factors[index]))
    return image


def batch(store, positions, generator=None):
    images = []
    for position in positions:
        image = store.image(position)
        if generator is not None:
            image = augment(image, generator)
        images.append(torch.from_numpy(np.array(image, dtype=np.float32, copy=True) / 255).permute(2, 0, 1))
    values = torch.stack(images)
    return (
        (values - torch.tensor(MEAN)[None, :, None, None]) / torch.tensor(STD)[None, :, None, None]
    ).contiguous(memory_format=torch.channels_last)


def evaluate(model, store, positions, targets, microbatch_size):
    model.eval()
    logits = []
    with torch.inference_mode():
        for indices in torch.as_tensor(positions).split(microbatch_size):
            logits.append(model(batch(store, indices))["dr_grade"].detach().cpu())
    values = torch.cat(logits)
    return values, _head_metrics(values, torch.as_tensor(targets, dtype=torch.long), 5)


def weighted_microbatch_backward(model, images, targets, class_weights, microbatch_size):
    """One weighted-CE optimizer window: divide every sum by the WHOLE weight sum.

    Model BatchNorm and stochastic layers still observe individual microbatches;
    only the weighted loss normalization matches the concatenated logical batch.
    """
    denominator = class_weights[targets].sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("Positive finite optimizer-window weight sum required")
    numerator = 0.0
    for start in range(0, len(targets), microbatch_size):
        labels = targets[start : start + microbatch_size]
        loss = F.cross_entropy(
            model(images[start : start + microbatch_size])["dr_grade"],
            labels,
            weight=class_weights,
            reduction="sum",
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite fitting loss")
        (loss / denominator).backward()
        numerator += float(loss.detach())
    return numerator, float(denominator)


def _save_checkpoint(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    if path.exists():
        raise ValueError("Preserve existing epoch checkpoint")
    torch.save(value, temporary)
    os.link(temporary, path)
    temporary.unlink()


def _link_or_verify(source, target):
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise ValueError("Preserve differing selected checkpoint")
    else:
        os.link(source, target)


def fit(model, store, frame, folder, config, *, provenance, resume=False):
    """Immutable epoch checkpoints restore model/optimizer/scheduler and all RNGs."""
    config.validate()
    validate_frame(frame, "fit_selection")
    if (
        store.metadata["scope"] != "fit_selection"
        or store.metadata["manifest_rows"] != frame.index.astype(int).tolist()
        or store.metadata["partitions"] != frame.split.tolist()
        or store.size != config.image_size
    ):
        raise ValueError("Disk RGB store/manifest/config order differs")
    folder = Path(folder)
    if folder.exists() and not resume:
        raise ValueError("Output exists; preserve it or explicitly resume")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "training_result.json").exists():
        raise ValueError("Preserve completed training output")
    torch.set_num_threads(config.cpu_threads)
    model.requires_grad_(True)
    model.to(memory_format=torch.channels_last)
    initial_parameters = parameter_hash(model.backbone)
    initial_bn = batchnorm_buffer_hash(model)
    train_positions = np.flatnonzero(frame.split.eq("train").to_numpy())
    selection_positions = np.flatnonzero(frame.split.eq("validation").to_numpy())
    targets = torch.tensor(frame.dr_grade.to_numpy(int), dtype=torch.long)
    counts = torch.bincount(targets[train_positions], minlength=5)
    if (counts == 0).any():
        raise ValueError("All five fitting grades required")
    weights = len(train_positions) / (5 * counts.float())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    shuffle_rng = torch.Generator().manual_seed(config.seed + 2000003)
    augment_rng = torch.Generator().manual_seed(config.seed + 1000003)
    torch.manual_seed(config.seed)
    binding = {
        "config": asdict(config),
        "provenance": provenance,
        "cache_metadata": store.metadata,
        "initial_backbone_sha256": initial_parameters,
        "initial_batchnorm_sha256": initial_bn,
    }
    history, stale, best_accuracy, best_epoch, steps, elapsed_before = [], 0, -1.0, 0, 0, 0.0
    previous_hash = None
    epoch_files = sorted(folder.glob("epoch_*.pt"))
    peak_rss = 0
    if resume and epoch_files:
        for number, path in enumerate(epoch_files, 1):
            old = torch.load(path, map_location="cpu", weights_only=True)
            if (
                path.name != f"epoch_{number:03}.pt"
                or old["binding"] != binding
                or old["previous_epoch_sha256"] != previous_hash
                or old["epoch"] != number - 1
            ):
                raise ValueError("Resume configuration/source/epoch chain differs")
            score_path = folder / f"selection_epoch_{number:03}.csv"
            if not score_path.is_file() or sha256_file(score_path) != old["selection_scores_sha256"]:
                raise ValueError("Resume epoch selection-score witness differs")
            previous_hash = sha256_file(path)
        model.load_state_dict(old["model_state"])
        optimizer.load_state_dict(old["optimizer_state"])
        scheduler.load_state_dict(old["scheduler_state"])
        shuffle_rng.set_state(old["shuffle_rng"])
        augment_rng.set_state(old["augment_rng"])
        torch.set_rng_state(old["torch_rng"])
        history, stale, best_accuracy, best_epoch, steps, elapsed_before = (
            old["history"],
            old["stale"],
            old["best_accuracy"],
            old["best_epoch"],
            old["metadata"]["optimizer_steps"],
            old["training_elapsed_seconds"],
        )
        peak_rss = old["peak_process_rss_bytes"]
    started = time.perf_counter()
    process = psutil.Process()
    complete, stop_reason = False, None
    for epoch in range(len(history), config.epochs):
        if stale >= config.patience:
            complete, stop_reason = True, "patience"
            break
        if elapsed_before + time.perf_counter() - started >= config.max_training_seconds:
            stop_reason = "training_time_budget"
            break
        model.train()
        epoch_started = time.perf_counter()
        numerator = denominator = 0.0
        permutation = train_positions[torch.randperm(len(train_positions), generator=shuffle_rng).numpy()]
        rate = optimizer.param_groups[0]["lr"]
        for offset in range(0, len(permutation), config.optimizer_batch_size):
            indices = permutation[offset : offset + config.optimizer_batch_size]
            # Only an eight-image window is materialized; activations use four.
            images = batch(store, indices, augment_rng)
            optimizer.zero_grad(set_to_none=True)
            num, den = weighted_microbatch_backward(
                model, images, targets[indices], weights, config.microbatch_size
            )
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("Nonfinite fitting gradient; preserve partial output")
            optimizer.step()
            steps += 1
            numerator += num
            denominator += den
            memory = process.memory_info()
            peak_rss = max(peak_rss, memory.rss, getattr(memory, "peak_wset", 0))
            if (
                memory.rss > config.maximum_process_rss_bytes
                or psutil.virtual_memory().available < config.minimum_available_bytes
            ):
                raise RuntimeError(
                    "Resource guard interrupted an incomplete epoch; previous immutable epochs preserved"
                )
            if (offset // config.optimizer_batch_size + 1) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "epoch": epoch + 1,
                            "fitting_images": min(offset + len(indices), len(permutation)),
                            "total_fitting_images": len(permutation),
                            "optimizer_steps": steps,
                            "epoch_elapsed_seconds": time.perf_counter() - epoch_started,
                            "process_rss_bytes": memory.rss,
                        }
                    ),
                    flush=True,
                )
        logits, metrics = evaluate(
            model, store, selection_positions, targets[selection_positions], config.microbatch_size
        )
        accuracy = metrics["accuracy"]
        improved = accuracy > best_accuracy
        if improved:
            best_accuracy, best_epoch, stale = accuracy, epoch + 1, 0
        else:
            stale += 1
        row = {
            "epoch": epoch + 1,
            "train_weighted_nll": numerator / denominator,
            "validation_accuracy": accuracy,
            "validation_macro_f1": metrics["macro_f1"],
            "validation_nll": metrics["nll"],
            "learning_rate": rate,
            "optimizer_steps": steps,
            "selected": improved,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        scheduler.step()
        changed = parameter_hash(model.backbone) != initial_parameters
        bn_changed = batchnorm_buffer_hash(model) != initial_bn
        if not changed or not bn_changed:
            raise ValueError("Expected parameter/active BatchNorm changes were not observed")
        metadata = {
            **provenance,
            "trained": True,
            "data_purpose": provenance.get("data_purpose", "real_research"),
            "backbone": "efficientnet_b0",
            "heads": {"dr_grade": 5},
            "image_size": config.image_size,
            "normalization_mean": MEAN,
            "normalization_std": STD,
            "task": "dr",
            "seed": config.seed,
            "model_version": f"aptos-{config.image_size}-v1-seed{config.seed}-epoch{epoch + 1}",
            "selected_epoch": epoch + 1,
            "optimizer_steps": steps,
            "training_config": asdict(config),
            "training_strategy": STRATEGY,
            "preprocessing": PREPROCESSING,
            "production_eligible": False,
            "backbone_parameters_changed": changed,
            "batchnorm_buffers_changed": bn_changed,
            "selection_metric": "strict_validation_accuracy_earliest_tie",
            "initial_backbone_parameters_sha256": initial_parameters,
            "backbone_parameters_sha256": parameter_hash(model.backbone),
            "initial_batchnorm_buffers_sha256": initial_bn,
            "batchnorm_buffers_sha256": batchnorm_buffer_hash(model),
        }
        records = pd.DataFrame(
            {
                "manifest_row": frame.index[selection_positions],
                "true_dr_grade": targets[selection_positions].numpy(),
            }
        )
        for grade in range(5):
            records[f"logit_{grade}"] = logits[:, grade].double().numpy()
        score_path = folder / f"selection_epoch_{epoch + 1:03}.csv"
        score_bytes = records.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
        if score_path.exists():
            if score_path.read_bytes() != score_bytes:
                raise ValueError("Preserve differing orphan epoch scores")
        else:
            with score_path.open("xb") as stream:
                stream.write(score_bytes)
        payload = {
            "format_version": FORMAT_VERSION,
            "metadata": metadata,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "shuffle_rng": shuffle_rng.get_state(),
            "augment_rng": augment_rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "binding": binding,
            "history": history,
            "stale": stale,
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "previous_epoch_sha256": previous_hash,
            "training_elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "selection_scores_sha256": sha256_file(score_path),
            "peak_process_rss_bytes": peak_rss,
        }
        epoch_path = folder / f"epoch_{epoch + 1:03}.pt"
        _save_checkpoint(epoch_path, payload)
        previous_hash = sha256_file(epoch_path)
        print(json.dumps({"event": "epoch_completed", **row, "best_epoch": best_epoch}), flush=True)
    else:
        complete, stop_reason = True, "maximum_epochs"
    if stale >= config.patience:
        complete, stop_reason = True, "patience"
    result = {
        "status": "completed" if complete else "resource_budget_incomplete",
        "stop_reason": stop_reason,
        "selected_epoch": best_epoch,
        "epochs_executed": len(history),
        "optimizer_steps": steps,
        "history": history,
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "sampled_process_and_windows_peak_rss_bytes": peak_rss,
        "memory_scope": "optimizer-step RSS samples and Windows process peak where available",
        "full_protocol_completed": complete,
        "convergence_established": False,
        "seed_variance_estimated": False,
        "epoch_checkpoint_sha256": {p.name: sha256_file(p) for p in sorted(folder.glob("epoch_*.pt"))},
    }
    if complete:
        if best_epoch < 1:
            raise ValueError("No selected supervised checkpoint")
        _link_or_verify(folder / f"epoch_{best_epoch:03}.pt", folder / "best.pt")
        _link_or_verify(folder / f"epoch_{len(history):03}.pt", folder / "last.pt")
        with (folder / "training_result.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
