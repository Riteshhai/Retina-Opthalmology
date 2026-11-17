"""Temperature scaling fitted on a dedicated calibration partition."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from qsart.models.checkpoint import load_checkpoint, sha256_file
from qsart.training.data import ManifestDataset, read_manifest, split_fingerprint


def _validated(logits: torch.Tensor, labels: torch.Tensor):
    logits = torch.as_tensor(logits, dtype=torch.float64).detach().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if logits.ndim != 2 or len(logits) < 2 or labels.shape != (len(logits),):
        raise ValueError("Calibration requires at least two labeled logit rows")
    if not torch.isfinite(logits).all() or (labels < 0).any() or (labels >= logits.shape[1]).any():
        raise ValueError("Invalid logits or calibration labels")
    return logits, labels


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits, labels = _validated(logits, labels)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=75,
                                 line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(math.log(0.05), math.log(20.0)).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    candidate = float(log_temperature.detach().clamp(math.log(0.05), math.log(20.0)).exp())
    # Numerical optimization cannot make the fitted calibration loss worse than T=1.
    if F.cross_entropy(logits / candidate, labels) > F.cross_entropy(logits, labels):
        return 1.0
    return candidate


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor,
                        temperature: float = 1.0, bins: int = 15) -> dict:
    logits, labels = _validated(logits, labels)
    if not math.isfinite(temperature) or temperature <= 0 or bins < 2:
        raise ValueError("Positive finite temperature and at least two bins required")
    probabilities = (logits / temperature).softmax(1)
    confidence, predictions = probabilities.max(1)
    correct = predictions.eq(labels)
    reliability, ece = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0
        count = int(mask.sum())
        accuracy = float(correct[mask].double().mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        reliability.append({"lower": lower, "upper": upper, "count": count,
                            "accuracy": accuracy, "confidence": mean_confidence})
    one_hot = F.one_hot(labels, probabilities.shape[1]).to(probabilities)
    return {"n": len(labels), "nll": float(F.cross_entropy(logits / temperature, labels)),
            "ece": ece, "brier": float(((probabilities - one_hot) ** 2).sum(1).mean()),
            "reliability_bins": reliability}


def calibrate_checkpoint(checkpoint_path: str | Path, manifest: str | Path,
                         *, allow_software_test: bool = False, batch_size: int = 16,
                         cpu_threads: int = 2) -> dict:
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("Batch size and CPU threads must be positive")
    torch.set_num_threads(cpu_threads)
    model, checkpoint = load_checkpoint(checkpoint_path, allow_software_test=allow_software_test)
    metadata = checkpoint["metadata"]
    frame = read_manifest(manifest)
    if sha256_file(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Calibration must use the training manifest with a dedicated calibration split")
    calibration_frame = frame.loc[frame.split == "calibration"]
    if len(calibration_frame) < 2:
        raise ValueError("A dedicated calibration partition with at least two records is required")
    fingerprint = split_fingerprint(calibration_frame)
    if fingerprint in {metadata["split_fingerprints"].get("train"),
                       metadata["split_fingerprints"].get("validation")}:
        raise ValueError("Calibration overlaps model selection or training data")
    dataset = ManifestDataset(calibration_frame, metadata["heads"], metadata["image_size"],
                              mean=metadata["normalization_mean"], std=metadata["normalization_std"])
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    first = {head: [] for head in metadata["heads"]}
    flipped = {head: [] for head in metadata["heads"]}
    all_labels = {head: [] for head in metadata["heads"]}
    with torch.inference_mode():
        for images, labels in loader:
            base_output, flip_output = model(images), model(images.flip(-1))
            for head in first:
                mask = labels[head] >= 0
                first[head].append(base_output[head][mask])
                flipped[head].append(((base_output[head] + flip_output[head]) / 2)[mask])
                all_labels[head].append(labels[head][mask])
    heads = {}
    for head in first:
        logits, tta_logits, labels = (torch.cat(items) for items in
                                     (first[head], flipped[head], all_labels[head]))
        temperature = fit_temperature(logits, labels)
        tta_temperature = fit_temperature(tta_logits, labels)
        heads[head] = {"temperature": temperature, "flip_tta_temperature": tta_temperature,
                       "before": calibration_metrics(logits, labels),
                       "after": calibration_metrics(logits, labels, temperature),
                       "flip_tta_after": calibration_metrics(tta_logits, labels, tta_temperature)}
    return {"checkpoint_sha256": sha256_file(checkpoint_path), "model_version": metadata["model_version"],
            "manifest_sha256": metadata["manifest_sha256"], "split": "calibration",
            "split_fingerprint": fingerprint, "data_purpose": metadata["data_purpose"], "heads": heads}


def write_calibration(dr_checkpoint: str | Path, quality_checkpoint: str | Path,
                      manifest: str | Path, output: str | Path, **kwargs) -> dict:
    dr = calibrate_checkpoint(dr_checkpoint, manifest, **kwargs)
    quality = calibrate_checkpoint(quality_checkpoint, manifest, **kwargs)
    if dr["data_purpose"] != quality["data_purpose"]:
        raise ValueError("DR and quality artifacts must have matching data purposes")
    artifact = {"format_version": 1, "method": "temperature_scaling", "dr": dr, "quality": quality,
                "data_purpose": dr["data_purpose"], "second_pass_strategy": "horizontal_flip_mean_logits",
                "note": "Metrics describe calibration fitting data, not independent test performance."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    # These diagrams are calibration-fit diagnostics, not held-out performance claims.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for task in ("dr", "quality"):
            for head, result in artifact[task]["heads"].items():
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot([0, 1], [0, 1], "--", color="gray")
                for label, color in (("before", "tab:red"), ("after", "tab:blue")):
                    valid = [item for item in result[label]["reliability_bins"] if item["count"]]
                    ax.plot([item["confidence"] for item in valid],
                            [item["accuracy"] for item in valid], "o-", label=label, color=color)
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy",
                       title=f"Calibration-fit reliability: {task}/{head}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output.with_name(f"{output.stem}_{task}_{head}_reliability.png"), dpi=180)
                plt.close(fig)
    except ImportError:
        pass
    return artifact
