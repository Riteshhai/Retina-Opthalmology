"""Calibrated DR and quality inference with a population second-pass budget."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

from qsart.models.checkpoint import ArtifactError, load_checkpoint, sha256_file
from qsart.training.data import image_to_tensor


class InferenceEngine:
    def __init__(self, dr_checkpoint: str | Path, quality_checkpoint: str | Path,
                 calibration_path: str | Path, *, allow_software_test: bool = False,
                 cpu_threads: int = 2):
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
        self.cpu_threads = cpu_threads
        self.dr_model, dr = load_checkpoint(dr_checkpoint, allow_software_test=allow_software_test)
        self.quality_model, quality = load_checkpoint(quality_checkpoint,
                                                       allow_software_test=allow_software_test)
        self.dr_metadata, self.quality_metadata = dr["metadata"], quality["metadata"]
        self.dr_metadata["selected_epoch"] = int(dr["epoch"]) + 1
        self.quality_metadata["selected_epoch"] = int(quality["epoch"]) + 1
        if self.dr_metadata["task"] != "dr" or self.quality_metadata["task"] != "quality":
            raise ArtifactError("DR and quality checkpoint tasks do not match their inputs")
        if self.dr_metadata["heads"] != {"dr_grade": 5}:
            raise ArtifactError("DR checkpoint must contain a five-grade output")
        if "quality_grade" not in self.quality_metadata["heads"]:
            raise ArtifactError("Overall quality head is missing")
        if self.dr_metadata["data_purpose"] != self.quality_metadata["data_purpose"]:
            raise ArtifactError("DR and quality data purposes differ")
        poor_indices = self.quality_metadata.get("poor_class_indices", [])
        if not poor_indices or any(not isinstance(i, int) or i < 0 or
                                    i >= self.quality_metadata["heads"]["quality_grade"]
                                    for i in poor_indices):
            raise ArtifactError("Quality poor-class mapping is invalid")
        try:
            self.calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactError("Valid calibration artifact not found") from exc
        if self.calibration.get("format_version") != 1:
            raise ArtifactError("Unsupported calibration schema")
        if (self.calibration.get("method") != "temperature_scaling" or
                self.calibration.get("second_pass_strategy") != "horizontal_flip_mean_logits"):
            raise ArtifactError("Unsupported calibration or second-pass method")
        if self.calibration.get("data_purpose") != self.dr_metadata["data_purpose"]:
            raise ArtifactError("Calibration data purpose differs from checkpoint")
        for task, path, metadata in (("dr", dr_checkpoint, self.dr_metadata),
                                     ("quality", quality_checkpoint, self.quality_metadata)):
            component = self.calibration.get(task, {})
            if component.get("checkpoint_sha256") != sha256_file(path):
                raise ArtifactError(f"{task} calibration does not match checkpoint checksum")
            if component.get("split") != "calibration":
                raise ArtifactError("Temperature scaling must use a dedicated calibration split")
            if component.get("manifest_sha256") != metadata["manifest_sha256"]:
                raise ArtifactError("Calibration manifest differs from training provenance")
            if component.get("split_fingerprint") != metadata["split_fingerprints"].get("calibration"):
                raise ArtifactError("Calibration partition fingerprint differs from training provenance")
            for head in metadata["heads"]:
                for name in ("temperature", "flip_tta_temperature"):
                    value = component.get("heads", {}).get(head, {}).get(name)
                    if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                        raise ArtifactError(f"Missing or invalid {task}/{head}/{name}")
                for stage in ("before", "after"):
                    values = component["heads"][head].get(stage, {})
                    for metric in ("n", "nll", "ece", "brier"):
                        value = values.get(metric)
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                            raise ArtifactError(f"Invalid calibration-fit metric: {task}/{head}/{stage}/{metric}")
        self.model_version = self.dr_metadata["model_version"]

    @classmethod
    def load(cls, dr_checkpoint: str | Path, quality_checkpoint: str | Path,
             calibration_path: str | Path, **kwargs):
        return cls(dr_checkpoint, quality_checkpoint, calibration_path, **kwargs)

    def info(self) -> dict:
        return {"model_version": self.model_version, "backbone": self.dr_metadata["backbone"],
                "image_size": self.dr_metadata["image_size"], "device": "cpu",
                "cpu_threads": self.cpu_threads, "data_purpose": self.dr_metadata["data_purpose"],
                "quality_model_version": self.quality_metadata["model_version"],
                "quality_heads": self.quality_metadata["heads"],
                "quality_label_mappings": self.quality_metadata.get("label_mappings", {}),
                "training_strategy": self.dr_metadata.get("training_strategy", "end_to_end"),
                "source_sha256": self.dr_metadata.get("source_sha256"),
                "training_provenance": {
                    task: {"split_counts": metadata.get("split_counts", {}), "seed": metadata["seed"],
                           "optimizer_steps": metadata["optimizer_steps"],
                           "selected_epoch": metadata["selected_epoch"],
                           "selection_metric": metadata.get("selection_metric", "validation_loss"),
                           "configured_epochs": metadata.get("training_config", {}).get("epochs"),
                           "manifest_sha256": metadata["manifest_sha256"],
                           "code_version": metadata.get("code_version", "unknown")}
                    for task, metadata in (("dr", self.dr_metadata), ("quality", self.quality_metadata))},
                "calibration_fit_metrics": {
                    task: {head: {"temperature": values["temperature"],
                                  "before": {key: values["before"][key] for key in ("n", "nll", "ece", "brier")},
                                  "after": {key: values["after"][key] for key in ("n", "nll", "ece", "brier")}}
                           for head, values in self.calibration[task]["heads"].items()}
                    for task in ("dr", "quality")},
                "calibration_method": "temperature_scaling",
                "second_pass_strategy": "horizontal_flip_mean_logits",
                "limitations": ["Research prototype; not clinically validated.",
                                 "Calibration-fit metrics are not independent test performance.",
                                 "Checkpoint provenance markers do not establish model accuracy.",
                                 "Second-pass activation is limited per request population."],
                "research_only": True}

    @staticmethod
    def _tensor(image, metadata):
        return image_to_tensor(image, metadata["image_size"],
                               metadata["normalization_mean"], metadata["normalization_std"]).unsqueeze(0)

    def _probabilities(self, logits, task: str, *, second_pass: bool = False):
        temperature_key = "flip_tta_temperature" if second_pass else "temperature"
        return {head: (values / self.calibration[task]["heads"][head][temperature_key]).softmax(1)[0]
                for head, values in logits.items()}

    def predict_images(self, images: list[Image.Image], *, second_pass_budget: float = 0.0,
                       second_pass_policy: str = "budget", uncertainty_threshold: float = 0.7,
                       severity_threshold: float = 0.5, include_uncalibrated: bool = False) -> list[dict]:
        """Budget applies to this request population; floor(B*N) guarantees the cap.

        Single-image requests with B < 1 receive no extra pass. This is explicit
        population accounting, not a guarantee of a long-run online latency cap.
        """
        if not 0 <= second_pass_budget <= 1 or not math.isfinite(second_pass_budget):
            raise ValueError("second_pass_budget must lie in [0,1]")
        if second_pass_policy not in {"none", "all", "uncertainty", "severity", "budget"}:
            raise ValueError("Unknown second-pass policy")
        if not 0 <= uncertainty_threshold <= 1 or not 0 <= severity_threshold <= 1:
            raise ValueError("Trigger thresholds must lie in [0,1]")
        if any(not isinstance(image, Image.Image) for image in images):
            raise ValueError("predict_images requires PIL images")
        if not images:
            return []
        rows = []
        with torch.inference_mode():
            for image in images:
                start = time.perf_counter()
                dr_tensor = self._tensor(image, self.dr_metadata)
                shared_transform = all(self.dr_metadata[name] == self.quality_metadata[name]
                                       for name in ("image_size", "normalization_mean", "normalization_std"))
                quality_tensor = dr_tensor if shared_transform else self._tensor(image, self.quality_metadata)
                dr_logits, quality_logits = self.dr_model(dr_tensor), self.quality_model(quality_tensor)
                dr_probs = self._probabilities(dr_logits, "dr")["dr_grade"]
                entropy = min(1.0, max(0.0, float(-(dr_probs * dr_probs.clamp_min(1e-12).log()).sum() / math.log(5))))
                rows.append({"dr_tensor": dr_tensor, "quality_tensor": quality_tensor,
                             "dr_logits": dr_logits, "quality_logits": quality_logits,
                             "entropy": entropy, "risk": float(dr_probs[2:].sum()),
                             "elapsed": (time.perf_counter() - start) * 1000, "second": False})
            candidates = list(range(len(rows)))
            if second_pass_policy == "none":
                candidates = []
            elif second_pass_policy == "uncertainty":
                candidates = [i for i in candidates if rows[i]["entropy"] >= uncertainty_threshold]
            elif second_pass_policy == "severity":
                candidates = [i for i in candidates if rows[i]["risk"] >= severity_threshold]
            if second_pass_policy == "all":
                # This baseline intentionally consumes a full second-pass population budget.
                if second_pass_budget != 1.0:
                    raise ValueError("The all-images baseline requires second_pass_budget=1")
            candidates.sort(key=lambda i: (-(rows[i]["entropy"] + rows[i]["risk"]) / 2, i))
            limit = math.floor(len(images) * second_pass_budget + 1e-12)
            for index in candidates[:limit]:
                row = rows[index]
                start = time.perf_counter()
                dr_flip = self.dr_model(row["dr_tensor"].flip(-1))
                quality_flip = self.quality_model(row["quality_tensor"].flip(-1))
                row["dr_logits"] = {name: (value + dr_flip[name]) / 2 for name, value in row["dr_logits"].items()}
                row["quality_logits"] = {name: (value + quality_flip[name]) / 2 for name, value in row["quality_logits"].items()}
                row["second"] = True
                row["elapsed"] += (time.perf_counter() - start) * 1000
        output = []
        for row in rows:
            dr = self._probabilities(row["dr_logits"], "dr", second_pass=row["second"])["dr_grade"]
            quality = self._probabilities(row["quality_logits"], "quality", second_pass=row["second"])
            overall_quality = quality["quality_grade"]
            entropy = min(1.0, max(0.0, float(-(dr * dr.clamp_min(1e-12).log()).sum() / math.log(5))))
            output.append({"model_version": self.model_version, "predicted_grade": int(dr.argmax()),
                           "grade_probabilities": dr.tolist(), "referable_probability": min(1.0, float(dr[2:].sum())),
                           "quality": {"probabilities": overall_quality.tolist(),
                                       "predicted_class": int(overall_quality.argmax()),
                                       "poor_probability": min(1.0, float(overall_quality[self.quality_metadata["poor_class_indices"]].sum())),
                                       "defects": {head: values.tolist() for head, values in quality.items()
                                                   if head != "quality_grade"}},
                           "uncertainty": {"entropy": entropy, "confidence": float(dr.max()),
                                           "method": "temperature_scaled_normalized_entropy"},
                           "second_pass_activated": row["second"],
                           "processing_time_ms": row["elapsed"],
                           "data_purpose": self.dr_metadata["data_purpose"]})
            if include_uncalibrated:
                output[-1]["uncalibrated"] = {
                    "grade_probabilities": row["dr_logits"]["dr_grade"].softmax(1)[0].tolist(),
                    "quality_probabilities": row["quality_logits"]["quality_grade"].softmax(1)[0].tolist()}
        return output
