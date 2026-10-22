"""Measure actual CPU latency on supplied images with real artifact-backed inference."""
from pathlib import Path
import sys
import argparse
import json
import platform
import statistics
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image
from qsart.inference.engine import InferenceEngine


def cpu_model():
    if platform.system() == "Windows":
        try:
            return subprocess.check_output(["powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dr-checkpoint", required=True)
    parser.add_argument("--quality-checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="holdout", choices=["validation", "calibration", "holdout", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--second-pass-budget", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, help="Deterministic manifest-prefix timing sample; default all")
    args = parser.parse_args()
    if min(args.batch_size, args.repeats) < 1:
        parser.error("Batch size and repeats must be positive")
    if args.max_images is not None and args.max_images < 1:
        parser.error("max-images must be positive")
    engine = InferenceEngine(args.dr_checkpoint, args.quality_checkpoint, args.calibration,
                             cpu_threads=args.cpu_threads)
    manifest_path = Path(args.manifest).resolve()
    frame = pd.read_csv(manifest_path)
    frame = frame.loc[frame.split == args.split].reset_index(drop=True)
    available_images = len(frame)
    if args.max_images:
        frame = frame.head(args.max_images)
    if frame.empty:
        raise ValueError("Requested benchmark partition is empty")
    images = []
    for path in frame.image_path:
        path = Path(path)
        path = path if path.is_absolute() else manifest_path.parent / path
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    engine.predict_images(images[:1])  # Explicit warm-up excluded from timings.
    process = psutil.Process()
    strategies = {"single": ("none", 0.0), "all": ("all", 1.0),
                  "uncertainty": ("uncertainty", args.second_pass_budget),
                  "severity": ("severity", args.second_pass_budget),
                  "budget": ("budget", args.second_pass_budget)}
    measurements, prediction_rows = {}, []
    for name, (policy, budget) in strategies.items():
        image_times, batch_times, activations, rss = [], [], [], [process.memory_info().rss]
        for repeat in range(args.repeats):
            for start in range(0, len(images), args.batch_size):
                batch = images[start:start + args.batch_size]
                before = time.perf_counter()
                results = engine.predict_images(batch, second_pass_policy=policy, second_pass_budget=budget)
                batch_times.append((time.perf_counter() - before) * 1000)
                rss.append(process.memory_info().rss)
                for index, result in enumerate(results):
                    image_times.append(result["processing_time_ms"])
                    activations.append(result["second_pass_activated"])
                    if repeat == 0:
                        row = frame.iloc[start + index]
                        prediction_rows.append({"strategy": name, "record_index": start + index,
                            "dr_grade": None if pd.isna(row.get("dr_grade")) else int(row.dr_grade),
                            "predicted_grade": result["predicted_grade"],
                            "grade_probabilities": json.dumps(result["grade_probabilities"]),
                            "referable_probability": result["referable_probability"],
                            "poor_probability": result["quality"]["poor_probability"],
                            "uncertainty": result["uncertainty"]["entropy"],
                            "second_pass_activated": result["second_pass_activated"]})
        measurements[name] = {"mean_ms_per_image": statistics.mean(image_times),
            "median_ms_per_image": statistics.median(image_times),
            "p95_ms_per_image": float(np.percentile(image_times, 95)),
            "mean_batch_wall_ms": statistics.mean(batch_times),
            "amortized_wall_ms_per_image": sum(batch_times) / len(image_times),
            "timing_note": "Per-image times include preprocessing/model passes; batch wall also includes controller/output assembly, excluding file decode and HTTP transport.",
            "second_pass_fraction": statistics.mean(activations),
            "max_observed_rss_mb": max(rss) / 1024 ** 2,
            "memory_note": "RSS sampled at batch boundaries; not a continuous peak measurement",
            "images": len(images), "repeats": args.repeats, "population_second_pass_budget": budget}
    result = {"hardware": {"cpu": cpu_model(), "os": platform.platform(), "python": platform.python_version(),
                            "torch": str(torch.__version__), "logical_cpus": psutil.cpu_count()},
              "configuration": {**engine.info(), "batch_size": args.batch_size, "split": args.split,
                                "model_forward_batch_size": 1, "cohort_chunk_size": args.batch_size,
                                "warmup_images": 1, "available_partition_images": available_images,
                                "timing_sample_images": len(frame),
                                "timing_sample_rule": "manifest prefix" if args.max_images else "entire partition"},
              "strategies": measurements,
              "limitations": ["Pilot latency measurements do not establish clinical performance.",
                               "Resolution/model selection requires larger independent validation."]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(output.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
