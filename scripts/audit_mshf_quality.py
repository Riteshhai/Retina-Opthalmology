"""Reconstruct the locked MSHF study independently from its archived inputs."""

from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
from importlib.metadata import version
from itertools import combinations
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
from openpyxl import load_workbook
from openpyxl.utils.cell import column_index_from_string
from PIL import Image, ImageOps
from scipy.fft import dctn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mshf_quality_v1"
RAW = ROOT / "data/raw/mshf_v2"
OUTPUT = ROOT / "reports" / EXPERIMENT
WITNESS = ROOT / "artifacts" / EXPERIMENT / "witness.json"
PROTOCOL = ROOT / "docs/research/MSHF_QUALITY_V1_PROTOCOL.md"
REFERENCE_CONFIG = ROOT / "configs/idrid_data_v1.json"
QUALITY = ROOT / "checkpoints/development_v1/selected/quality.pt"
CALIBRATION = ROOT / "checkpoints/development_v1/selected/calibration.json"
ARCHIVE_ROOT = "MSHF dataset 2.0"
WORKBOOKS = {"individual": "Individual_scores.xlsx.xlsx", "summary": "MSHF_quality_scores.xlsx"}
GROUPS = {"DR-XJU": 235, "DR-ZJU": 187, "Glaucoma": 52, "Healthy": 26}
GROUP_FORMATS = {"DR-XJU": "JPEG", "DR-ZJU": "JPEG", "Glaucoma": "TIFF", "Healthy": "TIFF"}
PAPER_COUNTS = {"DR-XJU": (117, 118), "DR-ZJU": (40, 147), "Glaucoma": (50, 2), "Healthy": (0, 26)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def match(expected, actual, label):
    if isinstance(expected, dict):
        for key, value in expected.items():
            require(key in actual, f"{label}.{key} is missing")
            match(value, actual[key], f"{label}.{key}")
    elif isinstance(expected, float):
        require(
            type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-12),
            f"{label} differs",
        )
    else:
        require(expected == actual, f"{label} differs")


def verify_archive():
    path = RAW / "MSHF.zip"
    receipt_path = RAW / "receipt.json"
    metadata_path = RAW / "figshare_v2.json"
    receipt, metadata = read(receipt_path), read(metadata_path)
    expected = {
        "article_id": 21507564,
        "version": 2,
        "file_id": 39485878,
        "doi": "10.6084/m9.figshare.21507564.v2",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "download_url": "https://ndownloader.figshare.com/files/39485878",
        "bytes": 1136891149,
        "publisher_md5": "58203e6c2e064dafc800f5b83887487b",
        "archive_sha256": "115def6513bc1516d77ceb4ac2e90a9fcf582e825604c08be9524f5fd6c4a219",
        "metadata_sha256": digest(metadata_path),
    }
    match(expected, receipt, "acquisition receipt")
    match(
        {
            "id": expected["article_id"],
            "version": expected["version"],
            "doi": expected["doi"],
            "license": {"name": expected["license"], "url": expected["license_url"]},
        },
        metadata,
        "deposit metadata",
    )
    require(len(metadata["files"]) == 1, "Unexpected file listing in author deposit")
    match(
        {
            "id": expected["file_id"],
            "size": expected["bytes"],
            "download_url": expected["download_url"],
            "supplied_md5": expected["publisher_md5"],
            "computed_md5": expected["publisher_md5"],
        },
        metadata["files"][0],
        "deposit file",
    )
    require(
        path.stat().st_size == expected["bytes"]
        and digest(path, "md5") == expected["publisher_md5"]
        and digest(path) == expected["archive_sha256"],
        "Archived release bytes differ from the author deposit receipt",
    )
    return path, {**expected, "receipt_sha256": digest(receipt_path)}


def reference_rows():
    config = read(REFERENCE_CONFIG)
    require(
        config["near_duplicate_distance"] == 2 and len(config["references"]) == 4, "Reference scope differs"
    )
    records = {}
    counts = Counter()
    for entry in config["references"]:
        path = ROOT / entry["path"]
        require(
            path.resolve().is_relative_to(ROOT) and digest(path) == entry["sha256"],
            "Reference digest differs",
        )
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        dataset = entry["dataset"]
        require(len(rows) == entry["rows"], "Reference row count differs")
        counts[dataset] += len(rows)
        for row in rows:
            require(
                row["dataset_source"] == dataset
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                and re.fullmatch(r"[0-9a-f]{16}", row["phash"]),
                "Reference row fingerprint differs",
            )
            key = dataset, row["image_id"]
            evidence = row["sha256"], int(row["phash"], 16)
            require(records.setdefault(key, evidence) == evidence, "Conflicting reference fingerprint")
    return records, counts


def phash(image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    data = np.asarray(rgb)
    lit = np.any(data > 10, axis=2)
    if lit.sum() >= 0.01 * lit.size:
        y, x = np.where(lit)
        rgb = rgb.crop((int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1))
    left, top = rgb.width // 8, rgb.height // 8
    right, bottom = rgb.width * 7 // 8, rgb.height * 7 // 8
    channel = rgb.crop((left, top, right, bottom)).getchannel("G")
    pixels = np.array(channel.resize((32, 32), Image.Resampling.BICUBIC), dtype=float)
    frequencies = dctn(pixels, type=2)[:8, :8].ravel()[1:]
    bits = 0
    for flag in frequencies > np.median(frequencies):
        bits = (bits << 1) | int(flag)
    return f"{bits:016x}"


def named_rows(sheet, first_row, names):
    result = {}
    for index, values in enumerate(sheet.iter_rows(min_row=first_row, values_only=True), first_row):
        name = values[0]
        if name in names:
            require(name not in result, "Workbook contains an ambiguous joined name")
            result[name] = index, values
    return result


def archive_rows(bundle):
    base = (ARCHIVE_ROOT, "Original", "CFP")
    members = sorted(
        name
        for name in bundle.namelist()
        if PurePosixPath(name).parts[:3] == base and PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg"}
    )
    paths = [PurePosixPath(name) for name in members]
    require(
        all(len(path.parts) == 5 for path in paths) and Counter(p.parent.name for p in paths) == GROUPS,
        "Conventional image cohort differs",
    )
    names = {path.name for path in paths}
    require(len(names) == len(paths), "Conventional image filenames are ambiguous")
    ratings_book = load_workbook(
        BytesIO(bundle.read(f"{ARCHIVE_ROOT}/{WORKBOOKS['individual']}")), read_only=True, data_only=False
    )
    summary_data = bundle.read(f"{ARCHIVE_ROOT}/{WORKBOOKS['summary']}")
    cached_book = load_workbook(BytesIO(summary_data), read_only=True, data_only=True)
    formula_book = load_workbook(BytesIO(summary_data), read_only=True, data_only=False)
    try:
        rating_rows = named_rows(ratings_book["Sheet1"], 3, names)
        headers = list(ratings_book["Sheet1"].iter_rows(min_row=1, max_row=2, values_only=True))
        require(
            headers[0][0] == "image name" and all(headers[1][i] == "Overall" for i in (4, 9, 14)),
            "Unexpected overall rating columns",
        )
        require(set(rating_rows) == names, "Missing individual ratings")
        summaries = {}
        for group in GROUPS:
            group_paths = [path for path in paths if path.parent.name == group]
            keys = {value for path in group_paths for value in (path.name, path.stem)}
            first_row = 3 if group == "DR-ZJU" else 2
            cached = named_rows(cached_book[group], first_row, keys)
            formulas = named_rows(formula_book[group], first_row, keys)
            for path in group_paths:
                matches = [key for key in (path.name, path.stem) if key in cached]
                require(len(matches) == 1, "Summary join is missing or ambiguous")
                name = matches[0]
                index, values = formulas[name]
                require(index == cached[name][0], "Summary formula/cached row mismatch")
                score = cached[name][1][4]
                require(type(score) is int and score in (0, 1), "Cached overall summary score is not binary")
                formula = values[4] if isinstance(values[4], str) and values[4].startswith("=") else None
                empty_inputs = False
                if formula:
                    cells = re.findall(r"([A-Z]+)([0-9]+)", formula)
                    require(
                        cells and all(int(row) == index for _, row in cells),
                        "Unexpected summary formula inputs",
                    )
                    columns = [column_index_from_string(column) - 1 for column, _ in cells]
                    empty_inputs = all(column >= len(values) or values[column] is None for column in columns)
                summaries[path.name] = (
                    name,
                    "exact" if name == path.name else "stem",
                    score,
                    formula,
                    empty_inputs,
                )
        rows = []
        blank_formula_inputs = Counter()
        for name, path in zip(members, paths, strict=True):
            raw_ratings = rating_rows[path.name][1]
            ratings = [raw_ratings[index] for index in (4, 9, 14)]
            require(
                all(type(value) is int and value in (0, 1) for value in ratings),
                "Invalid individual binary rating",
            )
            summary_name, join, summary, formula, blank = summaries[path.name]
            group = path.parent.name
            blank_formula_inputs[group] += blank
            payload = bundle.read(name)
            with Image.open(BytesIO(payload)) as image:
                image_format = image.format
                require(
                    image_format == GROUP_FORMATS[group]
                    and image.mode == "RGB"
                    and getattr(image, "n_frames", 1) == 1,
                    "Conventional image encoding differs from the archived release",
                )
                image.load()
                width, height = image.size
                perceptual = phash(image)
            rows.append(
                {
                    "member": name,
                    "group": group,
                    "filename": path.name,
                    "format": image_format,
                    "width": width,
                    "height": height,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "phash": perceptual,
                    "ratings": ratings,
                    "label": int(sum(ratings) >= 2),
                    "unanimous": len(set(ratings)) == 1,
                    "summary_filename": summary_name,
                    "summary_join": join,
                    "summary_cached": summary,
                    "summary_formula": formula,
                }
            )
        return rows, blank_formula_inputs
    finally:
        ratings_book.close()
        cached_book.close()
        formula_book.close()


def inventory(rows, blanks):
    dimensions = Counter((row["width"], row["height"]) for row in rows)
    groups = {}
    for group in GROUPS:
        part = [row for row in rows if row["group"] == group]
        groups[group] = {
            "images": len(part),
            "poor": sum(row["label"] == 0 for row in part),
            "good": sum(row["label"] == 1 for row in part),
            "unanimous": sum(row["unanimous"] for row in part),
            "summary_cached_poor": sum(row["summary_cached"] == 0 for row in part),
            "summary_cached_good": sum(row["summary_cached"] == 1 for row in part),
            "summary_disagreements": sum(row["summary_cached"] != row["label"] for row in part),
            "summary_stem_joins": sum(row["summary_join"] == "stem" for row in part),
            "summary_formula_rows": sum(row["summary_formula"] is not None for row in part),
            "summary_blank_formula_inputs": blanks[group],
            "paper_poor": PAPER_COUNTS[group][0],
            "paper_good": PAPER_COUNTS[group][1],
        }
    return {
        "images": len(rows),
        "images_decoded": len(rows),
        "formats": dict(Counter(row["format"] for row in rows)),
        "dimensions": [
            {"width": width, "height": height, "images": count}
            for (width, height), count in sorted(dimensions.items())
        ],
        "groups": groups,
    }


def overlaps(rows, references, counts):
    members = {row["member"]: row for row in rows}
    fingerprints = {name: int(row["phash"], 16) for name, row in members.items()}
    neighbors = defaultdict(set)
    internal = []
    for a, b in combinations(members, 2):
        distance = (fingerprints[a] ^ fingerprints[b]).bit_count()
        exact = members[a]["sha256"] == members[b]["sha256"]
        if exact or distance <= 2:
            neighbors[a].add(b)
            neighbors[b].add(a)
            internal.append(
                {
                    "first_image_id": a,
                    "second_image_id": b,
                    "byte_exact": exact,
                    "phash_distance": distance,
                    "label_conflict": members[a]["label"] != members[b]["label"],
                }
            )
    cross = []
    for name, row in members.items():
        for (dataset, identity), (sha, phash_value) in references.items():
            distance = (fingerprints[name] ^ phash_value).bit_count()
            exact = row["sha256"] == sha
            if exact or distance <= 2:
                cross.append(
                    {
                        "external_image_id": name,
                        "reference_dataset": dataset,
                        "reference_image_id": identity,
                        "byte_exact": exact,
                        "phash_distance": distance,
                    }
                )
    anchors = {pair["external_image_id"] for pair in cross}
    seen = set()
    component_stats = Counter()
    for start in members:
        if start in seen:
            continue
        pending = [start]
        component = {start}
        while pending:
            candidate = pending.pop()
            new = neighbors[candidate] - component
            component.update(new)
            pending.extend(new)
        seen.update(component)
        conflict = len({members[name]["label"] for name in component}) > 1
        cross_hit = bool(component & anchors)
        eligible = not (conflict or cross_hit)
        representative = min(component)
        digest_prefix = hashlib.sha256("|".join(sorted(component)).encode()).hexdigest()[:24]
        component_stats["internal_components"] += 1
        component_stats["internal_repeat_components"] += len(component) > 1
        component_stats["conflicting_components"] += conflict
        component_stats["cross_source_components"] += cross_hit
        for name in component:
            members[name].update(
                component=f"mshf_component_{digest_prefix}",
                label_conflict=conflict,
                cross_candidate=cross_hit,
                eligible=eligible,
                representative=eligible and name == representative,
            )
    datasets = {}
    for dataset, total in counts.items():
        pairs = [row for row in cross if row["reference_dataset"] == dataset]
        datasets[dataset] = {
            "rows": total,
            "unique_identities": sum(key[0] == dataset for key in references),
            "candidate_pairs": len(pairs),
            "exact_pairs": sum(row["byte_exact"] for row in pairs),
        }
    summary = {
        "near_duplicate_distance": 2,
        "fingerprint_method": "central_green_dct_phash_v1",
        **component_stats,
        "internal_candidate_pairs": len(internal),
        "internal_exact_pairs": sum(row["byte_exact"] for row in internal),
        "cross_source_candidate_pairs": len(cross),
        "cross_source_exact_pairs": sum(row["byte_exact"] for row in cross),
        "cross_source_candidate_images": len(anchors),
        "excluded_conflict_images": sum(row["label_conflict"] for row in rows),
        "excluded_cross_source_images": sum(row["cross_candidate"] for row in rows),
        "excluded_images": sum(not row["eligible"] for row in rows),
        "reference_datasets": datasets,
    }
    return internal, cross, summary


def verify_probabilities(rows, temperature):
    max_error = 0.0
    for row in rows:
        fields = ("logits", "raw_probabilities", "calibrated_probabilities")
        if not row["eligible"]:
            require(all(row[name] is None for name in fields), "Excluded rows have predictions")
            continue
        logits = np.asarray(row["logits"], dtype=np.float32)
        require(logits.shape == (2,) and np.isfinite(logits).all(), "Invalid stored logits")
        for name, divisor in (("raw_probabilities", 1.0), ("calibrated_probabilities", temperature)):
            values = logits / np.float32(divisor)
            exp = np.exp(values - values.max())
            calculated = exp / exp.sum()
            saved = np.asarray(row[name], dtype=np.float64)
            require(
                saved.shape == (2,)
                and np.isfinite(saved).all()
                and (saved >= 0).all()
                and (saved <= 1).all()
                and abs(saved.sum() - 1.0) <= 2e-7,
                "Invalid stored probabilities",
            )
            max_error = max(max_error, float(np.max(np.abs(calculated - saved))))
    require(max_error <= 2e-7, "Stored probabilities do not match frozen logits and temperature")
    return max_error


def discrimination(poor, scores):
    positives, negatives = scores[poor], scores[~poor]
    if not len(positives) or not len(negatives):
        return None, None
    comparisons = positives[:, None] - negatives[None, :]
    auc = float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))
    order = np.argsort(-scores)
    ordered_scores, ordered_poor = scores[order], poor[order]
    last_of_tie = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    indices = np.flatnonzero(last_of_tie)
    true_positives = np.cumsum(ordered_poor)[indices]
    recall = true_positives / len(positives)
    precision = true_positives / (indices + 1)
    average_precision = float(np.diff(recall, prepend=0.0) @ precision)
    return auc, average_precision


def binary_metrics(labels, probabilities):
    if not len(labels):
        return None
    poor = labels == 0
    predictions = np.where(probabilities[:, 0] > 0.50, 0, 1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    matrix = np.bincount(2 * labels + predictions, minlength=4).reshape(2, 2)
    sensitivity = float(matrix[0, 0] / poor.sum()) if poor.any() else None
    specificity = float(matrix[1, 1] / (~poor).sum()) if (~poor).any() else None
    balanced = (
        (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    )
    auc, average_precision = discrimination(poor, probabilities[:, 0])
    bins = np.searchsorted(np.arange(1, 10) / 10.0, confidence, side="right")
    residual = np.bincount(bins, weights=correct - confidence, minlength=10)
    expected = (labels[:, None] == np.arange(2)).astype(float)
    return {
        "confusion_matrix": matrix.tolist(),
        "poor_sensitivity": sensitivity,
        "good_specificity": specificity,
        "balanced_accuracy": balanced,
        "poor_auroc": auc,
        "poor_average_precision": average_precision,
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()),
        "brier": float(np.sum((probabilities - expected) ** 2) / len(labels)),
        "ece": float(np.abs(residual).sum() / len(labels)),
        "accuracy": float(correct.mean()),
    }


def population_metrics(rows):
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    poor = int((labels == 0).sum())
    return {
        "images": len(rows),
        "poor": poor,
        "good": len(rows) - poor,
        "always_good_accuracy": (len(rows) - poor) / len(rows) if rows else None,
        "raw": binary_metrics(labels, np.asarray([row["raw_probabilities"] for row in rows])),
        "calibrated": binary_metrics(labels, np.asarray([row["calibrated_probabilities"] for row in rows])),
    }


def verify_populations(rows, report):
    main = [row for row in rows if row["eligible"]]
    populations = {
        "main": main,
        "representative": [row for row in main if row["representative"]],
        "unanimous": [row for row in main if row["unanimous"]],
    }
    for name, part in populations.items():
        expected = {
            "overall": population_metrics(part),
            "groups": {
                group: population_metrics([row for row in part if row["group"] == group]) for group in GROUPS
            },
        }
        match(expected, report[name], f"{name} aggregate metrics")
    return main


def replay(bundle, rows, model_info):
    sys.path.insert(0, str(ROOT))
    import torch
    from qsart.models.checkpoint import load_checkpoint
    from qsart.training.data import image_to_tensor

    torch.set_num_threads(model_info["cpu_threads"])
    model, checkpoint = load_checkpoint(QUALITY)
    metadata = checkpoint["metadata"]
    require(
        metadata["task"] == "quality"
        and metadata["heads"]["quality_grade"] == 2
        and metadata["poor_class_indices"] == [0],
        "Checkpoint overall-quality encoding differs",
    )
    match(
        {"model_version": metadata["model_version"], "image_size": metadata["image_size"]},
        model_info,
        "model",
    )
    errors = {}
    with torch.inference_mode():
        for group in GROUPS:
            samples = sorted((row for row in rows if row["group"] == group), key=lambda row: row["member"])[
                :2
            ]
            deltas = []
            for row in samples:
                with Image.open(BytesIO(bundle.read(row["member"]))) as image:
                    image.load()
                    tensor = image_to_tensor(
                        image.convert("RGB"),
                        metadata["image_size"],
                        metadata["normalization_mean"],
                        metadata["normalization_std"],
                    )
                    actual = model(tensor.unsqueeze(0))["quality_grade"][0].numpy()
                error = float(np.max(np.abs(actual.astype(float) - row["logits"])))
                require(error <= 1e-6, "Replay logits differ from the saved first-pass witness")
                deltas.append(error)
            errors[group] = {"images": len(samples), "max_absolute_logit_delta": max(deltas, default=None)}
    return errors


def git_digest(commit, path):
    relative = path.relative_to(ROOT).as_posix()
    data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def canonical_pairs(items):
    return sorted(json.dumps(row, sort_keys=True) for row in items)


def run():
    evaluation = OUTPUT / "evaluation.json"
    report, witness = read(evaluation), read(WITNESS)
    require(report["experiment_id"] == EXPERIMENT, "Unexpected experiment")
    commit = report["source_commit"]
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "Invalid execution revision")
    source_hashes = report["source_sha256"]
    require("scripts/evaluate_mshf_quality.py" in source_hashes, "Missing runner source binding")
    for name, expected in source_hashes.items():
        path = ROOT / name
        require(
            path.resolve().is_relative_to(ROOT) and digest(path) == expected == git_digest(commit, path),
            "Execution source differs from the lock commit",
        )
    require(
        digest(PROTOCOL) == report["protocol_sha256"] == git_digest(commit, PROTOCOL),
        "Protocol was not locked",
    )
    auditor = Path(__file__).resolve()
    require(digest(auditor) == git_digest(commit, auditor), "Auditor source was not locked")
    require(digest(WITNESS) == report["witness_sha256"], "Private witness digest differs")
    require(
        all(version(name) == expected for name, expected in report["package_versions"].items()),
        "Frozen execution package versions differ",
    )
    archive, archive_info = verify_archive()
    match(archive_info, report["archive"], "archive provenance")
    calibration = read(CALIBRATION)["quality"]
    temperature = calibration["heads"]["quality_grade"]["temperature"]
    checkpoint_digest = digest(QUALITY)
    require(
        checkpoint_digest == calibration["checkpoint_sha256"],
        "Frozen quality checkpoint and temperature differ",
    )
    match(
        {
            "quality_checkpoint_sha256": checkpoint_digest,
            "calibration_sha256": digest(CALIBRATION),
            "cpu_threads": 2,
            "temperature": temperature,
            "model_version": calibration["model_version"],
            "poor_probability_threshold": 0.5,
        },
        report["model"],
        "frozen quality model",
    )
    require(temperature == 1.0349426972146167, "The declared production temperature differs")
    references, counts = reference_rows()
    bindings = [
        {key: value for key, value in item.items() if key != "path"}
        for item in read(REFERENCE_CONFIG)["references"]
    ]
    require(bindings == report["reference_manifest_bindings"], "Public reference manifest bindings differ")
    with ZipFile(archive) as bundle:
        for kind, filename in WORKBOOKS.items():
            payload = bundle.read(f"{ARCHIVE_ROOT}/{filename}")
            require(
                hashlib.sha256(payload).hexdigest() == report["workbook_sha256"][kind],
                "Archived workbook digest differs",
            )
        rows, blanks = archive_rows(bundle)
        match(inventory(rows, blanks), report["inventory"], "annotation inventory")
        internal, cross, summary = overlaps(rows, references, counts)
        match(summary, report["overlap"], "overlap summary")
        for key, expected in (("internal_candidates", internal), ("cross_source_candidates", cross)):
            require(canonical_pairs(expected) == canonical_pairs(witness[key]), f"{key} differs")
        saved = witness["rows"]
        require(len(rows) == len(saved), "Incomplete saved image witness")
        for expected, actual in zip(rows, saved, strict=True):
            match(expected, actual, "archived row witness")
        probability_delta = verify_probabilities(saved, temperature)
        main = verify_populations(saved, report["populations"])
        replays = replay(bundle, main, report["model"])
    return {
        "audit_status": "passed",
        "experiment_id": EXPERIMENT,
        "source_commit": commit,
        "evaluation_sha256": digest(evaluation),
        "auditor_sha256": digest(auditor),
        "private_output_sha256": digest(WITNESS),
        "archive_sha256": archive_info["archive_sha256"],
        "images_reconstructed": len(rows),
        "reference_manifest_rows": sum(counts.values()),
        "internal_candidate_pairs": len(internal),
        "cross_source_candidate_pairs": len(cross),
        "metric_cells_reconstructed": 30,
        "scored_images_verified": len(main),
        "max_absolute_probability_delta": probability_delta,
        "frozen_replay": replays,
    }


if __name__ == "__main__":
    target = OUTPUT / "independent_audit.json"
    require(not target.exists(), "Independent audit already exists")
    receipt = run()
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))
