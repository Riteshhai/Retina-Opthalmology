"""Verify the preserved seed-2026 anchors and read their immutable RGB caches."""

from dataclasses import dataclass

from qsart.models.checkpoint import sha256_file
from qsart.training.aptos384 import ImageStore, PREPROCESSING
from scripts.aptos_resolution_match import REFERENCE_REPORT_SHA256
from scripts.run_aptos_development import read_json


EXPERIMENT = "aptos_resolution_seed2027_v1"
SEED = 2027
ORIGINAL_SEED = 2026
ANCHORS = {
    224: (
        "aptos_224_matched_v1",
        "552503f95f8a049eef6d8c89226725ae2b7e3499beb5096c351a8558d029c73e",
        "671113893a87d8504e754de29b1830a42c9696da398aa02362f9a0288a13b58d",
    ),
    384: (
        "aptos_384_v1",
        REFERENCE_REPORT_SHA256,
        "a5a916c4f1e17169e5fedefb2232e719ba9c1b27bcf28b893893b52bef10c82b",
    ),
}


@dataclass(frozen=True)
class Anchor:
    report: dict
    lock: dict
    cache: dict
    bindings: dict[str, str]


def verified_json(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"Preserved evidence differs: {path.name}")
    return read_json(path)


def read_anchor(root, size):
    experiment, evaluation_sha, audit_sha = ANCHORS[size]
    reports = root / "reports" / experiment
    private = root / "artifacts" / experiment
    evaluation, audit_path = reports / "evaluation.json", reports / "independent_audit.json"
    report = verified_json(evaluation, evaluation_sha)
    audit = verified_json(audit_path, audit_sha)
    lock_path = private / "execution_lock.json"
    lock = verified_json(lock_path, report["execution_lock_sha256"])
    if audit["audit_status"] != "passed" or audit["evaluation_sha256"] != evaluation_sha:
        raise ValueError("Expected the audited seed-2026 result")
    bindings = {evaluation: evaluation_sha, audit_path: audit_sha, lock_path: report["execution_lock_sha256"]}
    return Anchor(report, lock, audit["disk_cache"]["fit_selection"],
                  {path.relative_to(root).as_posix(): digest for path, digest in bindings.items()})


def checked_cache(root, size, anchor, fitting, manifest_sha, content_sha):
    """Open the exact audited pixel store without ever building or editing a cache."""
    folder = root / "artifacts" / ANCHORS[size][0] / "input_store"
    entries = list(folder.glob("*/metadata.json"))
    if len(entries) != 1:
        raise ValueError("Expected one preserved fitting and selection cache")
    info = read_json(entries[0])
    metadata = info["metadata"]
    pixels = entries[0].with_name("pixels.rgb")
    expected = {
        "schema": "qsart.disk_rgb.v1",
        "scope": "fit_selection",
        "image_size": size,
        "preprocessing": PREPROCESSING,
        "manifest_sha256": manifest_sha,
        "input_content_sha256": content_sha,
        "manifest_rows": fitting.index.tolist(),
        "partitions": fitting.split.tolist(),
    }
    if (metadata != expected or pixels.stat().st_size != len(fitting) * size * size * 3
            or info["file_sha256"] != anchor.cache["sha256"] or sha256_file(pixels) != anchor.cache["sha256"]):
        raise ValueError("Read-only cache differs from the audited fitting and selection pixels")
    return ImageStore(pixels, metadata), {**info, "bytes": pixels.stat().st_size}
