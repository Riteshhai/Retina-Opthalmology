"""Generated fixtures exercise the repeat's read-only data and pairing guards."""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from qsart.training.aptos384 import PREPROCESSING
from scripts import aptos_resolution_repeat as repeat
from scripts.summarize_aptos_resolution_repeat import verify_matching_locks


def test_repeat_refuses_changed_cache_bytes(tmp_path):
    entry = tmp_path / "artifacts" / repeat.ANCHORS[224][0] / "input_store" / "fixture"
    entry.mkdir(parents=True)
    pixels = entry / "pixels.rgb"
    payload = bytes(224 * 224 * 3)
    pixels.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    anchor = SimpleNamespace(cache={"sha256": digest})
    fitting = pd.DataFrame({"split": ["train"]}, index=[13])
    metadata = {"schema": "qsart.disk_rgb.v1", "scope": "fit_selection", "image_size": 224,
                "preprocessing": PREPROCESSING, "manifest_sha256": "manifest", "input_content_sha256": "content",
                "manifest_rows": [13], "partitions": ["train"]}
    (entry / "metadata.json").write_text(json.dumps({"metadata": metadata, "file_sha256": digest}), encoding="utf-8")
    store, info = repeat.checked_cache(tmp_path, 224, anchor, fitting, "manifest", "content")
    assert store.image(0).tobytes() == payload and info["bytes"] == len(payload)
    pixels.write_bytes(b"\x01" + payload[1:])
    with pytest.raises(ValueError, match="Read-only cache differs"):
        repeat.checked_cache(tmp_path, 224, anchor, fitting, "manifest", "content")


def test_repeat_pair_rejects_different_seeded_heads():
    low = {"configuration": {"image_size": 224, "seed": 2027}, "source_commit": "source",
           "frozen_input_bindings": {}, "runtime_manifest_sha256": "manifest",
           "runtime_calibration_roles_sha256": "roles", "fitting_selection_content_sha256": "data",
           "imagenet_pretrained": {}, "environment": {}, "fresh_head_parameters_sha256": "head"}
    high = {**low, "configuration": {"image_size": 384, "seed": 2027}}
    verify_matching_locks(low, high)
    with pytest.raises(ValueError, match="initialization"):
        verify_matching_locks(low, {**high, "fresh_head_parameters_sha256": "different"})
