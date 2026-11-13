"""Generated software fixtures for the FDR26 archive join and overlap audit."""

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
from PIL import Image
import pytest

from scripts.audit_fdr26_readiness import canonical_pairs

from scripts.prepare_fdr26 import inspect_archive, prefix_sensitivity, screen_overlap


def test_internal_pair_order_does_not_change_identity_or_attributes():
    keys = ('first_image_id', 'second_image_id')
    pair = {'first_image_id': 'FIXTURE001', 'second_image_id': 'FIXTURE001(2)',
            'byte_exact': True, 'phash_distance': 0, 'label_conflict': False}
    reversed_pair = {**pair, keys[0]: pair[keys[1]], keys[1]: pair[keys[0]]}
    assert canonical_pairs([pair], keys) == canonical_pairs([reversed_pair], keys)


def image_payload(image_format):
    with BytesIO() as buffer:
        Image.new("RGB", (32, 24), (80, 120, 60)).save(buffer, format=image_format)
        return buffer.getvalue()


def write_archive(path, members):
    with ZipFile(path, "w") as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def test_archive_preserves_exact_stem_grade_join_and_decodes_jpeg(tmp_path):
    path = tmp_path / "generated.zip"
    labels = {"FIXTURE001": 0, "FIXTURE002": 4}
    pixels = image_payload("JPEG")
    write_archive(path, [("images/", b""), ("images/FIXTURE002.jpg", pixels), ("images/FIXTURE001.jpg", pixels)])

    rows = inspect_archive(path, labels)

    assert {row["image_id"]: row["dr_grade"] for row in rows} == labels
    assert {(row["format"], row["width"], row["height"]) for row in rows} == {("JPEG", 32, 24)}
    assert len({row["sha256"] for row in rows}) == len({row["phash"] for row in rows}) == 1


@pytest.mark.parametrize(
    ("members", "image_format"),
    [
        (["images/fixture001.jpg"], "JPEG"),
        (["images/FIXTURE001.jpg", "images/FIXTURE002.jpg"], "JPEG"),
        (["images/../FIXTURE001.jpg"], "JPEG"),
        (["images/FIXTURE001.jpg"], "PNG"),
    ],
)
def test_archive_rejects_nonexact_joins_and_disguised_png(tmp_path, members, image_format):
    path = tmp_path / "generated.zip"
    payload = image_payload(image_format)
    write_archive(path, [(name, payload) for name in members])

    with pytest.raises(ValueError):
        inspect_archive(path, {"FIXTURE001": 0})


def test_archive_keeps_duplicate_style_orphan_ungraded_and_excludes_linked_rows(tmp_path):
    path = tmp_path / "generated.zip"
    pixels = image_payload("JPEG")
    write_archive(path, [("images/FIXTURE001.jpg", pixels), ("images/FIXTURE001(2).jpg", pixels)])
    labels = {"FIXTURE001": 2, "FIXTURE002": 4}

    with pytest.raises(ValueError):
        inspect_archive(path, labels)
    rows = inspect_archive(path, labels, expected_unmatched=1)
    references = pd.DataFrame(
        [{"dataset_source": "synthetic", "image_id": "reference", "sha256": "f" * 64, "phash": "7" * 16}]
    )
    internal, cross, summary = screen_overlap(rows, references)

    assert {row["member"]: row["dr_grade"] for row in rows} == {
        "images/FIXTURE001.jpg": 2,
        "images/FIXTURE001(2).jpg": None,
    }
    assert len(internal) == 1 and internal[0]["byte_exact"] and not internal[0]["label_conflict"] and not cross
    assert summary["excluded_images"] == 2 and summary["eligible_images"] == 0
    assert summary["unmatched_component_excluded_images"] == 2 and summary["conflicting_components"] == 0
    assert not any(row["eligible"] for row in rows)


def test_reference_overlap_excludes_transitive_component_and_preserves_raw_rows():
    rows = [
        {
            "member": f"images/FIXTURE{index:03}.jpg",
            "image_id": f"FIXTURE{index:03}",
            "dr_grade": grade,
            "format": "JPEG",
            "sha256": f"{index:064x}",
            "phash": f"{fingerprint:016x}",
            "width": 32,
            "height": 24,
        }
        for index, (grade, fingerprint) in enumerate(
            [(0, 0), (0, 3), (0, 15), (3, 0x5555555555555555), (4, 0x5555555555555556), (1, 0x7FFFFFFFFFFFFFFF)],
            1,
        )
    ]
    references = pd.DataFrame(
        [{"dataset_source": "synthetic", "image_id": "reference", "sha256": "f" * 64, "phash": "0000000000000000"}]
    )

    internal, cross, summary = screen_overlap(rows, references)

    assert len(internal) == 3 and len(cross) == 2
    assert summary["excluded_images"] == 5
    assert summary["eligible_images"] == summary["representatives"] == 1
    assert [row["cross_candidate"] for row in rows] == [True, True, True, False, False, False]
    assert [row["label_conflict"] for row in rows] == [False, False, False, True, True, False]
    assert [row["representative"] for row in rows] == [False, False, False, False, False, True]
    assert [row["dr_grade"] for row in rows] == [0, 0, 0, 3, 4, 1]


def test_prefix_sensitivity_propagates_exclusion_across_alternating_links():
    rows = [
        {
            "image_id": image_id,
            "dr_grade": 0,
            "component": component,
            "eligible": eligible,
            "representative": representative,
        }
        for image_id, component, eligible, representative in [
            ("FIXTUREA01", "first", False, False),
            ("FIXTUREA02", "second", True, True),
            ("FIXTUREB01", "second", True, False),
            ("FIXTUREB02", "third", True, True),
            ("FIXTUREC01", "third", True, False),
            ("FIXTUREC02", "fourth", True, True),
            ("FIXTURED01", "fifth", True, True),
            ("FIXTUREE01", "fifth", True, False),
        ]
    ]

    assert prefix_sensitivity(rows) == {
        "prefix_sensitivity_excluded_images": 6,
        "prefix_sensitivity_eligible_images": 2,
        "prefix_sensitivity_representatives": 1,
    }
    assert sum(row["eligible"] for row in rows) == 7
