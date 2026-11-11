"""Generated action fixtures for patient rollups and capacity boundaries."""

from scripts.run_patient_triage import AUTO, RECAPTURE, REVIEW, patient_capacity, patient_rollup


def test_patient_rollup_distinguishes_partial_auto_exposure_from_full_clearance():
    grades = [0, 0, 3, 3, 2, 2, 0, 0, 0, 0, 1, 1]
    actions = [AUTO, AUTO, REVIEW, RECAPTURE, AUTO, AUTO, AUTO, AUTO, AUTO, AUTO, RECAPTURE, AUTO]
    rollup = patient_rollup(actions, grades, [range(0, 4), range(4, 8), range(8, 12)])
    assert rollup['patient_actions'] == {AUTO: 1, RECAPTURE: 1, REVIEW: 1}
    assert rollup['patients_with_any_automatic_image'] == 3
    assert rollup['patients_with_partial_automatic_images'] == 2
    assert rollup['referable_patients_with_any_automatic_image'] == 2
    assert rollup['missed_referable_images'] == 2
    assert rollup['patients_with_automatic_referable_image'] == 1
    assert rollup['fully_automatic_referable_patients'] == 1
    assert rollup['automatic_image_miss_fraction_among_exposed_patients'] == 1 / 3
    assert rollup['full_patient_clearance_miss_fraction'] == 1.


def test_patient_rollup_leaves_miss_rates_undefined_without_automatic_images():
    rollup = patient_rollup([REVIEW, RECAPTURE, REVIEW, REVIEW], [0, 0, 2, 2], [range(4)])
    assert rollup['patient_actions'] == {AUTO: 0, RECAPTURE: 0, REVIEW: 1}
    assert rollup['automatic_image_miss_fraction_among_exposed_patients'] is None
    assert rollup['full_patient_clearance_miss_fraction'] is None


def test_patient_capacity_requires_review_spill_when_recapture_cap_is_full():
    counts = {AUTO: 2, RECAPTURE: 3, REVIEW: 5}
    overload = patient_capacity(counts, .5, .2)
    assert overload['status'] == 'overloaded'
    assert overload['minimum_review_after_spill'] == 6
    assert overload['review_shortfall'] == 1
    assert overload['maximum_full_automatic_patients'] is None
    feasible = patient_capacity(counts, .6, .2)
    assert feasible['status'] == 'feasible'
    assert feasible['review_shortfall'] == 0
    assert feasible['maximum_full_automatic_patients'] == 2
