"""Synthetic software cases for routing and result availability in the Set-C runner."""

from scripts.run_set_c_validation import allocations, evaluate


def test_frozen_policy_grid_does_not_route_on_labels():
    images = [{'p': [.94, .02, .02, .01, .01], 'good_quality': .9, 'entropy': .22,
               'ordinal': [0, 1, 2, 3, 4], 'lac': [0], 'grade': 0, 'patient': str(i // 4)}
              for i in range(12)]
    alternate = [{**row, 'grade': 3 if row['patient'] == '0' else 0} for row in images]
    pointwise, scenarios = allocations(images)
    assert (pointwise, scenarios) == allocations(alternate)
    assert len(scenarios) == 96

    _, results, comparisons = evaluate(alternate, pointwise, scenarios)
    rows = {row['id']: row for row in results}
    blocked = rows['ordinal:dual_budget:0']
    assert blocked['status'] == 'overload' and blocked['outcomes'] is None
    assert blocked['diagnostics']['mandatory_review_shortfall'] == 9
    assert blocked['patient_capacity']['review_shortfall'] == 3
    all_review = rows['ordinal:dual_budget:3']['outcomes']
    assert all_review['patient_actions']['SPECIALIST_REVIEW'] == 3
    assert all_review['full_patient_clearance_miss_fraction'] is None
    auto = rows['no_cp:plain:0']['outcomes']
    assert auto['fully_automatic_severe_patients'] == 1
    assert auto['missed_severe_images'] == 4
    assert len(comparisons) == 84
    assert all(row['intervals'] is None for row in comparisons if row['status'] == 'unavailable')
