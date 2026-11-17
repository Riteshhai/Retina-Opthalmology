"""Generated checks for matched flip slots and exact random assignments."""

import numpy as np

from scripts.run_matched_flip import random_delta_histogram, select_slots, summarize_random


def test_equal_workload_selects_per_request_and_breaks_ties_by_input_order():
    scores = [.8, .8, .8, .7, .1, .5, .5, .6]
    assert np.flatnonzero(select_slots(scores, 2, chunk=4)).tolist() == [0, 1, 5, 7]


def test_uniform_slot_histogram_preserves_duplicate_positions_across_requests():
    histogram = random_delta_histogram([[1, 0, 0, -1], [0, 0, 1, 1]], 1)
    assert histogram == {-1: 2, 0: 6, 1: 6, 2: 2}


def test_uniform_quantiles_are_discrete_and_include_the_attainable_extremes():
    blocks = [[1, 1, -1, -1, 0, 0, 0, 0]]
    histogram = random_delta_histogram(blocks, 2)
    assert histogram == {-2: 1, -1: 8, 0: 10, 1: 8, 2: 1}
    summary = summarize_random(histogram, blocks, 2)
    assert summary['assignments'] == 28 and summary['mean_change'] == 0.
    assert summary['central_95_changes'] == [-2, 2]
