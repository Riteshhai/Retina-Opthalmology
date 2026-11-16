"""Exercise benchmark alignment and output comparison on software fixtures."""

import argparse

import pytest

from scripts import benchmark_qsart_x_cpu as benchmark


def test_interleaving_preserves_same_samples_and_rotates_targets(monkeypatch):
    calls = []
    sample = list(range(benchmark.PHOTO_COUNT))

    def measure(target, profile, images):
        calls.append((target, profile.name, tuple(images)))
        return tuple(images)

    monkeypatch.setattr(benchmark, 'measure_request', measure)
    measured = benchmark.collect_samples({thread: thread for thread in benchmark.THREADS}, sample)
    for profile in benchmark.WORKLOADS:
        size = profile.images_per_request
        expected = [tuple(sample[start:start+size]) for start in range(0, len(sample), size)]
        assert all(measured[threads][profile.name] == expected * benchmark.REPEATS for threads in benchmark.THREADS)
    timed_calls = calls[6:]
    assert [target for target, *_ in timed_calls[:3]] == [1, 2, 4]
    assert [target for target, *_ in timed_calls[3:6]] == [2, 4, 1]


def test_equivalence_tracks_numeric_tolerance_and_discrete_changes():
    baseline = {'results': [{'request_id': 'a', 'grade_probabilities': [.25], 'processing_time_ms': 12.,
                             'conformal': {'prediction_set': [2]}, 'second_pass_activated': True}]}
    near = {'results': [{'request_id': 'b', 'grade_probabilities': [.2500001], 'processing_time_ms': 90.,
                         'conformal': {'prediction_set': [2]}, 'second_pass_activated': True}]}
    result = benchmark.Comparison()
    result.compare(baseline, near)
    assert result.passed
    assert result.numeric_values_compared == 1

    near['results'][0]['grade_probabilities'] = [.25001]
    near['results'][0]['conformal']['prediction_set'] = [3]
    result.compare(baseline, near)
    assert not result.passed
    assert result.numeric_tolerance_violations == result.semantic_violations == 1


@pytest.mark.parametrize('target', ['1,https://127.0.0.1:8888,api', '1,http://example.com:8888,api',
                                    '1,http://localhost:8888,-api'])
def test_target_is_local_http_and_uses_safe_docker_container_name(target):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_target(target)
