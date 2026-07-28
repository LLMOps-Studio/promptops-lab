import pytest

from promptops_lab.evaluation.latency import compute_latency_stats


def test_single_sample_all_percentiles_equal_the_sample():
    stats = compute_latency_stats([2.5])

    assert stats.p50 == stats.p95 == stats.p99 == 2.5
    assert stats.n == 1


def test_percentiles_match_reference_values_for_1_to_100():
    # Linear-interpolation percentile of 1..100 (numpy's default method):
    # p50 = 50.5, p95 = 95.05, p99 = 99.01.
    samples = list(range(1, 101))

    stats = compute_latency_stats([float(s) for s in samples])

    assert stats.p50 == pytest.approx(50.5)
    assert stats.p95 == pytest.approx(95.05)
    assert stats.p99 == pytest.approx(99.01)
    assert stats.n == 100


def test_order_of_samples_does_not_matter():
    ordered = compute_latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    shuffled = compute_latency_stats([3.0, 1.0, 5.0, 2.0, 4.0])

    assert ordered == shuffled


def test_empty_samples_raises_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_latency_stats([])
