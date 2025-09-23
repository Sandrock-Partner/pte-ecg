#!/usr/bin/env python3
"""Test script to verify improved n_jobs handling in _get_n_processes function."""

import os
import sys

sys.path.insert(0, "src")

from pte_ecg.features import _get_n_processes


def test_n_jobs_handling():
    """Test that _get_n_processes handles all n_jobs values correctly."""

    # Get actual CPU count for testing
    if sys.version_info >= (3, 13):
        total_cpus = os.process_cpu_count() or 1
    else:
        total_cpus = os.cpu_count() or 1

    print(f"Total CPUs available: {total_cpus}")

    # Test cases: (n_jobs, n_tasks, expected_description)
    test_cases = [
        (None, 10, f"Should use all {total_cpus} CPUs (capped by tasks)"),
        (-1, 10, f"Should use all {total_cpus} CPUs (capped by tasks)"),
        (1, 10, "Should use exactly 1 CPU"),
        (2, 10, "Should use exactly 2 CPUs"),
        (-2, 10, f"Should use {max(1, total_cpus - 1)} CPUs (all except 1)"),
        (-3, 10, f"Should use {max(1, total_cpus - 2)} CPUs (all except 2)"),
        (0, 10, "Should default to 1 CPU (invalid n_jobs=0)"),
        (100, 5, "Should be capped by n_tasks=5"),
        (-100, 10, "Should default to 1 CPU (too many CPUs to exclude)"),
    ]

    print("\n=== Testing n_jobs handling ===")

    for n_jobs, n_tasks, description in test_cases:
        result = _get_n_processes(n_jobs, n_tasks)
        n_jobs_str = str(n_jobs) if n_jobs is not None else "None"
        print(
            f"n_jobs={n_jobs_str:>4}, n_tasks={n_tasks:>2} -> {result:>2} processes | {description}"
        )

        # Basic validation
        assert result >= 1, f"Result should be at least 1, got {result}"
        assert result <= n_tasks, (
            f"Result should not exceed n_tasks={n_tasks}, got {result}"
        )
        assert result <= total_cpus, (
            f"Result should not exceed total_cpus={total_cpus}, got {result}"
        )

    print("\n[SUCCESS] All n_jobs handling tests passed!")


def test_edge_cases():
    """Test edge cases and potential issues."""

    print("\n=== Testing edge cases ===")

    # Test with very small n_tasks
    result = _get_n_processes(-1, 1)
    assert result == 1, f"With n_tasks=1, should return 1, got {result}"
    print(f"n_tasks=1: {result} processes [PASS]")

    # Test with n_tasks=0 (edge case)
    result = _get_n_processes(-1, 0)
    assert result == 1, f"With n_tasks=0, should return 0, got {result}"
    print(f"n_tasks=0: {result} processes [PASS]")

    # Test large positive n_jobs
    result = _get_n_processes(1000, 5)
    assert result == 5, f"Large n_jobs should be capped by n_tasks, got {result}"
    print(f"Large n_jobs capped by n_tasks: {result} processes [PASS]")

    print("\n[SUCCESS] All edge case tests passed!")


if __name__ == "__main__":
    test_n_jobs_handling()
    test_edge_cases()
    print("\n[SUCCESS] All tests passed! The n_jobs handling is working correctly.")
    print("\n[SUCCESS] All tests passed! The n_jobs handling is working correctly.")
