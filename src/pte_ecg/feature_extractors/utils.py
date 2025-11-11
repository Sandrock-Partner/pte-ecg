"""Shared utility functions for feature extractors.

This module provides common functionality used across multiple feature extractors,
including CPU management, logging helpers, and input validation.
"""

import os
import sys
import time

import numpy as np

from .._logging import logger

# Small constant for numerical stability
EPS = 1e-10


def assert_3_dims(ecg_data: np.ndarray) -> None:
    """Ensure the input array has 3 dimensions.

    Args:
        ecg_data: Input array to check.

    Raises:
        ValueError: If input array doesn't have exactly 3 dimensions.
    """
    if ecg_data.ndim != 3:
        raise ValueError("ECG data must be 3D (n_samples, n_channels, n_timepoints)")


def get_n_processes(n_jobs: int | None, n_tasks: int) -> int:
    """Get the number of processes to use for parallel processing.

    Args:
        n_jobs: Number of parallel jobs to run.
                - None or -1: Use all available CPUs
                - Positive int: Use exactly that many CPUs
                - Negative int (< -1): Use (total_cpus + n_jobs + 1) CPUs
        n_tasks: Number of tasks to process (used to cap the number of processes)

    Returns:
        Number of processes to use, capped by n_tasks
    """
    # Get total number of CPUs
    if sys.version_info >= (3, 13):
        total_cpus = os.process_cpu_count()
        logger.debug(f"Using os.process_cpu_count: {total_cpus} CPUs available")
    else:
        total_cpus = os.cpu_count()
        logger.debug(f"Using os.cpu_count: {total_cpus} CPUs available")

    # Handle None or fallback if cpu_count returns None
    if total_cpus is None:
        logger.warning("Could not determine CPU count, defaulting to 1")
        total_cpus = 1

    # Handle different n_jobs values
    if n_jobs is None or n_jobs == -1:
        # Use all available CPUs
        n_processes = total_cpus
    elif n_jobs > 0:
        # Use exactly n_jobs CPUs
        n_processes = n_jobs
    elif n_jobs < -1:
        # Use (total_cpus + n_jobs + 1) CPUs
        # e.g., n_jobs=-2 means use all CPUs except 1
        n_processes = max(1, total_cpus + n_jobs + 1)
    else:
        # n_jobs == 0, which doesn't make sense, default to 1
        logger.warning(f"Invalid n_jobs value: {n_jobs}, defaulting to 1")
        n_processes = 1

    # Cap by number of tasks (no point using more processes than tasks)
    n_processes = min(n_processes, n_tasks)

    # Ensure at least 1 process
    n_processes = max(1, n_processes)

    logger.debug(f"Using {n_processes} processes for {n_tasks} tasks")
    return n_processes


def log_start(feature_name: str, n_samples: int) -> float:
    """Log the start of feature extraction and return the current time.

    Args:
        feature_name: Name of the feature type being extracted.
        n_samples: Number of samples.

    Returns:
        Current time.
    """
    logger.info(
        "Starting %s feature extraction for %s samples...", feature_name, n_samples
    )
    return time.time()


def log_end(feature_name: str, start_time: float, shape: tuple[int, int]) -> None:
    """Log the end of feature extraction.

    Args:
        feature_name: Name of the feature type being extracted.
        start_time: Start time of the feature extraction.
        shape: Shape of the extracted features.
    """
    logger.info(
        "Completed %s feature extraction. Shape: %s. Time taken: %.1f s",
        feature_name,
        shape,
        time.time() - start_time,
    )
