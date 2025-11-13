"""Type definitions for ECG data structures."""

from typing import Annotated, TypeAlias

import numpy as np
import numpy.typing as npt

# Type alias for ECG data array
# Shape: (n_ecgs, n_channels, n_timepoints)
#   - n_ecgs: Number of ECG recordings/samples
#   - n_channels: Number of leads/channels (must match lead_order length)
#   - n_timepoints: Number of time points
ECGData: TypeAlias = Annotated[
    npt.NDArray[np.floating],
    "Shape: (n_ecgs, n_channels, n_timepoints)",
]

