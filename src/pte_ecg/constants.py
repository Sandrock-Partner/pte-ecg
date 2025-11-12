"""Constants for ECG processing."""

# Standard 12-lead ECG lead names in the default order
STANDARD_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

# Allowed lead names (for validation)
ALLOWED_LEAD_NAMES = set(STANDARD_LEAD_NAMES)

