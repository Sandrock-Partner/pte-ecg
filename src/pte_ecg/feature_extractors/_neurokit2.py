import numpy as np
import pandas as pd
import scipy.signal
from neurokit2.ecg.ecg_delineate import (
    _dwt_ecg_delineator,
    _ecg_delineator_cwt,
    _ecg_delineator_peak,
    ecg_peaks,
    epochs_to_df,
)


def ecg_delineate(
    ecg_cleaned, rpeaks=None, sampling_rate=1000, method="dwt", show=False, show_type="peaks", check=False, **kwargs
):
    """**Delineate QRS complex**(MODIFIED FROM NEUROKIT2 version 0.2.12)

    This function is a modified version of the neurokit2.ecg.ecg_delineate function.
    It is modified to not remove NaN and None values, and to not return the signals dictionary.

    Function to delineate the QRS complex, i.e., the different waves of the cardiac cycles. A
    typical ECG heartbeat consists of a P wave, a QRS complex and a T wave. The P wave represents
    the wave of depolarization that spreads from the SA-node throughout the atria. The QRS complex
    reflects the rapid depolarization of the right and left ventricles. Since the ventricles are
    the largest part of the heart, in terms of mass, the QRS complex usually has a much larger
    amplitude than the P-wave. The T wave represents the ventricular repolarization of the
    ventricles.On rare occasions, a U wave can be seen following the T wave. The U wave is believed
    to be related to the last remnants of ventricular repolarization.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by ``ecg_clean()``.
    rpeaks : Union[list, np.array, pd.Series]
        The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info
        dictionary returned by ``ecg_findpeaks()``.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        Can be one of ``"peak"`` for a peak-based method, ``"prominence"`` for a peak-prominence-based
        method (Emrich et al., 2024), ``"cwt"`` for continuous wavelet transform or ``"dwt"`` (default)
        for discrete wavelet transform.
        The ``"prominence"`` method might be useful to detect the waves, allowing to set individual physiological
        limits (see kwargs), while the ``"dwt"`` method might be more precise for detecting the onsets and offsets
        of the waves (but might exhibit lower accuracy when there is significant variation in wave morphology).
        The ``"peak"`` method, which uses the zero-crossings of the signal derivatives, works best with very clean signals.
    show : bool
        If ``True``, will return a plot to visualizing the delineated waves information.
    show_type: str
        The type of delineated waves information showed in the plot.
        Can be ``"peaks"``, ``"bounds_R"``, ``"bounds_T"``, ``"bounds_P"`` or ``"all"``.
    check : bool
        Defaults to ``False``. If ``True``, replaces the delineated features with ``np.nan`` if its
        standardized distance from R-peaks is more than 3.
    **kwargs
        Other optional arguments:
        If using the ``"prominence"`` method, additional parameters (in milliseconds) can be passed to set
        individual physiological limits for the search boundaries:
        - ``max_qrs_interval``: The maximum allowable QRS complex interval. Defaults to 180 ms.
        - ``max_pr_interval``: The maximum PR interval duration. Defaults to 300 ms.
        - ``max_r_rise_time``: Maximum duration for the R-wave rise. Defaults to 120 ms.
        - ``typical_st_segment``: Typical duration of the ST segment. Defaults to 150 ms.
        - ``max_p_basepoint_interval``: The maximum interval between P-wave on- and offset. Defaults to 100 ms.
        - ``max_r_basepoint_interval``: The maximum interval between R-wave on- and offset. Defaults to 100 ms.
        - ``max_t_basepoint_interval``: The maximum interval between T-wave on- and offset. Defaults to 200 ms.

    Returns
    -------
    waves : dict
        A dictionary containing additional information.
        For derivative method, the dictionary contains the samples at which P-peaks, Q-peaks,
        S-peaks, T-peaks, P-onsets and T-offsets occur, accessible with the keys ``"ECG_P_Peaks"``,
        ``"ECG_Q_Peaks"``, ``"ECG_S_Peaks"``, ``"ECG_T_Peaks"``, ``"ECG_P_Onsets"``,
        ``"ECG_T_Offsets"``, respectively.

        For the wavelet and prominence methods, in addition to the above information, the dictionary contains the
        samples at which QRS-onsets and QRS-offsets occur, accessible with the key
        ``"ECG_P_Peaks"``, ``"ECG_T_Peaks"``, ``"ECG_P_Onsets"``, ``"ECG_P_Offsets"``,
        ``"ECG_Q_Peaks"``, ``"ECG_S_Peaks"``, ``"ECG_T_Onsets"``, ``"ECG_T_Offsets"``,
        ``"ECG_R_Onsets"``, ``"ECG_R_Offsets"``, respectively.

    See Also
    --------
    ecg_clean, .signal_fixpeaks, ecg_peaks, .signal_rate, ecg_process, ecg_plot

    Examples
    --------
    * Step 1. Delineate

    .. ipython:: python

      import neurokit2 as nk

      # Simulate ECG signal
      ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
      # Get R-peaks location
      _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=1000)
      # Delineate cardiac cycle
      waves = nk_ecg_delineate(ecg, rpeaks, sampling_rate=1000)

    * Step 2. Plot P-Peaks and T-Peaks

    .. ipython:: python

      @savefig p_ecg_delineate1.png scale=100%
      nk.events_plot([waves["ECG_P_Peaks"], waves["ECG_T_Peaks"]], ecg)
      @suppress
      plt.close()

    References
    --------------
    - Martínez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based
      ECG delineator: evaluation on standard databases. IEEE Transactions on biomedical engineering,
      51(4), 570-581.
    - Emrich, J., Gargano, A., Koka, T., & Muma, M. (2024). Physiology-Informed ECG Delineation Based
      on Peak Prominence. 32nd European Signal Processing Conference (EUSIPCO), 1402-1406.

    """

    # Sanitize input for ecg_cleaned
    if isinstance(ecg_cleaned, pd.DataFrame):
        cols = [col for col in ecg_cleaned.columns if "ECG_Clean" in col]
        if cols:
            ecg_cleaned = ecg_cleaned[cols[0]].values
        else:
            raise ValueError("NeuroKit error: ecg_delineate(): Wrong input, we couldn't extractcleaned signal.")

    elif isinstance(ecg_cleaned, dict):
        for i in ecg_cleaned:
            cols = [col for col in ecg_cleaned[i].columns if "ECG_Clean" in col]
            if cols:
                signals = epochs_to_df(ecg_cleaned)
                ecg_cleaned = signals[cols[0]].values

            else:
                raise ValueError("NeuroKit error: ecg_delineate(): Wrong input, we couldn't extractcleaned signal.")

    elif isinstance(ecg_cleaned, pd.Series):
        ecg_cleaned = ecg_cleaned.values

    # Sanitize input for rpeaks
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]

    if isinstance(rpeaks, dict):
        rpeaks = rpeaks["ECG_R_Peaks"]

    method = method.lower()  # remove capitalised letters
    if method in ["peak", "peaks", "derivative", "gradient"]:
        waves = _ecg_delineator_peak(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)
    elif method in ["cwt", "continuous wavelet transform"]:
        waves = _ecg_delineator_cwt(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)
    elif method in ["dwt", "discrete wavelet transform"]:
        waves = _dwt_ecg_delineator(
            ecg_cleaned, rpeaks, sampling_rate=sampling_rate, analysis_sampling_rate=sampling_rate
        )
    elif method in ["prominence", "peak-prominence", "emrich", "emrich2024"]:
        waves = _prominence_ecg_delineator(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, **kwargs)

    else:
        raise ValueError(
            "NeuroKit error: ecg_delineate(): 'method' should be one of 'peak', 'prominence','cwt' or 'dwt'."
        )

    for _, value in waves.items():
        if np.isnan(value[-1]):
            continue
        assert value[-1] < len(ecg_cleaned), (
            f"Wave index {value[-1]} is larger than ECG signal length {len(ecg_cleaned)}"
        )

    return waves


# =============================================================================
# PROMINENCE METHOD (Emrich et al., 2024)
# =============================================================================
def _prominence_ecg_delineator(ecg: np.ndarray, rpeaks: np.ndarray, sampling_rate: float, **kwargs):
    # pysiology-informed boundaries in milliseconds, adapt if needed
    max_qrs_interval = int(kwargs.get("max_qrs_interval", 180) * sampling_rate / 1000)
    max_pr_interval = int(kwargs.get("max_pr_interval", 300) * sampling_rate / 1000)
    max_r_rise_time = int(kwargs.get("max_r_rise_time", 120) * sampling_rate / 1000)
    typical_st_segment = int(kwargs.get("typical_st_segment", 150) * sampling_rate / 1000)
    # max basepoint intervals
    max_p_basepoint_interval = int(kwargs.get("max_p_basepoint_interval", 100) * sampling_rate / 1000)
    max_r_basepoint_interval = int(kwargs.get("max_r_basepoint_interval", 100) * sampling_rate / 1000)
    max_t_basepoint_interval = int(kwargs.get("max_t_basepoint_interval", 200) * sampling_rate / 1000)
    max_q_basepoint_interval = int(kwargs.get("max_q_basepoint_interval", 80) * sampling_rate / 1000)
    max_s_basepoint_interval = int(kwargs.get("max_s_basepoint_interval", 100) * sampling_rate / 1000)

    waves = {
        "ECG_P_Onsets": [],
        "ECG_P_Peaks": [],
        "ECG_P_Offsets": [],
        "ECG_Q_Onsets": [],
        "ECG_Q_Peaks": [],
        "ECG_Q_Offsets": [],
        "ECG_R_Onsets": [],
        "ECG_R_Offsets": [],
        "ECG_S_Onsets": [],
        "ECG_S_Peaks": [],
        "ECG_S_Offsets": [],
        "ECG_T_Onsets": [],
        "ECG_T_Peaks": [],
        "ECG_T_Offsets": [],
    }

    # calculate RR intervals
    rr = np.diff(rpeaks)
    rr = np.insert(rr, 0, min(rr[0], 2 * rpeaks[0]))
    rr = np.insert(rr, -1, min(rr[-1], 2 * rpeaks[-1]))

    # iterate over all beats
    left = 0
    for i in range(len(rpeaks)):
        # 1. split signal into segments
        rpeak_pos = min(rpeaks[i], rr[i] // 2)
        left = rpeaks[i] - rpeak_pos
        right = rpeaks[i] + rr[i + 1] // 2
        ecg_seg = ecg[left:right]

        current_wave = {
            "ECG_R_Peaks": rpeak_pos,
        }

        # 2. find local extrema in signal
        local_maxima = scipy.signal.find_peaks(ecg_seg)[0]
        local_minima = scipy.signal.find_peaks(-ecg_seg)[0]
        local_extrema = np.concatenate((local_maxima, local_minima))

        # 3. compute prominence weight
        weight_maxima = _calc_prominence(local_maxima, ecg_seg, current_wave["ECG_R_Peaks"])
        weight_minima = _calc_prominence(local_minima, ecg_seg, current_wave["ECG_R_Peaks"], minima=True)

        if local_extrema.any():
            # find waves
            _prominence_find_q_wave(weight_minima, current_wave, max_r_rise_time)
            _prominence_find_s_wave(ecg_seg, weight_minima, current_wave, max_qrs_interval)
            _prominence_find_p_wave(local_maxima, weight_maxima, current_wave, max_pr_interval)
            _prominence_find_t_wave(local_extrema, (weight_minima + weight_maxima), current_wave, typical_st_segment)
            _prominence_find_on_offsets(
                ecg_seg,
                sampling_rate,
                local_minima,
                current_wave,
                max_p_basepoint_interval,
                max_r_basepoint_interval,
                max_t_basepoint_interval,
                max_q_basepoint_interval,
                max_s_basepoint_interval,
            )

        # append waves for current beat / complex
        for key in waves:
            if key == "ECG_R_Peaks":
                waves[key].append(int(rpeaks[i]))
            elif key in current_wave:
                waves[key].append(int(current_wave[key] + left))
            else:
                waves[key].append(np.nan)

    return waves


def _calc_prominence(peaks, sig, Rpeak=None, minima=False):
    """Returns an array of the same length as sig with prominences computed for the provided peaks.

    Prominence of the R-peak is excluded if the R-peak position is given.

    """
    w = np.zeros_like(sig)

    if len(peaks) < 1:
        return w
    # get prominence
    _sig = -sig if minima else sig
    w[peaks] = scipy.signal.peak_prominences(_sig, peaks)[0]
    # optional: set rpeak prominence to zero to emphasize other peaks
    if Rpeak is not None:
        w[Rpeak] = 0
    return w


def _prominence_find_q_wave(weight_minima, current_wave, max_r_rise_time):
    if "ECG_R_Peaks" not in current_wave:
        return
    q_bound = max(current_wave["ECG_R_Peaks"] - max_r_rise_time, 0)

    current_wave["ECG_Q_Peaks"] = np.argmax(weight_minima[q_bound : current_wave["ECG_R_Peaks"]]) + q_bound


def _prominence_find_s_wave(sig, weight_minima, current_wave, max_qrs_interval):
    if "ECG_Q_Peaks" not in current_wave:
        return
    s_bound = current_wave["ECG_Q_Peaks"] + max_qrs_interval
    s_wave = np.argmax(weight_minima[current_wave["ECG_R_Peaks"] : s_bound] > 0) + current_wave["ECG_R_Peaks"]
    current_wave["ECG_S_Peaks"] = (
        np.argmin(sig[current_wave["ECG_R_Peaks"] : s_bound]) + current_wave["ECG_R_Peaks"]
        if s_wave == current_wave["ECG_R_Peaks"]
        else s_wave
    )


def _prominence_find_p_wave(local_maxima, weight_maxima, current_wave, max_pr_interval):
    if "ECG_Q_Peaks" not in current_wave:
        return
    p_candidates = local_maxima[
        (current_wave["ECG_Q_Peaks"] - max_pr_interval <= local_maxima) & (local_maxima <= current_wave["ECG_Q_Peaks"])
    ]
    if p_candidates.any():
        current_wave["ECG_P_Peaks"] = p_candidates[np.argmax(weight_maxima[p_candidates])]


def _prominence_find_t_wave(local_extrema, weight_extrema, current_wave, typical_st_segment):
    if "ECG_S_Peaks" not in current_wave:
        return
    t_candidates = local_extrema[(current_wave["ECG_S_Peaks"] + typical_st_segment <= local_extrema)]
    if t_candidates.any():
        current_wave["ECG_T_Peaks"] = t_candidates[np.argmax(weight_extrema[t_candidates])]


def _prominence_find_on_offsets(
    sig,
    sampling_rate,
    local_minima,
    current_wave,
    max_p_basepoint_interval,
    max_r_basepoint_interval,
    max_t_basepoint_interval,
    max_q_basepoint_interval,
    max_s_basepoint_interval,
):
    if "ECG_P_Peaks" in current_wave:
        _, p_on, p_off = scipy.signal.peak_prominences(
            sig, [current_wave["ECG_P_Peaks"]], wlen=max_p_basepoint_interval
        )
        if not np.isnan(p_on):
            current_wave["ECG_P_Onsets"] = p_on[0]
        if not np.isnan(p_off):
            current_wave["ECG_P_Offsets"] = p_off[0]

    if "ECG_T_Peaks" in current_wave:
        p = -1 if np.isin(current_wave["ECG_T_Peaks"], local_minima) else 1

        _, t_on, t_off = scipy.signal.peak_prominences(
            p * sig, [current_wave["ECG_T_Peaks"]], wlen=max_t_basepoint_interval
        )
        if not np.isnan(t_on):
            current_wave["ECG_T_Onsets"] = t_on[0]
        if not np.isnan(t_off):
            current_wave["ECG_T_Offsets"] = t_off[0]

    if "ECG_Q_Peaks" in current_wave:
        p = -1 if np.isin(current_wave["ECG_Q_Peaks"], local_minima) else 1

        _, q_on, q_off = scipy.signal.peak_prominences(
            p * sig, [current_wave["ECG_Q_Peaks"]], wlen=max_q_basepoint_interval
        )
        if not np.isnan(q_on):
            current_wave["ECG_Q_Onsets"] = q_on[0]
        if not np.isnan(q_off):
            current_wave["ECG_Q_Offsets"] = q_off[0]

    if "ECG_S_Peaks" in current_wave:
        p = -1 if np.isin(current_wave["ECG_S_Peaks"], local_minima) else 1
        _, s_on, s_off = scipy.signal.peak_prominences(
            p * sig, [current_wave["ECG_S_Peaks"]], wlen=max_s_basepoint_interval
        )
        if not np.isnan(s_on):
            current_wave["ECG_S_Onsets"] = s_on[0]
        if not np.isnan(s_off):
            current_wave["ECG_S_Offsets"] = s_off[0]

    # correct R-peak position towards local maxima (otherwise prominence will be falsely computed)
    r_pos = _correct_peak(sig, sampling_rate, current_wave["ECG_R_Peaks"])
    _, r_on, r_off = scipy.signal.peak_prominences(sig, [r_pos], wlen=max_r_basepoint_interval)
    if not np.isnan(r_on):
        current_wave["ECG_R_Onsets"] = r_on[0]

    if not np.isnan(r_off):
        current_wave["ECG_R_Offsets"] = r_off[0]


def _correct_peak(sig, fs, peak, window=0.02):
    """Correct peak towards local maxima within provided window."""

    left = peak - int(window * fs)
    right = peak + int(window * fs)
    if len(sig[left:right]) > 0:
        return np.argmax(sig[left:right]) + left
    else:
        return peak
