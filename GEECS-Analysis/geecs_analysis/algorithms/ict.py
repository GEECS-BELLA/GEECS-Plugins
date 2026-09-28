"""ICT (integrating current transformer) charge from an oscilloscope trace.

Ported unchanged from ImageAnalysis' ``algorithms.ict_algorithms``
(``apply_ict_analysis`` and its helpers; the Butterworth low-pass is the
same ``scipy.signal.butter`` + ``filtfilt`` pair ImageAnalysis'
``apply_butterworth_filter`` wraps), so the charge is the legacy
``ICT1DAnalyzer``'s bit for bit — pinned by a differential test that uses
ImageAnalysis as the oracle. The 8 steps: low-pass filter; locate the pulse
in the raw trace; fit and subtract a sinusoidal background twice, outside
the pulse window; find the pulse's zero crossings in the cleaned trace;
integrate and calibrate to pC.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def identify_primary_valley(data: np.ndarray) -> np.ndarray:
    """Identify signal region by finding zero crossings around minimum.

    Finds the primary (largest negative) peak in the data and identifies
    the region where the signal crosses zero before and after the peak.

    Parameters
    ----------
    data : np.ndarray
        Input signal array (expected to have negative pulse)

    Returns
    -------
    np.ndarray
        Array of indices corresponding to the signal region
    """
    try:
        # Find minimum value (largest negative peak)
        min_ind = np.argmin(data)

        # Find where signal goes to zero before spike
        count = 1
        test_val = data[min_ind]
        try:
            while test_val < 0:
                test_ind = min_ind - count
                test_val = data[test_ind]
                count += 1
            valley_min = int(test_ind + 1)
        except (IndexError, ValueError):
            valley_min = 0

        # Find where signal goes to zero after spike
        count = 1
        test_val = data[min_ind]
        try:
            while test_val < 0:
                test_ind = min_ind + count
                test_val = data[test_ind]
                count += 1
            valley_max = int(test_ind)
        except (IndexError, ValueError):
            valley_max = len(data)

        # Return array of indices corresponding to signal region
        valley_ind = np.arange(valley_min, valley_max)

        return valley_ind
    except Exception as e:
        logger.error(f"Valley identification failed: {e}")
        raise


def get_sinusoidal_noise(
    data: np.ndarray, signal_region: Tuple[Optional[int], Optional[int]]
) -> np.ndarray:
    """Fit and return sinusoidal background noise.

    Fits a sinusoidal function to the noise regions (excluding the signal region)
    using FFT to estimate frequency and curve_fit to optimize amplitude, phase, and offset.

    Replicates the working implementation from picoscope_ICT_analysis_RJ.py faithfully.

    Parameters
    ----------
    data : np.ndarray
        Full signal array
    signal_region : tuple of (int or None, int or None)
        (start_index, end_index) of signal region to exclude from fit.
        If start is None, fit from end onwards.
        If end is None, fit up to start.
        If both None, fit entire signal.

    Returns
    -------
    np.ndarray
        Sinusoidal fit for the full data range
    """
    try:
        x_axis = np.arange(len(data))
        p1 = signal_region[0]
        p2 = signal_region[1]

        # Extract noise regions (exclude signal region)
        if p1 is None and p2 is not None:
            bg_data = data[p2:]
            bg_axis = x_axis[p2:]
        elif p2 is None and p1 is not None:
            bg_data = data[:p1]
            bg_axis = x_axis[:p1]
        elif p1 is None and p2 is None:
            bg_data = data
            bg_axis = x_axis
        else:
            # Combine regions before and after signal
            bg_data = np.concatenate((data[:p1], data[p2:]))
            bg_axis = np.concatenate((x_axis[:p1], x_axis[p2:]))

        if len(bg_data) < 3:
            # Not enough data to fit
            return np.zeros_like(data)

        # Define sinusoidal model with 4 parameters (amplitude, frequency, phase, offset)
        def sin_model(t, amplitude, frequency, phase, offset):
            return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

        # Use FFT to estimate dominant frequency with proper spacing
        fft_data = np.fft.rfft(bg_data)
        fft_axis = np.fft.rfftfreq(
            len(bg_data), d=(bg_axis[1] - bg_axis[0]) if len(bg_axis) > 1 else 1
        )

        # Find dominant frequency (excluding DC component)
        if len(fft_data) > 2:
            dominant_freq_idx = np.argmax(np.abs(fft_data)[1:]) + 1
        else:
            dominant_freq_idx = np.argmax(np.abs(fft_data))

        initial_freq = fft_axis[dominant_freq_idx]

        # Intelligent phase estimation based on initial data values
        ave_val = np.mean(bg_data)
        std_val = np.std(bg_data)

        if bg_data[0] > ave_val + std_val:
            phi_est = np.pi / 2
        elif bg_data[0] < ave_val - std_val:
            phi_est = -np.pi / 2
        elif len(bg_data) > 100 and bg_data[100] > bg_data[0]:
            phi_est = 0
        else:
            phi_est = np.pi

        # Initial parameter estimates
        p0 = [std_val, initial_freq, phi_est, np.mean(bg_data)]

        # Fit sinusoid to background data using downsampled data [::4]
        try:
            params, _ = curve_fit(
                sin_model, bg_axis[::4], bg_data[::4], p0=p0, maxfev=10000
            )
        except RuntimeError:
            # Fit failed, return zeros
            return np.zeros_like(data)

        # Generate sinusoidal fit for full data range
        background_model = sin_model(x_axis, *params)

        return background_model
    except Exception as e:
        logger.error(f"Sinusoidal noise fitting failed: {e}")
        return np.zeros_like(data)


def apply_ict_analysis(
    data: np.ndarray,
    dt: float,
    butterworth_order: int = 1,
    butterworth_crit_f: float = 0.125,
    calibration_factor: float = 0.1,
) -> Tuple[float, float]:
    """Complete ICT analysis pipeline returning charge and peak time.

    Implements the 8-step ICT signal processing pipeline:
    1. Apply Butterworth low-pass filter
    2. Identify signal region (primary valley) in RAW data
    3. Fit sinusoidal background (pass 1)
    4. Subtract sinusoidal background
    5. Fit sinusoidal background (pass 2)
    6. Subtract sinusoidal background again
    7. Identify signal region in cleaned data
    8. Integrate and calibrate to get charge in pC

    Replicates the working implementation from picoscope_ICT_analysis_RJ.py faithfully.

    Parameters
    ----------
    data : np.ndarray
        Input voltage trace from oscilloscope (Volts)
    dt : float
        Time step between samples (seconds)
    butterworth_order : int, default=1
        Butterworth filter order
    butterworth_crit_f : float, default=0.125
        Normalized critical frequency for Butterworth filter
    calibration_factor : float, default=0.1
        Calibration factor in V·s/C (volts·seconds per coulomb)

    Returns
    -------
    tuple of (float, float)
        - charge_pC: Charge in picocoulombs (pC)
        - peak_time_us: Time of signal peak in microseconds (µs)

    Raises
    ------
    ValueError
        If input data is invalid or processing fails
    """
    try:
        # Step 1: Apply Butterworth filter
        b, a = signal.butter(butterworth_order, butterworth_crit_f, "low")
        value = np.array(signal.filtfilt(b, a, data))

        # Step 2: Identify signal location in RAW data (not filtered)
        # Use fixed offsets (±100 and +600 samples) from the minimum
        signal_location = np.argmin(data)
        first_interval_end = signal_location - 100 if signal_location > 100 else None
        second_interval_start = (
            signal_location + 600 if signal_location + 600 < len(value) else None
        )
        signal_region = (first_interval_end, second_interval_start)

        logger.debug(
            f"Signal location: {signal_location}, "
            f"Signal region for sinusoid fitting: ({first_interval_end}, {second_interval_start})"
        )

        # Step 3-4: Fit and subtract sinusoidal background (pass 1)
        value -= get_sinusoidal_noise(data=value, signal_region=signal_region)

        # Step 5-6: Fit and subtract sinusoidal background (pass 2)
        value -= get_sinusoidal_noise(data=value, signal_region=signal_region)

        # Step 7: Identify signal region in cleaned data using zero-crossing detection
        signal_region_indices_clean = identify_primary_valley(value)
        if len(signal_region_indices_clean) == 0:
            logger.warning("No signal region in cleaned data, returning 0 pC")
            return 0.0, 0.0

        # Step 8: Integrate and calibrate
        signal_data = np.array(value[signal_region_indices_clean])
        integrated_signal = np.trapezoid(signal_data, x=None, dx=dt)
        charge_pC = integrated_signal * (-calibration_factor) * 1e12

        # Calculate peak time in microseconds
        peak_time_us = signal_location * dt * 1e6

        logger.debug(
            f"ICT Analysis: integrated_signal={integrated_signal:.6e}, "
            f"charge={charge_pC:.2f} pC, peak_time={peak_time_us:.3f} µs"
        )

        return float(charge_pC), float(peak_time_us)

    except Exception as e:
        logger.error(f"ICT analysis failed: {e}")
        raise ValueError(f"ICT analysis pipeline failed: {e}") from e
