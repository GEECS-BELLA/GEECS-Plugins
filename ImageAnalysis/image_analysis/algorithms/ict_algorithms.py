"""ICT (Integrated Current Transformer) signal processing algorithms.

This module provides core signal processing functions for ICT charge measurements
from oscilloscope traces. The algorithms implement an 8-step processing pipeline:

1. Apply Butterworth low-pass filter
2. Identify signal region (primary valley)
3. Fit sinusoidal background (pass 1)
4. Subtract sinusoidal background
5. Fit sinusoidal background (pass 2)
6. Subtract sinusoidal background again
7. Identify signal region in cleaned data
8. Integrate and calibrate to get charge in pC

The module follows NumPy docstring conventions and returns pure float values
for integration with the ICT1DAnalyzer.
"""

from __future__ import annotations

from typing import Optional, Tuple
import logging

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import curve_fit

from image_analysis.processing.array1d.filtering import apply_butterworth_filter

logger = logging.getLogger(__name__)


class ICTAnalysisConfig(BaseModel):
    """Typed configuration for :class:`ICT1DAnalyzer`.

    Validated from ``line_config.analysis`` at analyzer init time.
    Default values match the algorithm function signature so that a
    bare ``analysis: {}`` in the YAML config is equivalent to the
    previous hard-coded defaults.

    Attributes
    ----------
    butterworth_order : int
        Butterworth low-pass filter order.
    butterworth_crit_f : float
        Normalised critical frequency for the Butterworth filter.
    calibration_factor : float
        Calibration factor in V·s/C.
    dt : float or None
        Time step override in seconds. ``None`` means derive from data.
    """

    butterworth_order: int = Field(1, description="Butterworth filter order")
    butterworth_crit_f: float = Field(
        0.125, description="Normalised critical frequency"
    )
    calibration_factor: float = Field(0.1, description="Calibration factor [V·s/C]")
    dt: Optional[float] = Field(
        None, description="Time step override [s]; None = derive from data"
    )


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
) -> float:
    """Complete ICT analysis pipeline returning charge in picocoulombs.

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
    float
        Charge in picocoulombs (pC)

    Raises
    ------
    ValueError
        If input data is invalid or processing fails
    """
    try:
        # Step 1: Apply Butterworth filter
        value = np.array(
            apply_butterworth_filter(
                data,
                order=butterworth_order,
                crit_f=butterworth_crit_f,
                filt_type="low",
            )
        )

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
            return 0.0

        # Step 8: Integrate and calibrate
        signal_data = np.array(value[signal_region_indices_clean])
        integrated_signal = np.trapz(signal_data, x=None, dx=dt)
        charge_pC = integrated_signal * (-calibration_factor) * 1e12

        logger.debug(
            f"ICT Analysis: integrated_signal={integrated_signal:.6e}, "
            f"charge={charge_pC:.2f} pC"
        )

        return float(charge_pC)

    except Exception as e:
        logger.error(f"ICT analysis failed: {e}")
        raise ValueError(f"ICT analysis pipeline failed: {e}") from e


def apply_ict_analysis_with_details(
    data: np.ndarray,
    dt: float,
    butterworth_order: int = 1,
    butterworth_crit_f: float = 0.125,
    calibration_factor: float = 0.1,
) -> Tuple[float, dict]:
    """Complete ICT analysis pipeline with intermediate processing details.

    Implements the 8-step ICT signal processing pipeline and returns both
    the final charge value and intermediate processing data for visualization.

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
    tuple of (float, dict)
        - charge_pC: Charge in picocoulombs
        - details: Dictionary containing intermediate processing data:
            - 'raw_data': Original input data
            - 'filtered_data': After Butterworth filter
            - 'sinusoidal_bg_1': First sinusoidal background fit
            - 'after_sub1': After first background subtraction
            - 'sinusoidal_bg_2': Second sinusoidal background fit
            - 'cleaned_data': Final cleaned data (after both subtractions)
            - 'signal_region_indices': Indices of integrated region
            - 'signal_region_start': Start index of integration region
            - 'signal_region_end': End index of integration region
            - 'integrated_signal': Integrated signal value
            - 'charge_pC': Final charge in picocoulombs

    Raises
    ------
    ValueError
        If input data is invalid or processing fails
    """
    try:
        # Step 1: Apply Butterworth filter
        filtered_data = np.array(
            apply_butterworth_filter(
                data,
                order=butterworth_order,
                crit_f=butterworth_crit_f,
                filt_type="low",
            )
        )

        # Step 2: Identify signal location in RAW data (not filtered)
        signal_location = np.argmin(data)
        first_interval_end = signal_location - 100 if signal_location > 100 else None
        second_interval_start = (
            signal_location + 600
            if signal_location + 600 < len(filtered_data)
            else None
        )
        signal_region = (first_interval_end, second_interval_start)

        logger.debug(
            f"Signal location: {signal_location}, "
            f"Signal region for sinusoid fitting: ({first_interval_end}, {second_interval_start})"
        )

        # Step 3: Fit sinusoidal background (pass 1)
        sinusoidal_background_1 = get_sinusoidal_noise(filtered_data, signal_region)

        # Step 4: Subtract sinusoidal background
        value_after_sub1 = filtered_data - sinusoidal_background_1

        # Step 5: Fit sinusoidal background (pass 2)
        sinusoidal_background_2 = get_sinusoidal_noise(value_after_sub1, signal_region)

        # Step 6: Subtract sinusoidal background again
        subtracted_value = value_after_sub1 - sinusoidal_background_2

        # Step 7: Identify signal region in cleaned data using zero-crossing detection
        signal_region_indices_clean = identify_primary_valley(subtracted_value)
        if len(signal_region_indices_clean) == 0:
            logger.warning("No signal region in cleaned data, returning 0 pC")
            charge_pC = 0.0
            signal_region_start = 0
            signal_region_end = 0
            integrated_signal = 0.0
        else:
            # Step 8: Integrate and calibrate
            signal_data = np.array(subtracted_value[signal_region_indices_clean])
            integrated_signal = np.trapz(signal_data, x=None, dx=dt)
            charge_pC = integrated_signal * (-calibration_factor) * 1e12
            signal_region_start = signal_region_indices_clean[0]
            signal_region_end = signal_region_indices_clean[-1]

        logger.debug(
            f"ICT Analysis: integrated_signal={integrated_signal:.6e}, "
            f"charge={charge_pC:.2f} pC"
        )

        # Build details dictionary
        details = {
            "raw_data": data.copy(),
            "filtered_data": filtered_data.copy(),
            "sinusoidal_bg_1": sinusoidal_background_1.copy(),
            "after_sub1": value_after_sub1.copy(),
            "sinusoidal_bg_2": sinusoidal_background_2.copy(),
            "cleaned_data": subtracted_value.copy(),
            "signal_region_indices": signal_region_indices_clean.copy()
            if len(signal_region_indices_clean) > 0
            else np.array([]),
            "signal_region_start": int(signal_region_start),
            "signal_region_end": int(signal_region_end),
            "integrated_signal": float(integrated_signal),
            "charge_pC": float(charge_pC),
        }

        return float(charge_pC), details

    except Exception as e:
        logger.error(f"ICT analysis with details failed: {e}")
        raise ValueError(f"ICT analysis pipeline failed: {e}") from e
