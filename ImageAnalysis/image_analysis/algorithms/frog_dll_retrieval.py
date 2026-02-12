r"""FROG DLL Retrieval — Python wrapper for Kane's FROG.dll.

This module provides a high-level Python interface to the FROG pulse retrieval
algorithm implemented in FROG.dll (Daniel Kane, Mesa Photonics / Swamp Optics).

Because the DLL is 32-bit and the project typically runs in 64-bit Python,
this module uses a subprocess call to a standalone worker script
(_frog_dll_worker.py) that runs under a 32-bit Python interpreter.

Requirements:
    - Windows OS (the DLL is a Windows shared library)
    - 32-bit Python interpreter (embeddable package is sufficient)
    - FROG.dll file
    - Paths configured in ~/.config/geecs_python_api/config.ini:
        [Paths]
        frog_dll_path = D:\\path\\to\\FROG.dll
        frog_python32_path = D:\\path\\to\\python32\\python.exe
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Path to the worker script, which lives next to this file
_WORKER_SCRIPT = Path(__file__).parent / "_frog_dll_worker.py"


@dataclass
class FrogRetrievalResult:
    """Result of a FROG pulse retrieval.

    Contains the complete output from the FROG DLL: E-field, FROG traces,
    spectral/temporal profiles, and scalar metrics.

    Attributes
    ----------
    E_real : NDArray[np.float64]
        Real part of the retrieved E-field envelope, length N.
    E_imag : NDArray[np.float64]
        Imaginary part of the retrieved E-field envelope, length N.
    frog_error : float
        Best FROG error (dimensionless, typically < 0.01 for good retrieval).
    num_iterations : int
        Number of iterations performed.
    N : int
        Grid size used by the DLL (e.g. 512, 256, 128, 64).
    temporal_fwhm : float
        Temporal FWHM of the retrieved pulse.
    spectral_fwhm : float
        Spectral FWHM of the retrieved pulse.
    input_trace : NDArray[np.float64]
        Gridded input FROG trace, shape (N, N).
    retrieved_trace : NDArray[np.float64]
        Retrieved (simulated) FROG trace, shape (N, N).
    wavelength : NDArray[np.float64]
        Wavelength axis for spectral data, length wave_n.
    spectral_intensity : NDArray[np.float64]
        Spectral intensity vs wavelength, length wave_n.
    spectral_phase : NDArray[np.float64]
        Spectral phase vs wavelength (radians), length wave_n.
    time : NDArray[np.float64]
        Time axis for temporal data, length time_n.
    temporal_intensity : NDArray[np.float64]
        Temporal intensity vs time, length time_n.
    temporal_phase : NDArray[np.float64]
        Temporal phase vs time (radians), length time_n.
    """

    # E-field
    E_real: NDArray[np.float64]
    E_imag: NDArray[np.float64]

    # Scalars
    frog_error: float
    num_iterations: int
    N: int
    temporal_fwhm: float
    spectral_fwhm: float

    # 2D traces
    input_trace: NDArray[np.float64]
    retrieved_trace: NDArray[np.float64]

    # Spectral data
    wavelength: NDArray[np.float64]
    spectral_intensity: NDArray[np.float64]
    spectral_phase: NDArray[np.float64]

    # Temporal data
    time: NDArray[np.float64]
    temporal_intensity: NDArray[np.float64]
    temporal_phase: NDArray[np.float64]

    @property
    def E_complex(self) -> NDArray[np.complex128]:
        """Complex E-field envelope."""
        return self.E_real + 1j * self.E_imag

    @property
    def amplitude(self) -> NDArray[np.float64]:
        """E-field amplitude |E(t)|."""
        return np.abs(self.E_complex)

    @property
    def phase(self) -> NDArray[np.float64]:
        """E-field phase angle(E(t)) in radians."""
        return np.angle(self.E_complex)

    @property
    def intensity(self) -> NDArray[np.float64]:
        """E-field intensity |E(t)|^2."""
        return np.square(self.amplitude)


class FrogDllRetrieval:
    """High-level interface to the FROG DLL pulse retrieval algorithm.

    This class manages the subprocess communication with the 32-bit FROG.dll
    worker script, handling serialization of input data and deserialization
    of results.

    Parameters
    ----------
    dll_path : str or Path
        Path to FROG.dll.
    python32_path : str or Path
        Path to a 32-bit Python interpreter (e.g., embeddable package).

    Raises
    ------
    FileNotFoundError
        If dll_path or python32_path do not exist.
    """

    # FROG geometry constants
    GEOMETRY_SHG = 1
    GEOMETRY_PG = 2
    GEOMETRY_SD = 3

    # Run type constants
    RUN_TYPE_THEORY = 0
    RUN_TYPE_EXPERIMENT = 1
    RUN_TYPE_READIN = 2

    # Valid grid sizes
    VALID_GRID_SIZES = {512, 256, 128, 64}

    # Binning factors: grid_size → time-axis binning
    GRID_BINNING = {512: 3, 256: 5, 128: 7, 64: 9}

    def __init__(
        self,
        dll_path: str | Path,
        python32_path: str | Path,
    ):
        self.dll_path = Path(dll_path)
        self.python32_path = Path(python32_path)

        if not self.dll_path.exists():
            raise FileNotFoundError(
                f"FROG.dll not found at: {self.dll_path}\n"
                "Ensure the path is configured correctly in your config.ini:\n"
                "  [Paths]\n"
                "  frog_dll_path = D:\\path\\to\\FROG.dll"
            )

        if not self.python32_path.exists():
            raise FileNotFoundError(
                f"32-bit Python not found at: {self.python32_path}\n"
                "Download the Python 3.10 32-bit embeddable package from:\n"
                "  https://www.python.org/ftp/python/3.10.11/"
                "python-3.10.11-embed-win32.zip\n"
                "Extract it and configure the path in your config.ini:\n"
                "  [Paths]\n"
                "  frog_python32_path = D:\\path\\to\\python32\\python.exe"
            )

    def retrieve_pulse(
        self,
        trace: NDArray[np.float64],
        delt: float,
        dellam: float,
        lam0: float,
        N: int = 512,
        geometry: int = GEOMETRY_SHG,
        run_type: int = RUN_TYPE_EXPERIMENT,
        max_iterations: int = 200,
        target_error: float = 0.005,
        max_time_seconds: float = 60.0,
        timeout: float = 120.0,
    ) -> FrogRetrievalResult:
        """Run the FROG retrieval algorithm on a Grenouille trace.

        Parameters
        ----------
        trace : NDArray[np.float64]
            2D array of shape (ntau, nw) — the raw Grenouille camera image.
            Time delay on axis 0 (rows), wavelength on axis 1 (columns).
            For the standard Grenouille camera this is (576, 768).
            The time axis will be binned automatically based on grid size N.
        delt : float
            Time delay step per raw pixel in femtoseconds (before binning).
        dellam : float
            Wavelength step in nanometers (may be negative per instrument convention).
        lam0 : float
            Center wavelength of the trace in nanometers.
        N : int
            Grid size for the DLL: 512, 256, 128, or 64. The DLL internally
            resamples the trace onto an N×N grid.
        geometry : int
            FROG geometry: 1=SHG, 2=PG, 3=SD (default 1 for Grenouille/SHG).
        run_type : int
            Run type: 0=THEORY, 1=EXPERIMENT, 2=READIN (default 1).
        max_iterations : int
            Maximum number of retrieval iterations (default 200).
        target_error : float
            Target FROG error for early stopping (default 0.005).
        max_time_seconds : float
            Maximum wall-clock time for the retrieval loop in seconds (default 60).
        timeout : float
            Maximum time for the entire subprocess in seconds (default 120).

        Returns
        -------
        FrogRetrievalResult
            Dataclass containing the complete retrieval output.

        Raises
        ------
        RuntimeError
            If the DLL subprocess fails or times out.
        ValueError
            If the trace array is not 2D or N is not a valid grid size.
        """
        trace = np.asarray(trace, dtype=np.float64)
        if trace.ndim != 2:
            raise ValueError(
                f"Expected 2D trace array, got {trace.ndim}D with shape {trace.shape}"
            )

        if N not in self.VALID_GRID_SIZES:
            raise ValueError(
                f"Grid size N must be one of {self.VALID_GRID_SIZES}, got {N}"
            )

        # Pre-bin the time axis (axis 0 = rows = time delay)
        # This matches LabVIEW's preprocessing before GridData.
        bin_factor = self.GRID_BINNING[N]
        n_rows = trace.shape[0]
        n_binned_rows = n_rows // bin_factor
        # Trim rows to be evenly divisible by bin_factor, then average
        trimmed = trace[: n_binned_rows * bin_factor, :]
        trace = trimmed.reshape(n_binned_rows, bin_factor, -1).mean(axis=1)

        # Scale delt by the bin factor (each binned pixel spans bin_factor
        # raw pixels in time)
        delt = delt * bin_factor

        # After binning: axis 0 = time (ntau), axis 1 = wavelength (nw)
        ntau, nw = trace.shape
        logger.info(
            "Starting FROG DLL retrieval: binned trace shape (%d, %d), "
            "bin_factor=%d, delt=%.4f fs, dellam=%.4f nm, lam0=%.2f nm, N=%d",
            ntau,
            nw,
            bin_factor,
            delt,
            dellam,
            lam0,
            N,
        )

        # Create temp files for IPC
        with (
            tempfile.NamedTemporaryFile(
                suffix=".bin", delete=False, prefix="frog_input_"
            ) as f_in,
            tempfile.NamedTemporaryFile(
                suffix=".bin", delete=False, prefix="frog_output_"
            ) as f_out,
            tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, prefix="frog_params_", mode="w"
            ) as f_params,
        ):
            input_path = f_in.name
            output_path = f_out.name
            params_path = f_params.name

            # Write input binary: [nw, ntau, trace_data...]
            f_in.write(struct.pack("<d", float(nw)))
            f_in.write(struct.pack("<d", float(ntau)))
            trace_flat = trace.flatten()
            f_in.write(struct.pack(f"<{len(trace_flat)}d", *trace_flat))

            # Write parameters JSON
            params = {
                "dll_path": str(self.dll_path),
                "delt": delt,
                "dellam": dellam,
                "lam0": lam0,
                "N": N,
                "geometry": geometry,
                "run_type": run_type,
                "max_iterations": max_iterations,
                "target_error": target_error,
                "max_time_seconds": max_time_seconds,
            }
            json.dump(params, f_params)

        # Run the worker script in 32-bit Python
        try:
            cmd = [
                str(self.python32_path),
                str(_WORKER_SCRIPT),
                input_path,
                output_path,
                params_path,
            ]
            logger.debug("Running FROG worker: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FROG DLL worker failed (return code {result.returncode}).\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            if result.stdout.strip():
                logger.info("FROG worker: %s", result.stdout.strip())

            # Read output binary
            return self._read_output(output_path)

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"FROG DLL retrieval timed out after {timeout} seconds. "
                "Try increasing the timeout or reducing max_iterations."
            )
        finally:
            # Clean up temp files
            for path in [input_path, output_path, params_path]:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _read_output(output_path: str) -> FrogRetrievalResult:
        """Read and parse the worker script's output binary file."""
        with open(output_path, "rb") as f:
            raw = f.read()

        header_size = 7 * 8  # 7 doubles
        if len(raw) < header_size:
            raise RuntimeError(
                "FROG DLL worker produced empty or corrupt output file. "
                "Check that the DLL path is correct and the trace data is valid."
            )

        # Parse header: N, frog_error, num_iterations, temporal_fwhm,
        #               spectral_fwhm, time_n, wave_n
        N = int(struct.unpack_from("<d", raw, 0)[0])
        frog_error = struct.unpack_from("<d", raw, 8)[0]
        num_iterations = int(struct.unpack_from("<d", raw, 16)[0])
        temporal_fwhm = struct.unpack_from("<d", raw, 24)[0]
        spectral_fwhm = struct.unpack_from("<d", raw, 32)[0]
        time_n = int(struct.unpack_from("<d", raw, 40)[0])
        wave_n = int(struct.unpack_from("<d", raw, 48)[0])

        # Calculate expected file size
        arrays_size = (
            2 * N  # Er, Ei
            + 2 * N * N  # A, AR
            + 3 * wave_n  # waveout, specOut, phaseOut
            + 3 * time_n  # timeout, field, fphase
        )
        expected_size = header_size + arrays_size * 8
        if len(raw) < expected_size:
            raise RuntimeError(
                f"FROG output file too small: expected {expected_size} bytes, "
                f"got {len(raw)}. N={N}, time_n={time_n}, wave_n={wave_n}."
            )

        # Parse arrays sequentially
        offset = header_size

        def read_array(count):
            nonlocal offset
            arr = np.array(struct.unpack_from(f"<{count}d", raw, offset))
            offset += count * 8
            return arr

        Er = read_array(N)
        Ei = read_array(N)
        A_flat = read_array(N * N)
        AR_flat = read_array(N * N)
        waveout = read_array(wave_n)
        specOut = read_array(wave_n)
        phaseOut = read_array(wave_n)
        timeout = read_array(time_n)
        field = read_array(time_n)
        fphase = read_array(time_n)

        return FrogRetrievalResult(
            E_real=Er,
            E_imag=Ei,
            frog_error=frog_error,
            num_iterations=num_iterations,
            N=N,
            temporal_fwhm=temporal_fwhm,
            spectral_fwhm=spectral_fwhm,
            input_trace=A_flat.reshape(N, N),
            retrieved_trace=AR_flat.reshape(N, N),
            wavelength=waveout,
            spectral_intensity=specOut,
            spectral_phase=phaseOut,
            time=timeout,
            temporal_intensity=field,
            temporal_phase=fphase,
        )

    @classmethod
    def from_config(
        cls,
        dll_path: Optional[str | Path] = None,
        python32_path: Optional[str | Path] = None,
    ) -> FrogDllRetrieval:
        """Create a FrogDllRetrieval instance using paths from the GEECS config.

        Attempts to load paths from GeecsPathsConfig if not provided directly.

        Parameters
        ----------
        dll_path : str or Path, optional
            Override for the DLL path. If None, reads from config.
        python32_path : str or Path, optional
            Override for the 32-bit Python path. If None, reads from config.

        Returns
        -------
        FrogDllRetrieval
            Configured instance ready for retrieval.

        Raises
        ------
        FileNotFoundError
            If paths cannot be found in config or on disk.
        """
        if dll_path is None or python32_path is None:
            try:
                from geecs_data_utils import GeecsPathsConfig

                config = GeecsPathsConfig()

                if dll_path is None:
                    cfg_dll = getattr(config, "frog_dll_path", None)
                    if cfg_dll is None:
                        raise FileNotFoundError(
                            "frog_dll_path not found in config.ini. "
                            "Add it under [Paths]:\n"
                            "  frog_dll_path = D:\\path\\to\\FROG.dll"
                        )
                    dll_path = cfg_dll

                if python32_path is None:
                    cfg_py32 = getattr(config, "frog_python32_path", None)
                    if cfg_py32 is None:
                        raise FileNotFoundError(
                            "frog_python32_path not found in config.ini. "
                            "Add it under [Paths]:\n"
                            "  frog_python32_path = "
                            "D:\\path\\to\\python32\\python.exe"
                        )
                    python32_path = cfg_py32

            except ImportError:
                raise FileNotFoundError(
                    "geecs_data_utils not available and no explicit paths provided. "
                    "Install geecs_data_utils or provide dll_path and python32_path "
                    "directly."
                )

        return cls(dll_path=dll_path, python32_path=python32_path)
