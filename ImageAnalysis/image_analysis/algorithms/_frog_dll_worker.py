r"""Standalone FROG DLL worker script.

This script is designed to be run by a 32-bit Python interpreter to interface
with the 32-bit FROG.dll (Daniel Kane's FROG retrieval algorithm). It uses
ONLY Python standard library modules (no numpy, no project dependencies).

Usage:
    python32.exe _frog_dll_worker.py <input.bin> <output.bin> <params.json>

Input binary format:
    - All values are little-endian float64 (8 bytes each)
    - First value: nw (number of wavelength points, cast to int)
    - Second value: ntau (number of time delay points, cast to int)
    - Remaining values: raw trace data, row-major (nw * ntau float64 values)

Output binary format (all little-endian float64):
    Header (7 doubles):
        N, frog_error, num_iterations, temporal_fwhm, spectral_fwhm, time_n, wave_n
    Arrays:
        Er[N], Ei[N]
        A[N*N], AR[N*N]
        waveout[wave_n], specOut[wave_n], phaseOut[wave_n]
        timeout[time_n], field[time_n], fphase[time_n]

Params JSON format:
    {
        "dll_path": "D:\\path\\to\\FROG.dll",
        "delt": 0.931,
        "dellam": -0.079,
        "lam0": 410.4,
        "N": 512,
        "geometry": 1,
        "run_type": 1,
        "max_iterations": 200,
        "target_error": 0.005,
        "max_time_seconds": 60
    }
"""

import ctypes
import json
import struct
import sys
import time


def _write_array(f, arr, n):
    """Write a ctypes array of length n to file as float64."""
    for i in range(n):
        f.write(struct.pack("<d", arr[i]))


def main():
    """Run FROG DLL retrieval via command-line binary file interface."""
    if len(sys.argv) != 4:
        print(
            "Usage: python32.exe _frog_dll_worker.py "
            "<input.bin> <output.bin> <params.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    params_path = sys.argv[3]

    # Load parameters
    with open(params_path, "r") as f:
        params = json.load(f)

    dll_path = params["dll_path"]
    delt = params["delt"]
    dellam = params["dellam"]
    lam0 = params["lam0"]
    N = params.get("N", 512)
    geometry = params.get("geometry", 1)
    run_type = params.get("run_type", 1)
    max_iterations = params.get("max_iterations", 200)
    target_error = params.get("target_error", 0.005)
    max_time_seconds = params.get("max_time_seconds", 60)

    # Read input trace
    with open(input_path, "rb") as f:
        raw = f.read()

    # Parse header: nw, ntau
    nw = int(struct.unpack_from("<d", raw, 0)[0])
    ntau = int(struct.unpack_from("<d", raw, 8)[0])
    trace_count = nw * ntau
    trace_data = list(struct.unpack_from("<{0}d".format(trace_count), raw, 16))

    # Convert to ctypes array
    RawDataArray = ctypes.c_double * trace_count
    rawdata = RawDataArray(*trace_data)

    # Load DLL
    frog = ctypes.WinDLL(dll_path)

    # ----------------------------------------------------------------
    # DLLInit — initialize the retrieval engine
    # ----------------------------------------------------------------
    DoubleN = ctypes.c_double * N
    DoubleNN = ctypes.c_double * (N * N)

    Er = DoubleN(*([0.0] * N))
    Ei = DoubleN(*([0.0] * N))
    Ekr = DoubleN(*([0.0] * N))
    Eki = DoubleN(*([0.0] * N))
    frogl = DoubleNN(*([0.0] * (N * N)))
    froglk = DoubleNN(*([0.0] * (N * N)))

    status = ctypes.c_int16(0)

    frog.DLLInit(
        ctypes.cast(Er, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Ei, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Ekr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Eki, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(frogl, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(froglk, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int16(N),
        ctypes.c_int16(geometry),
        ctypes.c_int16(run_type),
        ctypes.byref(status),
    )

    if status.value != 0:
        print("DLLInit returned status {0}".format(status.value), file=sys.stderr)

    # ----------------------------------------------------------------
    # GridData — resample the raw trace onto the N x N grid
    # ----------------------------------------------------------------
    frog.GridData(
        ctypes.cast(rawdata, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(ntau),
        ctypes.c_int32(nw),
        ctypes.c_double(delt),
        ctypes.c_double(dellam),
        ctypes.c_double(lam0),
        ctypes.c_char_p(b"frog_output"),
    )

    # ----------------------------------------------------------------
    # DoOneIteration loop — iterative pulse retrieval
    # ----------------------------------------------------------------
    frogerr = ctypes.c_double(0.0)
    iter_status = ctypes.c_int16(0)
    num_iterations = 0
    start_time = time.time()

    for k in range(max_iterations):
        frog.DoOneIteration(
            ctypes.c_int16(k),
            ctypes.byref(frogerr),
            ctypes.byref(iter_status),
        )
        num_iterations = k + 1

        # Early stop on convergence
        if frogerr.value > 0 and frogerr.value < target_error:
            break

        # Early stop on time limit
        if (time.time() - start_time) > max_time_seconds:
            break

    # ----------------------------------------------------------------
    # GetTheBest2 — extract best result metrics
    # ----------------------------------------------------------------
    besterr = ctypes.c_double(0.0)
    ResultArray = ctypes.c_double * 22
    res = ResultArray(*([0.0] * 22))

    frog.GetTheBest2(
        ctypes.byref(besterr),
        ctypes.cast(res, ctypes.POINTER(ctypes.c_double)),
    )

    temporal_fwhm = res[0]
    spectral_fwhm = res[2]

    # ----------------------------------------------------------------
    # GetTimeN / GetWaveN — get output array dimensions
    # ----------------------------------------------------------------
    frog.GetTimeN.restype = ctypes.c_int32
    frog.GetWaveN.restype = ctypes.c_int32

    time_n = frog.GetTimeN()
    wave_n = frog.GetWaveN()

    # ----------------------------------------------------------------
    # GetData — extract all retrieval outputs
    # ----------------------------------------------------------------
    A = DoubleNN(*([0.0] * (N * N)))
    AR = DoubleNN(*([0.0] * (N * N)))

    DoubleTimeN = ctypes.c_double * time_n
    DoubleWaveN = ctypes.c_double * wave_n

    waveout = DoubleWaveN(*([0.0] * wave_n))
    specOut = DoubleWaveN(*([0.0] * wave_n))
    phaseOut = DoubleWaveN(*([0.0] * wave_n))
    timeout = DoubleTimeN(*([0.0] * time_n))
    field = DoubleTimeN(*([0.0] * time_n))
    fphase = DoubleTimeN(*([0.0] * time_n))

    frog.GetData(
        ctypes.cast(A, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(AR, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(waveout, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(specOut, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(phaseOut, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(timeout, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(field, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(fphase, ctypes.POINTER(ctypes.c_double)),
    )

    # ----------------------------------------------------------------
    # Write output binary
    # ----------------------------------------------------------------
    with open(output_path, "wb") as f:
        # Header: 7 doubles
        f.write(struct.pack("<d", float(N)))
        f.write(struct.pack("<d", besterr.value))
        f.write(struct.pack("<d", float(num_iterations)))
        f.write(struct.pack("<d", temporal_fwhm))
        f.write(struct.pack("<d", spectral_fwhm))
        f.write(struct.pack("<d", float(time_n)))
        f.write(struct.pack("<d", float(wave_n)))

        # E-field arrays
        _write_array(f, Er, N)
        _write_array(f, Ei, N)

        # Input and retrieved FROG traces (N x N flattened)
        _write_array(f, A, N * N)
        _write_array(f, AR, N * N)

        # Spectral data (wave_n length)
        _write_array(f, waveout, wave_n)
        _write_array(f, specOut, wave_n)
        _write_array(f, phaseOut, wave_n)

        # Temporal data (time_n length)
        _write_array(f, timeout, time_n)
        _write_array(f, field, time_n)
        _write_array(f, fphase, time_n)

    elapsed = time.time() - start_time
    print(
        "FROG retrieval complete: {0} iterations, error={1:.6f}, "
        "temporal_fwhm={2:.2f}, spectral_fwhm={3:.2f}, "
        "elapsed={4:.1f}s".format(
            num_iterations, besterr.value, temporal_fwhm, spectral_fwhm, elapsed
        )
    )


if __name__ == "__main__":
    main()
