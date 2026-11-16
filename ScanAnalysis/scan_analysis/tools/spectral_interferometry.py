"""Analyse one spectrometer file as a spectral interferogram, now.

The file-reading edge of
:mod:`geecs_analysis.algorithms.spectral_interferometry`: read a
two-column wavelength (nm) / counts file with data-utils' reader, recover
the relative delay τ0 and differential GDD of the two pulses, print the
JSON result and optionally save the diagnostic figure::

    python -m scan_analysis.tools.spectral_interferometry SPECTRUM.txt --png ~/si.png

From Python, :func:`analyze_spectrum_file` returns the analysis (``.result``
is the JSON model) and
:func:`geecs_analysis.render.spectral_interferometry.diagnostic_figure`
draws it.

Outputs go only where you name them, into a folder that already exists:
nothing is ever created next to the data (root CLAUDE.md, "Analysis code
is a consumer of scan folders, never a producer").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from geecs_analysis.algorithms.spectral_interferometry import (
    SIGN_CONVENTION,
    SpectralInterferometryAnalysis,
    SpectralInterferometryConfig,
    analyze_interferogram,
)


def analyze_spectrum_file(
    path: Path | str, config: Optional[SpectralInterferometryConfig] = None
) -> SpectralInterferometryAnalysis:
    """Read a tab-separated wavelength (nm) / counts file and analyse it."""
    from geecs_data_utils.io.array1d import Data1DConfig, Data1DType, read_1d_data

    data = read_1d_data(Path(path), Data1DConfig(data_type=Data1DType.TSV)).data
    return analyze_interferogram(data[:, 0], data[:, 1], config)


def _output(path: Optional[str]) -> Optional[Path]:
    """An output path whose folder already exists; never create one."""
    if path is None:
        return None
    out = Path(path).expanduser()
    if not out.parent.is_dir():
        raise SystemExit(
            f"output folder does not exist (not creating it): {out.parent}"
        )
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scan_analysis.tools.spectral_interferometry",
        description="Relative delay and differential GDD from one spectral interferogram.",
        epilog="Sign convention: " + SIGN_CONVENTION,
    )
    parser.add_argument("spectrum", help="two-column file: wavelength (nm), counts")
    parser.add_argument(
        "--resolution-nm", type=float, help="spectrometer optical resolution (FWHM)"
    )
    parser.add_argument("--fit-range", type=float, nargs=2, metavar=("NM", "NM"))
    parser.add_argument(
        "--reference-nm", type=float, help="report tau0 at this wavelength"
    )
    parser.add_argument(
        "--window-fs", type=float, nargs=2, metavar=("FS", "FS"), help="|delay| window"
    )
    parser.add_argument(
        "--region", type=float, nargs=2, metavar=("NM", "NM"), help="region transformed"
    )
    parser.add_argument(
        "--sideband", choices=("positive", "negative"), default="positive"
    )
    parser.add_argument("--png", help="save the diagnostic figure here")
    parser.add_argument(
        "--json", help="save the JSON result here (it is always printed)"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line; exit 0 when the result is reliable, 1 when flagged."""
    args = _parser().parse_args(argv)
    png, json_path = _output(args.png), _output(args.json)
    config = SpectralInterferometryConfig(
        resolution_nm=args.resolution_nm,
        fit_range_nm=tuple(args.fit_range) if args.fit_range else None,
        reference_wavelength_nm=args.reference_nm,
        sideband_window_fs=tuple(args.window_fs) if args.window_fs else None,
        spectral_region_nm=tuple(args.region) if args.region else None,
        sideband=args.sideband,
    )
    analysis = analyze_spectrum_file(args.spectrum, config)
    text = analysis.result.model_dump_json(indent=2)
    print(text)
    if json_path is not None:
        json_path.write_text(text + "\n", encoding="utf-8")
    if png is not None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        from geecs_analysis.render.spectral_interferometry import diagnostic_figure

        figure = diagnostic_figure(analysis, title=Path(args.spectrum).name)
        FigureCanvasAgg(figure)
        figure.savefig(png, dpi=110)
    return 0 if analysis.result.reliable else 1


if __name__ == "__main__":
    sys.exit(main())
