# Changelog — image-analysis

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.1] — 2026-05-06

### Removed
- `lcls-tools` dependency (closes #231). The package was not used in any active
  code paths; the one internal helper (`gaussian_fit_beam_size`) has been
  rewritten using `scipy.optimize.curve_fit`, which is already a dependency.
  `image_analysis/algorithms/lcls_tools_gauss_fit.py` (a thin wrapper that was
  never imported) has been deleted.

## [1.1.0] — current
