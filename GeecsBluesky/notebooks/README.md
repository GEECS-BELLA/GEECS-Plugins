# GeecsBluesky Notebooks

Interactive notebooks in this directory are operator-facing runbooks for
validating the Bluesky backend against live GEECS hardware. They are not unit
tests and may create real scan folders, move devices, or fire shots when their
operator switches are enabled.

Run them from the `GeecsBluesky/` package directory:

```bash
poetry install
poetry run jupyter lab notebooks/
```

Before committing notebook changes:

- Keep all outputs cleared.
- Keep hardware-moving or shot-control cells gated behind explicit `False`
  switches.
- Prefer thin notebook cells that call package APIs rather than hiding reusable
  logic in the notebook.

## Notebooks

- `bluesky_hardware_smoke.ipynb` — run `BlueskyScanner` NOSCAN and optional
  STANDARD scans from a notebook, collect event documents, and confirm scan
  folder metadata.
- `external_asset_readback.ipynb` — fill GEECS camera external asset
  Resource/Datum documents locally into NumPy arrays, including a parameterized
  existing-scan lookup and a synthetic no-hardware smoke test.
