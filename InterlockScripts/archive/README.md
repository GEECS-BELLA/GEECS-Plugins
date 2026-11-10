# Archive

Files moved here on 2026-09-22 during the interlock cleanup/reorg pass.
Nothing was deleted — these are candidates for deletion pending manual
review, not confirmed-dead code.

- **`thing.ipynb`** — untracked scratch notebook, ad-hoc `CAM-2BL-Aegeaon`
  polling. No apparent production use.
- **`listen_to_lulu.py`** — standalone raw-socket UDP listener (port from `.env`);
  doesn't use the interlock library at all. Debug/exploration script.
- **`dev_interlock.py`** — named "dev," structurally a near-duplicate of
  `tape_interlock.py` with the same camera device. Looks like a scratch
  variant used while testing, not a deployed script.
- **`monitor_diagnostics_example.py`** — example diagnostic client meant to
  run alongside `tape_interlock_refactored_example.py`, which does not exist
  anywhere in the repo. Broken/orphaned example.
- **`Tape_interlock_DEPRICATED.ipynb`** — already named deprecated by a
  previous author.
- **`tape_interlock.py`** — narrower-scope predecessor of
  `PW_interlocks/windmill_interlock.py` (same `CAM-PL1-TapeDrivePointing`
  device, fewer checks). Possibly superseded — flagged here rather than
  assumed dead since it wasn't confirmed still unused at the experiment.

If you confirm any of these are truly unused, delete them outright rather
than leaving them here indefinitely.
