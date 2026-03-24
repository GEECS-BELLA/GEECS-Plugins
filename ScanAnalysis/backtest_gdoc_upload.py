"""
Backtest script for gdoc auto-upload (issue #286).

Lets you test the upload path against a historical scan without re-running
analysis.  Two modes:

  MODE 1 — Upload from status YAML (analysis already ran)
      python backtest_gdoc_upload.py

      Reads display_files stored in analysis_status/<analyzer_id>.yaml for
      the given scan and uploads them to the configured gdoc_slot.

  MODE 2 — Upload a specific image directly
      Hard-code IMAGE_PATH below.  Useful for testing the Drive upload and
      table insertion without any analysis at all.

Configuration
-------------
Edit the constants in the "── CONFIG ──" block below.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────

# Historical scan to test against
YEAR = 2025
MONTH = 4
DAY = 3
SCAN_NUMBER = 2
EXPERIMENT = "Undulator"

# Override the Google Doc ID (leave None to use the ID from the experiment INI)
# Find this in the doc URL: https://docs.google.com/document/d/<DOC_ID>/edit
DOCUMENT_ID: str | None = None

# --- Mode 2: bypass analysis and upload a single image directly ---
# Set to a Path (or string) to test the Drive upload + table insertion alone.
# Set to None to use Mode 1 (read display_files from status YAML instead).
IMAGE_PATH: Path | str | None = None

# Slot to use when IMAGE_PATH is set (0–3)
DIRECT_SLOT: int = 0

# Optional: limit status discovery to specific analyzer IDs (None = all)
ONLY_ANALYZER_IDS: list[str] | None = None

# ── END CONFIG ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("backtest_gdoc_upload")


def _run_direct(scan_tag, image_path: Path | str) -> None:
    """Upload a single image directly to the configured slot."""
    from scan_analysis.gdoc_upload import upload_summary_to_gdoc

    image_path = str(image_path)
    logger.info("MODE 2 — direct upload: %s → slot %d", image_path, DIRECT_SLOT)
    ok = upload_summary_to_gdoc(
        scan_tag=scan_tag,
        display_files=[image_path],
        gdoc_slot=DIRECT_SLOT,
        document_id=DOCUMENT_ID,
    )
    logger.info("Upload %s.", "succeeded" if ok else "failed or skipped")


def _run_from_status(scan_tag) -> None:
    """Read display_files from completed status YAMLs and re-upload."""
    from geecs_data_utils import ScanPaths

    from scan_analysis.gdoc_upload import SLOT_TO_ROW_COL, upload_summary_to_gdoc
    from scan_analysis.task_queue import STATUS_DIR_NAME, TaskStatus

    scan_folder = ScanPaths.get_scan_folder_path(tag=scan_tag)
    status_dir = scan_folder / STATUS_DIR_NAME

    if not status_dir.exists():
        logger.error("No status directory found at %s", status_dir)
        logger.error(
            "Run the analyzers first, or use IMAGE_PATH for a direct upload test."
        )
        return

    statuses = []
    for f in sorted(status_dir.glob("*.yaml")):
        try:
            statuses.append(TaskStatus.from_file(f))
        except Exception as exc:
            logger.warning("Could not read %s: %s", f, exc)

    if not statuses:
        logger.error("No status files found in %s", status_dir)
        return

    logger.info("Found %d status file(s) in %s", len(statuses), status_dir)

    uploaded = 0
    for st in statuses:
        if ONLY_ANALYZER_IDS and st.analyzer_id not in ONLY_ANALYZER_IDS:
            continue

        if not st.display_files:
            logger.info(
                "  [%s] state=%s — no display_files recorded, skipping.",
                st.analyzer_id,
                st.state,
            )
            continue

        # Derive slot from priority order as a fallback hint; in practice the
        # user should configure gdoc_slot in the YAML and the factory stamps it.
        # For this backtest we just try every status file that has display_files
        # and ask the user to confirm which slot they want (or limit via ONLY_ANALYZER_IDS).
        logger.info(
            "  [%s] state=%s  display_files=%s",
            st.analyzer_id,
            st.state,
            st.display_files,
        )

        # Prompt for slot if not supplied via ONLY_ANALYZER_IDS narrowing
        slot_input = input(
            f"    Upload '{st.display_files[-1]}' to gdoc? "
            f"Enter slot (0-3) or press Enter to skip: "
        ).strip()
        if not slot_input:
            logger.info("    Skipped.")
            continue

        try:
            slot = int(slot_input)
        except ValueError:
            logger.warning("    Invalid slot '%s', skipping.", slot_input)
            continue

        if slot not in SLOT_TO_ROW_COL:
            logger.warning("    Slot %d out of range (0-3), skipping.", slot)
            continue

        ok = upload_summary_to_gdoc(
            scan_tag=scan_tag,
            display_files=st.display_files,
            gdoc_slot=slot,
            document_id=DOCUMENT_ID,
        )
        if ok:
            uploaded += 1
        logger.info("    Upload %s.", "succeeded" if ok else "failed or skipped")

    logger.info("Done. %d image(s) uploaded.", uploaded)


def main() -> None:
    """Build a ScanTag from CONFIG constants and dispatch to the chosen upload mode."""
    from geecs_data_utils import ScanTag

    scan_tag = ScanTag(
        year=YEAR,
        month=MONTH,
        day=DAY,
        number=SCAN_NUMBER,
        experiment=EXPERIMENT,
    )
    logger.info(
        "Backtest: scan=%02d/%02d/%04d Scan%03d experiment=%s",
        MONTH,
        DAY,
        YEAR,
        SCAN_NUMBER,
        EXPERIMENT,
    )
    logger.info("Document ID override: %s", DOCUMENT_ID or "(from experiment INI)")

    if IMAGE_PATH is not None:
        _run_direct(scan_tag, IMAGE_PATH)
    else:
        _run_from_status(scan_tag)


if __name__ == "__main__":
    main()
