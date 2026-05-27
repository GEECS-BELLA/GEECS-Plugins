"""Generate documentation screenshots for ConfigFileGUI and LiveWatchGUI.

Runs each app headlessly (QT_QPA_PLATFORM=offscreen) and grabs representative
states to docs/apps/assets/. Designed to be re-run when the GUIs change.

Requires the sister repo ``GEECS-Plugins-Configs`` checked out next to
``GEECS-Plugins-docs-sweep`` so the analyzer/group sample configs resolve.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Resolve paths relative to this script so the script is portable across
# checkouts as long as the sister `GEECS-Plugins-Configs` repo is alongside.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SAMPLE_CONFIGS = REPO.parent / "GEECS-Plugins-Configs" / "scan_analysis_configs"
OUT = REPO / "docs/apps/assets"
OUT.mkdir(parents=True, exist_ok=True)

if not SAMPLE_CONFIGS.is_dir():
    raise SystemExit(
        f"Sample configs not found at {SAMPLE_CONFIGS}. "
        "Clone GEECS-Plugins-Configs next to this repo, or edit SAMPLE_CONFIGS."
    )

sys.path.insert(0, str(REPO / "ScanAnalysis"))

from PyQt5.QtWidgets import QApplication  # noqa: E402


def _settle(app: QApplication, n: int = 3) -> None:
    for _ in range(n):
        app.processEvents()


def _save(widget, name: str) -> None:
    out = OUT / name
    pix = widget.grab()
    pix.save(str(out), "PNG")
    print(f"  saved {out.name}  ({pix.width()}x{pix.height()})")


def shoot_configgui(app: QApplication) -> None:
    """Render ConfigFileGUI through four representative states."""
    from ConfigFileGUI.config_editor_window import ConfigEditorWindow

    print("ConfigFileGUI:")

    w = ConfigEditorWindow(scan_config_dir=SAMPLE_CONFIGS)
    w.resize(1400, 900)
    w.show()
    _settle(app)
    _save(w, "configgui_01_initial.png")

    cam_path = SAMPLE_CONFIGS / "analyzers/HTU/UC_TopView.yaml"
    w._on_file_selected(cam_path, "analyzer")
    _settle(app)
    _save(w, "configgui_02_analyzer_camera.png")

    w._toggle_yaml_action.setChecked(True)
    w._on_toggle_yaml_preview(True)
    _settle(app)
    _save(w, "configgui_03_yaml_preview.png")

    w._toggle_yaml_action.setChecked(False)
    w._on_toggle_yaml_preview(False)
    grp_path = SAMPLE_CONFIGS / "groups/HTU/baseline.yaml"
    w._on_file_selected(grp_path, "group")
    _settle(app)
    _save(w, "configgui_04_group.png")

    w.close()


def shoot_livewatch(app: QApplication) -> None:
    """Render LiveWatchGUI with sample configs wired into the group dropdown."""
    # Point ScanPaths at the sample configs *before* importing LiveWatchWindow
    # so its module-level discovery picks up the sample groups.
    from geecs_data_utils import ScanPaths

    ScanPaths.paths_config.scan_analysis_configs_path = SAMPLE_CONFIGS

    from LiveWatchGUI.live_watch_window import LiveWatchWindow

    print("LiveWatchGUI:")

    # 1. Default state — group dropdown auto-populated from sample configs
    w = LiveWatchWindow()
    w.resize(620, 720)
    w.show()
    _settle(app)
    _save(w, "livewatch_01_initial.png")

    # 2. Same window with a group selected — closer to the typical
    # pre-Start configuration the user will see day-to-day.
    # Labelled "Analyzer Group" in the UI but called combo_experiment internally
    grp_combo = w.combo_experiment
    if grp_combo.count() > 0:
        for i in range(grp_combo.count()):
            if "baseline" in grp_combo.itemText(i).lower():
                grp_combo.setCurrentIndex(i)
                break
        else:
            grp_combo.setCurrentIndex(0)
        _settle(app)
        _save(w, "livewatch_02_group_selected.png")

    w.close()


def main() -> int:
    """Generate every documentation screenshot in one pass."""
    app = QApplication(sys.argv)
    shoot_configgui(app)
    shoot_livewatch(app)
    return 0


if __name__ == "__main__":
    sys.exit(main())
