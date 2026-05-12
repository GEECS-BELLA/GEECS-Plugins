"""Analysis Preview dialog for the Config File Editor GUI.

Provides a dialog that lets users select a device config and an image file,
run the image-analysis pipeline, and view raw vs. analysed previews
side-by-side with scalar outputs.

This module is consumed by :mod:`config_editor_window`.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCompleter,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy matplotlib imports – keeps the module importable even when
# matplotlib is not installed (the dialog will show an error instead).
# ---------------------------------------------------------------------------
_MPL_AVAILABLE = False
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _MPL_AVAILABLE = True
except ImportError:
    FigureCanvasQTAgg = None  # type: ignore[assignment,misc]
    Figure = None  # type: ignore[assignment,misc]
    plt = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGE_FILTERS = (
    "Image files (*.png *.tiff *.tif *.npy *.tsv *.h5 *.tdms *.csv);;All files (*)"
)

_2D_INDICATOR_KEYS = {"bit_depth", "background", "crosshair_masking"}
_1D_INDICATOR_KEYS = {"data_loading", "data_type"}


# ---------------------------------------------------------------------------
# Helper: detect config type from a dict
# ---------------------------------------------------------------------------
def _detect_config_type_from_dict(
    config_dict: Dict[str, Any],
) -> str:
    """Return ``'camera_2d'``, ``'line_1d'``, or ``'unknown'``.

    Parameters
    ----------
    config_dict : dict
        The raw configuration dictionary.

    Returns
    -------
    str
        Detected config type.
    """
    keys = set(config_dict.keys())
    if keys & _2D_INDICATOR_KEYS:
        return "camera_2d"
    if keys & _1D_INDICATOR_KEYS:
        return "line_1d"
    return "unknown"


# ======================================================================
# Worker thread
# ======================================================================
class AnalysisWorker(QThread):
    """Run image analysis off the main (GUI) thread.

    Signals
    -------
    raw_image_ready(object)
        Emitted with the raw image data (``numpy.ndarray``) after loading.
    analysis_complete(object, object)
        Emitted with ``(ImageAnalyzerResult, matplotlib.figure.Figure)``.
    analysis_error(str)
        Emitted with a human-readable error description.
    """

    raw_image_ready = pyqtSignal(object)
    analysis_complete = pyqtSignal(object, object)
    analysis_error = pyqtSignal(str)

    def __init__(
        self,
        config_dict: Dict[str, Any],
        config_type: str,
        image_path: Path,
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._config_dict = config_dict
        self._config_type = config_type
        self._image_path = image_path

    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: C901 – unavoidable branching
        """Execute the analysis pipeline in a background thread."""
        try:
            if self._config_type == "camera_2d":
                self._run_2d()
            elif self._config_type == "line_1d":
                self._run_1d()
            else:
                self.analysis_error.emit(
                    f"Unsupported config type: {self._config_type}"
                )
        except Exception:
            self.analysis_error.emit(traceback.format_exc())

    # ------------------------------------------------------------------
    # 2D (camera / beam) path
    # ------------------------------------------------------------------
    def _run_2d(self) -> None:
        from image_analysis.algorithms.basic_beam_stats import (
            beam_profile_stats,
            flatten_beam_stats,
        )
        from image_analysis.config_loader import load_camera_config
        from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
        from image_analysis.processing.array2d.pipeline import (
            apply_camera_processing_pipeline,
            create_background_manager_from_config,
        )
        from image_analysis.types import ImageAnalyzerResult
        from image_analysis.utils import read_imaq_image

        # 1. Load raw image
        raw_image = read_imaq_image(self._image_path)
        self.raw_image_ready.emit(raw_image)

        # 2. Build validated CameraConfig from the dict
        camera_config = load_camera_config(self._config_dict)

        # 3. Apply processing pipeline
        bg_manager = create_background_manager_from_config(camera_config)
        processed_image = apply_camera_processing_pipeline(
            raw_image, camera_config, bg_manager
        )

        # 4. Beam statistics
        beam_stats = beam_profile_stats(processed_image)
        scalars = flatten_beam_stats(beam_stats, prefix=camera_config.name)

        # 5. Build result object
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=processed_image,
            scalars=scalars,
            metadata={},
        )

        # Add projection overlays for rendering
        if processed_image is not None:
            result.render_data = {
                "horizontal_projection": processed_image.sum(axis=0),
                "vertical_projection": processed_image.sum(axis=1),
            }

        # 6. Render analysed figure
        fig, _ax = BeamAnalyzer.render_image(result)
        self.analysis_complete.emit(result, fig)

    # ------------------------------------------------------------------
    # 1D (line / spectrum) path
    # ------------------------------------------------------------------
    def _run_1d(self) -> None:
        from image_analysis.config_loader import load_line_config
        from image_analysis.data_1d_utils import read_1d_data
        from image_analysis.processing.array1d.pipeline import (
            apply_line_processing_pipeline,
        )
        from image_analysis.types import ImageAnalyzerResult

        # 1. Build validated Line1DConfig from the dict
        line_config = load_line_config(self._config_dict)

        # 2. Load raw data
        data_result = read_1d_data(self._image_path, line_config.data_loading)
        raw_data = data_result.data  # Nx2 array
        self.raw_image_ready.emit(raw_data)

        # 3. Scale + process
        scaled = raw_data.copy()
        if line_config.x_scale_factor is not None:
            scaled[:, 0] *= line_config.x_scale_factor
        if line_config.y_scale_factor is not None:
            scaled[:, 1] *= line_config.y_scale_factor

        processed = apply_line_processing_pipeline(scaled, line_config)

        # 4. Build metadata
        metadata = {
            "line_name": line_config.name,
            "x_units": line_config.x_units or data_result.x_units,
            "y_units": line_config.y_units or data_result.y_units,
            "x_label": data_result.x_label,
            "y_label": data_result.y_label,
        }

        # 5. Build result
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=processed,
            scalars={},
            metadata=metadata,
        )

        # 6. Render using a fresh figure
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100, constrained_layout=True)
        ax.plot(processed[:, 0], processed[:, 1], linewidth=0.8)

        x_label = metadata.get("x_label", "X")
        x_units = metadata.get("x_units")
        if x_units:
            x_label = f"{x_label} ({x_units})"
        y_label = metadata.get("y_label", "Y")
        y_units = metadata.get("y_units")
        if y_units:
            y_label = f"{y_label} ({y_units})"

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        if line_config.name:
            ax.set_title(line_config.name)
        ax.grid(True, alpha=0.3)

        self.analysis_complete.emit(result, fig)


# ======================================================================
# Helper: matplotlib canvas widget
# ======================================================================
class _PlotCanvas(QWidget):
    """Thin wrapper around a ``FigureCanvasQTAgg``.

    Parameters
    ----------
    title : str
        Label shown above the canvas.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(title)
        layout.addWidget(self._label)

        self._figure = Figure(figsize=(4, 3), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas, stretch=1)

    # ------------------------------------------------------------------
    def set_figure(self, fig: Figure) -> None:
        """Replace the displayed figure with *fig*.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            New figure to display.
        """
        layout = self.layout()
        # Remove old canvas
        layout.removeWidget(self._canvas)
        self._canvas.setParent(None)  # type: ignore[arg-type]
        self._canvas.close()

        # Install new canvas
        self._figure = fig
        self._canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(self._canvas, stretch=1)
        self._canvas.draw()

    def show_array_2d(self, data: np.ndarray) -> None:
        """Display a 2-D image with ``imshow``.

        Parameters
        ----------
        data : numpy.ndarray
            2-D array to display.
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.imshow(data, cmap="gray", aspect="auto")
        ax.set_title("Raw Image")
        self._figure.tight_layout()
        self._canvas.draw()

    def show_array_1d(self, data: np.ndarray) -> None:
        """Display a 1-D line plot from an Nx2 array.

        Parameters
        ----------
        data : numpy.ndarray
            Nx2 array (col 0 = x, col 1 = y).
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.plot(data[:, 0], data[:, 1], linewidth=0.8)
        ax.set_title("Raw Data")
        ax.grid(True, alpha=0.3)
        self._figure.tight_layout()
        self._canvas.draw()


# ======================================================================
# Dialog
# ======================================================================
class AnalysisPreviewDialog(QDialog):
    """Modal-less dialog for previewing image-analysis results.

    Parameters
    ----------
    config_dir : Path or None
        Directory containing device-config YAML files.  Used for
        auto-complete suggestions in the config selector.
    get_current_config : callable or None
        Callback that returns
        ``(config_name: str, config_dict: dict, config_type: str)``
        for the config currently loaded in the editor.  When the user
        selects a config whose name matches *config_name*, the
        in-memory dict is used instead of reading from disk.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        get_current_config: Optional[
            Callable[[], Tuple[str, Dict[str, Any], str]]
        ] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=None)  # No parent → independent window
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self._config_dir = config_dir
        self._get_current_config = get_current_config
        self._worker: Optional[AnalysisWorker] = None

        self.setWindowTitle("Analysis Preview")
        self.resize(1000, 700)

        self._build_ui()
        self._populate_completer()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Assemble the dialog layout."""
        if not _MPL_AVAILABLE:
            layout = QVBoxLayout(self)
            layout.addWidget(
                QLabel(
                    "matplotlib is not installed.\n"
                    "Install it with:  pip install matplotlib"
                )
            )
            return

        root = QVBoxLayout(self)

        # ---- Row 1: Config selector ----
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Config:"))
        self._config_edit = QLineEdit()
        self._config_edit.setPlaceholderText("Type config name…")
        row1.addWidget(self._config_edit, stretch=1)

        row1.addWidget(QLabel("  Image:"))
        self._image_edit = QLineEdit()
        self._image_edit.setPlaceholderText("Path to image / data file")
        row1.addWidget(self._image_edit, stretch=2)

        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._on_browse)
        row1.addWidget(self._browse_btn)
        root.addLayout(row1)

        # ---- Row 2: Run button + status ----
        row2 = QHBoxLayout()
        self._run_btn = QPushButton("▶ Run Analysis")
        self._run_btn.clicked.connect(self._on_run)
        row2.addWidget(self._run_btn)

        self._status_label = QLabel("")
        row2.addWidget(self._status_label, stretch=1)
        root.addLayout(row2)

        # ---- Row 3: Side-by-side canvases ----
        canvas_splitter = QSplitter()
        self._raw_canvas = _PlotCanvas("Raw Image")
        self._analyzed_canvas = _PlotCanvas("Analyzed Preview")
        canvas_splitter.addWidget(self._raw_canvas)
        canvas_splitter.addWidget(self._analyzed_canvas)
        canvas_splitter.setSizes([500, 500])
        root.addWidget(canvas_splitter, stretch=1)

        # ---- Row 4: Scalars output ----
        root.addWidget(QLabel("Scalars / Analysis Output:"))
        self._scalars_text = QPlainTextEdit()
        self._scalars_text.setReadOnly(True)
        self._scalars_text.setMaximumHeight(120)
        root.addWidget(self._scalars_text)

    # ------------------------------------------------------------------
    # Auto-complete population
    # ------------------------------------------------------------------
    def _populate_completer(self) -> None:
        """Fill the ``QCompleter`` with YAML stems from *config_dir*."""
        if not _MPL_AVAILABLE:
            return

        names: List[str] = []
        if self._config_dir is not None and self._config_dir.is_dir():
            for p in sorted(self._config_dir.iterdir()):
                if p.suffix in (".yaml", ".yml"):
                    names.append(p.stem)

        completer = QCompleter(names)
        completer.setCaseSensitivity(0)  # Qt.CaseInsensitive
        self._config_edit.setCompleter(completer)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_browse(self) -> None:
        """Open a file dialog for the image path."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image / Data File",
            "",
            _IMAGE_FILTERS,
        )
        if path:
            self._image_edit.setText(path)

    def _on_run(self) -> None:
        """Kick off the analysis worker."""
        if not _MPL_AVAILABLE:
            return

        config_name = self._config_edit.text().strip()
        image_path_str = self._image_edit.text().strip()

        if not config_name:
            self._status_label.setText("Error: enter a config name")
            return
        if not image_path_str:
            self._status_label.setText("Error: select an image file")
            return

        image_path = Path(image_path_str)
        if not image_path.is_file():
            self._status_label.setText(f"Error: file not found — {image_path}")
            return

        # Resolve config dict + type
        config_dict, config_type = self._resolve_config(config_name)
        if config_dict is None:
            self._status_label.setText(f"Error: could not load config '{config_name}'")
            return

        if config_type == "unknown":
            self._status_label.setText("Error: cannot determine config type (2D / 1D)")
            return

        # Disable button while running
        self._run_btn.setEnabled(False)
        self._status_label.setText("Analyzing…")
        self._scalars_text.setPlainText("")

        self._worker = AnalysisWorker(
            config_dict=config_dict,
            config_type=config_type,
            image_path=image_path,
        )
        self._worker.raw_image_ready.connect(self._on_raw_ready)
        self._worker.analysis_complete.connect(self._on_complete)
        self._worker.analysis_error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------
    def _resolve_config(
        self,
        config_name: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Return ``(config_dict, config_type)`` for *config_name*.

        If the name matches the currently-loaded config in the editor
        the in-memory (possibly unsaved) dict is used.  Otherwise the
        config is loaded from disk.

        Parameters
        ----------
        config_name : str
            Stem of the config file (without extension).

        Returns
        -------
        tuple
            ``(config_dict, config_type)`` or ``(None, "unknown")`` on
            failure.
        """
        # Check if it matches the editor's current config
        if self._get_current_config is not None:
            try:
                editor_name, editor_dict, editor_type = self._get_current_config()
                if editor_name and editor_name == config_name and editor_dict:
                    logger.info(
                        "Using in-memory config for '%s' (type=%s)",
                        config_name,
                        editor_type,
                    )
                    return editor_dict, editor_type
            except Exception:
                logger.debug("get_current_config callback failed", exc_info=True)

        # Fall back to loading from disk
        if self._config_dir is None:
            return None, "unknown"

        for ext in (".yaml", ".yml"):
            candidate = self._config_dir / f"{config_name}{ext}"
            if candidate.is_file():
                try:
                    import yaml

                    with open(candidate) as fh:
                        data = yaml.safe_load(fh)
                    if not isinstance(data, dict):
                        return None, "unknown"
                    ctype = _detect_config_type_from_dict(data)
                    return data, ctype
                except Exception:
                    logger.warning(
                        "Failed to load config '%s'",
                        candidate,
                        exc_info=True,
                    )
                    return None, "unknown"

        return None, "unknown"

    # ------------------------------------------------------------------
    # Worker result handlers
    # ------------------------------------------------------------------
    def _on_raw_ready(self, data: np.ndarray) -> None:
        """Display the raw image / data on the left canvas.

        Parameters
        ----------
        data : numpy.ndarray
            Raw image (2-D) or raw line data (Nx2).
        """
        if data.ndim == 2 and data.shape[1] != 2:
            self._raw_canvas.show_array_2d(data)
        elif data.ndim == 2 and data.shape[1] == 2:
            self._raw_canvas.show_array_1d(data)
        elif data.ndim == 1:
            # Unlikely, but handle gracefully
            self._raw_canvas.show_array_1d(
                np.column_stack([np.arange(len(data)), data])
            )

    def _on_complete(self, result: Any, fig: Any) -> None:
        """Display the analysed figure and scalars.

        Parameters
        ----------
        result : ImageAnalyzerResult
            Analysis result containing scalars and metadata.
        fig : matplotlib.figure.Figure
            Pre-rendered matplotlib figure from the worker.
        """
        self._analyzed_canvas.set_figure(fig)

        # Format scalars
        scalars = getattr(result, "scalars", {})
        if scalars:
            lines = [f"{k}: {v}" for k, v in scalars.items()]
            self._scalars_text.setPlainText("\n".join(lines))
        else:
            self._scalars_text.setPlainText("(no scalars returned)")

        self._status_label.setText("Complete")

    def _on_error(self, message: str) -> None:
        """Show an error in the scalars area.

        Parameters
        ----------
        message : str
            Error traceback or description.
        """
        self._scalars_text.setPlainText(f"ERROR:\n{message}")
        self._status_label.setText("Error")

    def _on_worker_finished(self) -> None:
        """Re-enable the Run button when the worker exits."""
        self._run_btn.setEnabled(True)
