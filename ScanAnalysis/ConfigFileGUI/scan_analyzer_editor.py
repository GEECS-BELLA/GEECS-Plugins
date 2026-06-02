"""Editor panel for a unified diagnostic YAML.

Builds a form against :class:`image_analysis.config.DiagnosticAnalysisConfig`
(the post-PR-E unified-diagnostic schema) with four sections:

* **General** — ``name`` + optional ``output_name`` / ``metric_suffix``
  (output identifier + scalar-key suffix, applied by ScanAnalysis on
  consumption)
* **Image Analyzer** — ``image_analyzer.class_path`` +
  ``image_analyzer.kwargs``
* **Image** — discriminator type combo (``camera`` / ``line`` /
  ``(none)``) + an embedded :class:`ConfigEditorPanel` showing the
  full form for the chosen variant. ``ConfigEditorPanel`` already
  knows how to round-trip :class:`CameraConfig` / :class:`Line1DConfig`
  (including ``metadata`` / ``analysis`` / pipeline ordering), so we
  reuse it rather than rebuilding the per-section plumbing.
* **Scan** — :class:`ScanRuntimeConfig` fields rendered via
  :class:`SectionWidget`.

The general / image-analyzer / scan sections are small enough to stay
fixed-size; the image section gets stretch so the embedded
``ConfigEditorPanel`` (with its own scroll area) takes any extra
vertical space and handles scrolling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config_editor_panel import ConfigEditorPanel
from .section_widget import SectionWidget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prune_empty(value: Any) -> Any:
    """Recursively drop ``None`` values and dicts that bottom out at None.

    Used at the output boundary to ensure round-trip fidelity:
    SectionWidget always emits ``None`` for unset Optional fields and a
    default-filled struct for every nested BaseModel, even when the
    source YAML didn't mention those keys. Pruning here keeps the
    on-disk YAML from accumulating noise on every save.

    Behaviour:
    * Scalars: ``None`` → dropped (signalled by returning ``None`` for
      the caller to filter); everything else passes through.
    * Dicts: drop keys whose value prunes to ``None``; if every value
      prunes, return ``None`` so the caller drops the dict itself.
    * Lists: pass through unchanged (lists are usually meaningful even
      when their entries have Nones).
    """
    if value is None:
        return None
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            pv = _prune_empty(v)
            if pv is None:
                continue
            cleaned[k] = pv
        return cleaned if cleaned else None
    return value


def _prune_scan_section_values(scan_values: Any) -> Any:
    """Prune scan-section values while preserving explicit empty directives."""
    pruned = _prune_empty(scan_values)
    raw_background_source = (
        scan_values.get("background_source") if isinstance(scan_values, dict) else None
    )
    if (
        isinstance(raw_background_source, dict)
        and raw_background_source.get("scan_number") is None
        and raw_background_source.get("from_current_scan") is None
        and raw_background_source.get("autodetect") == {}
    ):
        pruned = pruned or {}
        pruned["background_source"] = {"autodetect": {}}

    # Special case: ``background_source`` always renders as a nested
    # section, and ``FromCurrentScanSpec.method`` has a non-None default
    # ("median"). So an untouched background_source survives pruning as
    # ``{"from_current_scan": {"method": "median"}}``. Treat that shape
    # as "user didn't configure a directive" and drop it.
    if pruned and pruned.get("background_source") == {
        "from_current_scan": {"method": "median"}
    }:
        pruned = {k: v for k, v in pruned.items() if k != "background_source"}

    return pruned


# ---------------------------------------------------------------------------
# Lightweight JSON dict editor (used for image_analyzer.kwargs)
# ---------------------------------------------------------------------------


class _JsonDictWidget(QWidget):
    """Inline widget for editing ``Dict[str, Any]`` values as JSON text.

    Displays a ``QPlainTextEdit`` for entering JSON. On every change
    the text is validated and an error label is shown when the JSON
    is malformed. ``get_value`` returns the parsed dict (or ``{}`` on
    empty / invalid input).
    """

    valueChanged = pyqtSignal()

    def __init__(
        self,
        label_text: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        if label_text:
            lbl = QLabel(label_text)
            layout.addWidget(lbl)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setMaximumHeight(90)
        self._text_edit.setPlaceholderText('{"key": "value"}')
        layout.addWidget(self._text_edit)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red; font-size: 11px;")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        self._text_edit.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self) -> None:
        text = self._text_edit.toPlainText().strip()
        if text:
            try:
                json.loads(text)
                self._error_label.hide()
            except json.JSONDecodeError as exc:
                self._error_label.setText(f"Invalid JSON: {exc}")
                self._error_label.show()
        else:
            self._error_label.hide()
        self.valueChanged.emit()

    def get_value(self) -> Dict[str, Any]:
        """Return the parsed JSON as a dict (empty on invalid / empty input)."""
        text = self._text_edit.toPlainText().strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    def set_value(self, value: Optional[Dict[str, Any]]) -> None:
        """Populate the text area from a dict; empty / None clears it."""
        if value:
            self._text_edit.setPlainText(json.dumps(value, indent=2))
        else:
            self._text_edit.setPlainText("")


# ---------------------------------------------------------------------------
# Main editor
# ---------------------------------------------------------------------------


class ScanAnalyzerEditorPanel(QWidget):
    """Editor for one unified diagnostic YAML.

    Public API:

    * :meth:`load_config(data)` — populate the form from a raw YAML dict
    * :meth:`get_config_dict()` — return the current form state as a
      ``DiagnosticAnalysisConfig``-shaped dict

    Signals
    -------
    config_changed
        Emitted whenever any field in the form is edited.
    """

    config_changed = pyqtSignal()

    # Image-section discriminator values shown in the UI.
    _TYPE_CAMERA = "camera"
    _TYPE_LINE = "line"
    _TYPE_NONE = "(none)"

    # ConfigEditorPanel's internal type strings (a separate vocabulary).
    _CEP_CAMERA = "camera_2d"
    _CEP_LINE = "line_1d"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._scan_section: Optional[SectionWidget] = None
        self._suppress_signals: bool = False
        # Tracks whether the image panel is currently configured to
        # show an editable image config. ``QWidget.isVisible()`` is
        # not reliable until the parent window is shown, so we keep
        # our own flag.
        self._image_active: bool = False

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the four section group boxes.

        No outer scroll area: the image section's embedded
        ConfigEditorPanel handles scrolling for the long CameraConfig
        / Line1DConfig form. The general / image_analyzer / scan
        sections stay visible above and below.
        """
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # ── General ─────────────────────────────────────────────────────
        general_box = QGroupBox("General")
        general_form = QFormLayout(general_box)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Device identifier (e.g. UC_VisaEBeam1)")
        self._name_edit.textChanged.connect(self._on_value_changed)
        general_form.addRow("name:", self._name_edit)

        # Optional output identifier + scalar-key suffix. Empty string
        # = field absent = use the schema defaults (output_name → name,
        # suffix → ""). These are applied by ScanAnalysis at
        # consumption time; ImageAnalysis emits bare keys.
        self._output_name_edit = QLineEdit()
        self._output_name_edit.setPlaceholderText("(defaults to name)")
        self._output_name_edit.setToolTip(
            "Output identifier. Drives BOTH the s-file column prefix AND "
            "the per-analyzer output directory name. Leave blank to "
            "default to 'name'. Set to a different string when you want "
            "outputs (columns and dir) labelled differently from the "
            "input device — e.g. running two BeamAnalyzer variants over "
            "the same camera (output_name=UC_TopView_left / _right)."
        )
        self._output_name_edit.textChanged.connect(self._on_value_changed)
        general_form.addRow("output_name:", self._output_name_edit)

        self._metric_suffix_edit = QLineEdit()
        self._metric_suffix_edit.setPlaceholderText("(none)")
        self._metric_suffix_edit.setToolTip(
            "Suffix appended to every scalar metric key by ScanAnalysis. "
            "Scalar-key-only — does NOT affect directory or file names "
            "(use output_name for those). Leave blank for no suffix. "
            "Use to distinguish post-processed column variants."
        )
        self._metric_suffix_edit.textChanged.connect(self._on_value_changed)
        general_form.addRow("metric_suffix:", self._metric_suffix_edit)

        outer.addWidget(general_box)

        # ── Image Analyzer ──────────────────────────────────────────────
        ia_box = QGroupBox("Image Analyzer")
        ia_form = QFormLayout(ia_box)
        self._class_path_edit = QLineEdit()
        self._class_path_edit.setPlaceholderText(
            "image_analysis.analyzers.beam_analyzer.BeamAnalyzer"
        )
        self._class_path_edit.textChanged.connect(self._on_value_changed)
        ia_form.addRow("class_path:", self._class_path_edit)

        self._ia_kwargs_widget = _JsonDictWidget()
        self._ia_kwargs_widget.valueChanged.connect(self._on_value_changed)
        ia_form.addRow("kwargs:", self._ia_kwargs_widget)
        outer.addWidget(ia_box)

        # ── Image (discriminated, takes any extra vertical space) ───────
        self._image_box = QGroupBox("Image")
        image_layout = QVBoxLayout(self._image_box)

        # Header row: collapse toggle + type discriminator combo. The
        # toggle hides the embedded ConfigEditorPanel below while leaving
        # the type combo visible, so users can see the current image
        # kind at a glance even when the form is folded away.
        type_row = QHBoxLayout()
        self._image_collapse_btn = QToolButton()
        self._image_collapse_btn.setText("▼")
        self._image_collapse_btn.setToolTip("Collapse image config")
        self._image_collapse_btn.setStyleSheet(
            "QToolButton { border: none; font-size: 10px; }"
        )
        self._image_collapse_btn.clicked.connect(self._on_image_collapse_clicked)
        type_row.addWidget(self._image_collapse_btn)

        type_row.addWidget(QLabel("type:"))
        self._image_type_combo = QComboBox()
        self._image_type_combo.addItems(
            [self._TYPE_CAMERA, self._TYPE_LINE, self._TYPE_NONE]
        )
        self._image_type_combo.setToolTip(
            "Image discriminator. 'camera' = CameraConfig (2D); "
            "'line' = Line1DConfig (1D); '(none)' = HASO-style analyzer "
            "with no embedded image config."
        )
        self._image_type_combo.currentTextChanged.connect(self._on_image_type_changed)
        type_row.addWidget(self._image_type_combo)
        type_row.addStretch()
        image_layout.addLayout(type_row)

        # Embedded ConfigEditorPanel — handles its own scrolling for the
        # long CameraConfig/Line1DConfig form. Per #412 the image
        # configs no longer carry a ``name`` field, so there's nothing
        # to suppress — the panel renders identically in standalone
        # and embedded use.
        self._image_panel = ConfigEditorPanel()
        self._image_panel.valueChanged.connect(self._on_value_changed)
        image_layout.addWidget(self._image_panel, stretch=1)

        # User-controlled toggle state. ``_image_active`` (set in
        # ``_apply_image_kind``) is the orthogonal signal that there's
        # actually something to show; ``_refresh_image_panel_visibility``
        # ANDs the two.
        self._image_expanded = True

        outer.addWidget(self._image_box, stretch=1)

        # ── Scan ────────────────────────────────────────────────────────
        self._scan_container = QGroupBox("Scan")
        self._scan_container_layout = QVBoxLayout(self._scan_container)
        self._build_scan_section()
        outer.addWidget(self._scan_container)

    def _build_scan_section(self) -> None:
        """Create the Scan SectionWidget against ScanRuntimeConfig."""
        from scan_analysis.config.diagnostic_models import ScanRuntimeConfig

        if self._scan_section is not None:
            self._scan_container_layout.removeWidget(self._scan_section)
            self._scan_section.deleteLater()
            self._scan_section = None

        self._scan_section = SectionWidget("scan", ScanRuntimeConfig)
        self._scan_section.valueChanged.connect(self._on_value_changed)
        self._scan_container_layout.addWidget(self._scan_section)

    # ------------------------------------------------------------------
    # Image-section discriminator
    # ------------------------------------------------------------------

    def _apply_image_kind(
        self, kind: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Show / hide the image ConfigEditorPanel and load payload into it.

        ``kind`` is one of ``camera`` / ``line`` / ``(none)``.
        ``payload`` is the raw dict for the chosen kind; ``None``
        means the user hasn't supplied anything (build a default-empty
        model).
        """
        if kind == self._TYPE_NONE:
            self._image_active = False
            self._refresh_image_panel_visibility()
            return

        self._image_active = True
        self._refresh_image_panel_visibility()

        if kind == self._TYPE_CAMERA:
            model_cls, cep_type = self._camera_classes()
        elif kind == self._TYPE_LINE:
            model_cls, cep_type = self._line_classes()
        else:
            logger.warning("Unknown image type: %r", kind)
            return

        # Note: we no longer inject the top-level diagnostic name into
        # ``payload["name"]`` here. The embedded ConfigEditorPanel runs
        # in ``embedded_mode=True`` (hides the image-level Name row),
        # so there's no UI surface that would display an injected
        # value. The DiagnosticAnalysisConfig validator still injects
        # ``image.name = name`` on validate-time when image.name is
        # absent, which is the only consumer that cares.
        model = self._build_image_model(model_cls, payload)
        # ConfigEditorPanel.load_config requires a Path; we don't have a
        # standalone file here (the image config lives inside the
        # diagnostic YAML) so pass a sentinel — ConfigEditorPanel only
        # stores it.
        self._image_panel.load_config(model, cep_type, Path("/__embedded__"))

    @staticmethod
    def _camera_classes():
        from image_analysis.config import CameraConfig

        return CameraConfig, ScanAnalyzerEditorPanel._CEP_CAMERA

    @staticmethod
    def _line_classes():
        from image_analysis.config import Line1DConfig

        return Line1DConfig, ScanAnalyzerEditorPanel._CEP_LINE

    @staticmethod
    def _build_image_model(model_cls, payload: Optional[Dict[str, Any]]):
        """Build a CameraConfig/Line1DConfig from a raw dict.

        Strips the ``type`` discriminator before validation (the combo
        is the source of truth). When ``payload`` is missing or
        incomplete, seeds the minimum required fields for each model
        class so an empty form is always constructible — this is what
        gets shown when the user switches the type combo from one
        kind to another.

        Concretely:

        * :class:`CameraConfig` has no required fields after PR #420
          (``name`` is ``Optional[str]``); an empty payload validates.
        * :class:`Line1DConfig` still requires ``data_loading`` (with
          a ``data_type`` selector); we seed ``{data_type: 'csv'}``
          as a sensible starter that the user can change in the form.
        """
        clean = {k: v for k, v in (payload or {}).items() if k != "type"}

        # Per-model minimum required defaults so model_validate succeeds
        # on a switch-to-empty.
        if model_cls.__name__ == "Line1DConfig":
            clean.setdefault("data_loading", {"data_type": "csv"})

        try:
            return model_cls.model_validate(clean)
        except Exception as exc:
            logger.warning(
                "Failed to validate embedded image data as %s; falling back "
                "to default model. Error: %s",
                model_cls.__name__,
                exc,
            )
            # Last-resort fallback: build a model with just the required
            # fields. Both branches must produce a valid model so the UI
            # can show an empty form without crashing.
            if model_cls.__name__ == "Line1DConfig":
                from image_analysis.config.array1d_processing import Data1DConfig

                return model_cls(data_loading=Data1DConfig(data_type="csv"))
            return model_cls()

    def _on_image_type_changed(self, new_kind: str) -> None:
        """Rebuild the image panel for ``new_kind`` (with empty payload)."""
        if self._suppress_signals:
            return
        self._apply_image_kind(new_kind, payload=None)
        self._on_value_changed()

    def _on_image_collapse_clicked(self) -> None:
        """Toggle the user-controlled expanded/collapsed state.

        Flips ``_image_expanded`` and updates panel visibility through
        the same helper that ``_apply_image_kind`` uses, so the
        user-controlled toggle and the discriminator-driven hide stay
        consistent.
        """
        self._image_expanded = not self._image_expanded
        self._refresh_image_panel_visibility()

    def _refresh_image_panel_visibility(self) -> None:
        """Show the image ConfigEditorPanel only when active **and** expanded.

        Two orthogonal signals decide whether the embedded
        ``ConfigEditorPanel`` is visible:

        * ``_image_active`` — set by ``_apply_image_kind`` based on the
          discriminator (False for ``(none)``, True for ``camera`` /
          ``line``).
        * ``_image_expanded`` — user-controlled collapse toggle in the
          header row.

        Also keeps the toggle button's arrow glyph and tooltip in sync.
        """
        visible = self._image_active and self._image_expanded
        self._image_panel.setVisible(visible)
        # Update the toggle glyph + tooltip regardless of active state,
        # so the button always reflects the user's last preference.
        if self._image_expanded:
            self._image_collapse_btn.setText("▼")
            self._image_collapse_btn.setToolTip("Collapse image config")
        else:
            self._image_collapse_btn.setText("▶")
            self._image_collapse_btn.setToolTip("Expand image config")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_config(self, data: Dict[str, Any]) -> None:
        """Populate the form from a raw diagnostic YAML dict.

        Parameters
        ----------
        data : dict
            Diagnostic YAML as parsed by ``load_analyzer_yaml`` —
            shape matching ``DiagnosticAnalysisConfig`` but not yet
            validated.
        """
        self._suppress_signals = True
        try:
            # General
            self._name_edit.setText(str(data.get("name", "")))
            # Optional output identifier + scalar-key suffix. Render
            # absent / null as an empty edit so a round-trip without
            # user changes doesn't introduce these keys in the YAML.
            on = data.get("output_name")
            self._output_name_edit.setText("" if on is None else str(on))
            ms = data.get("metric_suffix")
            self._metric_suffix_edit.setText("" if ms is None else str(ms))

            # Image Analyzer
            ia = data.get("image_analyzer", {})
            if isinstance(ia, str):
                self._class_path_edit.setText(ia)
                self._ia_kwargs_widget.set_value({})
            elif isinstance(ia, dict):
                cp = ia.get("class_path") or ia.get("class") or ""
                self._class_path_edit.setText(str(cp))
                self._ia_kwargs_widget.set_value(ia.get("kwargs"))
            else:
                self._class_path_edit.setText("")
                self._ia_kwargs_widget.set_value({})

            # Image (discriminated)
            image = data.get("image")
            if image is None:
                kind = self._TYPE_NONE
            elif isinstance(image, dict):
                declared = image.get("type", self._TYPE_CAMERA)
                kind = (
                    declared
                    if declared in (self._TYPE_CAMERA, self._TYPE_LINE)
                    else self._TYPE_CAMERA
                )
            else:
                kind = self._TYPE_NONE
            self._image_type_combo.setCurrentText(kind)
            self._apply_image_kind(
                kind, payload=image if isinstance(image, dict) else None
            )

            # Scan — rebuild from scratch so set_values takes a fresh
            # SectionWidget with default state.
            self._build_scan_section()
            scan = data.get("scan")
            if isinstance(scan, dict) and self._scan_section is not None:
                self._scan_section.set_values(scan)
        finally:
            self._suppress_signals = False

    def get_config_dict(self) -> Dict[str, Any]:
        """Return the form's current state as a diagnostic YAML dict.

        The output is shape-compatible with
        :class:`image_analysis.config.DiagnosticAnalysisConfig` and
        ready to hand to ``save_analyzer_yaml``.
        """
        out: Dict[str, Any] = {}

        # General
        name = self._name_edit.text().strip()
        if name:
            out["name"] = name
        # Only emit output_name / metric_suffix when the user actually
        # entered something. Empty edits stay absent from the YAML so
        # the DiagnosticAnalysisConfig defaults take over on load
        # (effective_output_name falls back to ``name``, suffix to "").
        # We deliberately do NOT .strip() — a leading/trailing space in
        # a suffix is unusual but legal, and silently mangling it would
        # be surprising.
        output_name = self._output_name_edit.text()
        if output_name:
            out["output_name"] = output_name
        metric_suffix = self._metric_suffix_edit.text()
        if metric_suffix:
            out["metric_suffix"] = metric_suffix

        # Image Analyzer — emit the bare-string form when there are
        # no extra kwargs, otherwise the verbose dict form.
        class_path = self._class_path_edit.text().strip()
        kwargs = self._ia_kwargs_widget.get_value()
        if class_path:
            if kwargs:
                out["image_analyzer"] = {"class_path": class_path, "kwargs": kwargs}
            else:
                out["image_analyzer"] = class_path

        # Image — re-inject the discriminator from the combo.
        kind = self._image_type_combo.currentText()
        if kind != self._TYPE_NONE and self._image_active:
            values = self._image_panel.get_config_dict() or {}
            # Note: the embedded ConfigEditorPanel runs in
            # ``embedded_mode=True`` and so never emits ``name`` in
            # ``values``. The DiagnosticAnalysisConfig validator
            # injects ``image.name = name`` at validate-time, so
            # downstream consumers still see the field; on disk it
            # stays absent and the top-level name remains the single
            # source of truth.
            # 1D pipeline workaround: ConfigEditorPanel's default
            # pipeline for line configs starts with "data_loading",
            # which is the name of a config section but NOT a valid
            # ``PipelineStepType``. Strip it so the output validates.
            if kind == self._TYPE_LINE:
                pipeline = values.get("pipeline")
                if isinstance(pipeline, dict) and "steps" in pipeline:
                    steps = pipeline.get("steps") or []
                    pipeline["steps"] = [s for s in steps if s != "data_loading"]
            out["image"] = {"type": kind, **values}

        # Scan — strip None-valued Optional fields so absent stays
        # absent on disk. SectionWidget emits ``None`` for every
        # unset Optional and a default-filled struct for every nested
        # BaseModel; left in place they pollute the YAML with values
        # the user never wrote.
        if self._scan_section is not None:
            scan_values = self._scan_section.get_values()
            pruned = _prune_scan_section_values(scan_values)
            if pruned:
                out["scan"] = pruned

        return out

    # ------------------------------------------------------------------
    # Signal routing
    # ------------------------------------------------------------------

    def _on_value_changed(self) -> None:
        """Re-emit ``config_changed`` when any sub-widget reports an edit."""
        if not self._suppress_signals:
            self.config_changed.emit()
