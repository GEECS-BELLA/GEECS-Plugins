"""The Builder tab: devices (B1), variables (B2), the check (B3), interlocks (B4), the save bar (B5).

Pick a device, pick a variable, set the check, choose the interlock it
joins (an existing one groups it, "new interlock…" starts one). Every edit
lands in a :class:`~geecs_interlocks.builder.Draft`; the running
interlocks change only at "Save and reload", which validates the draft,
shows the diff, writes the file and reloads the engine (Master Control
stays connected).

The catalog lookup (the experiment DB) runs on a worker thread, never the
window's: off the lab network it can block for a long time. Live values in
B2 come from the engine for watched devices and from a short-lived preview
subscription for the device on screen.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDoubleValidator, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geecs_schemas import InterlockCondition

from geecs_interlocks.builder import Draft
from geecs_interlocks.catalog import Catalog, StaticCatalog, named_in
from geecs_interlocks.config import SaveRefused, save_target
from geecs_interlocks.conditions import Inputs, describe, evaluate
from geecs_interlocks.engine import Engine
from geecs_interlocks.gui.reloader import Reloader
from geecs_interlocks.gui.theme import MUTED
from geecs_interlocks.sources import PreviewFactory, Source

logger = logging.getLogger(__name__)

#: B1's pseudo-device listing the UDP signals.
EXTERNAL = "External signals (UDP)"
#: B3's kinds: (label, schema kind).
KINDS = (
    ("threshold", "threshold"),
    ("alignment", "alignment"),
    ("settings unchanged", "config_change"),
    ("stale", "stale"),
    ("external signal", "external"),
)
NEW_INTERLOCK = "new interlock…"
OPERATORS = (">=", ">", "<=", "<", "==", "!=")
#: Milliseconds between two refreshes of B2's live values.
VALUES_MS = 500

_CONDITION = TypeAdapter(InterlockCondition)


def check_line(check: Mapping[str, Any]) -> str:
    """One line for a check in B4 (its own words when it does not validate)."""
    try:
        return describe(_CONDITION.validate_python(check))
    except ValidationError:
        return f"{check.get('kind', '?')} (incomplete)"


def footnote(text: str) -> QLabel:
    """A muted line under a panel."""
    label = QLabel(text)
    label.setObjectName("footnote")
    label.setWordWrap(True)
    return label


def _number(field: QLineEdit) -> float | None:
    text = field.text().strip()
    try:
        return float(text) if text else None
    except ValueError:
        return None


def _number_field(value: str = "") -> QLineEdit:
    field = QLineEdit(value)
    validator = QDoubleValidator()
    validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
    field.setValidator(validator)
    return field


class _InterlockTree(QTreeWidget):
    """B4: drops a check onto another interlock to regroup it."""

    moved = pyqtSignal(str, int, str)

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Regroup through the draft instead of moving the item."""
        if event is None:
            return
        dragged = self.currentItem()
        target = self.itemAt(event.position().toPoint())
        event.ignore()
        if dragged is None or target is None or dragged.parent() is None:
            return
        if target.parent() is not None:
            target = target.parent()
        source = dragged.parent()
        assert source is not None
        self.moved.emit(
            source.data(0, Qt.ItemDataRole.UserRole),
            source.indexOfChild(dragged),
            target.data(0, Qt.ItemDataRole.UserRole),
        )


class BuilderTab(QWidget):
    """The Builder tab over *engine*.

    Parameters
    ----------
    engine :
        The running engine (its file is the one saved, and reloaded).
    catalog :
        Where B1 and B2 come from; called on a worker thread.
    make_preview :
        Builds B2's live-value source for a device nothing watches.
    notify :
        Shows one line to the operator.
    """

    changed = pyqtSignal()
    _catalog_ready = pyqtSignal(object, str)

    def __init__(
        self,
        engine: Engine,
        catalog: Catalog,
        make_preview: PreviewFactory | None,
        notify: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self._catalog: Catalog = StaticCatalog(named_in(engine.interlocks))
        self._make_preview = make_preview
        self._preview: Source | None = None
        self._preview_device: str | None = None
        self._notify = notify
        self.draft = Draft(engine.interlocks)
        self._editing: tuple[str, int] | None = None
        self._reloader = Reloader(engine, self)

        self._build_devices()
        self._build_variables()
        self._build_check()
        self._build_interlocks()
        self._build_save_bar()

        left_top = QHBoxLayout()
        left_top.addWidget(self.devices_box, 1)
        left_top.addWidget(self.variables_box, 1)
        left = QVBoxLayout()
        left.addLayout(left_top, 1)
        left.addWidget(self.check_box, 0)
        body = QHBoxLayout()
        body.addLayout(left, 2)
        body.addWidget(self.interlocks_box, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(self.save_bar)

        self._catalog_ready.connect(self._take_catalog)
        threading.Thread(
            target=self._load_catalog, args=(catalog,), daemon=True
        ).start()
        self.values_timer = QTimer(self)
        self.values_timer.timeout.connect(self._refresh_values)
        self.values_timer.start(VALUES_MS)
        self._redraw_all()

    # -- B1 devices -----------------------------------------------------------

    def _build_devices(self) -> None:
        self.devices_box = QGroupBox("Devices")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search, or type a name and press Enter")
        self.search.setClearButtonEnabled(True)
        self.device_list = QTreeWidget()
        self.device_list.setHeaderHidden(True)
        self.device_list.setColumnCount(2)
        self.device_list.setRootIsDecorated(False)
        self.device_list.setColumnWidth(0, 170)
        self.devices_note = footnote("Loading the experiment's devices…")
        box = QVBoxLayout(self.devices_box)
        box.addWidget(self.search)
        box.addWidget(self.device_list, 1)
        box.addWidget(self.devices_note)
        self.search.textChanged.connect(self._draw_devices)
        self.search.returnPressed.connect(self._add_typed_device)
        self.device_list.currentItemChanged.connect(lambda *_: self._draw_variables())

    def _load_catalog(self, catalog: Catalog) -> None:
        try:
            devices = catalog.devices()
            self._catalog_ready.emit(
                catalog, f"{len(devices)} devices in the experiment."
            )
        except Exception:  # the DB is out of reach: offer the file's devices
            logger.exception("the experiment's device list is unavailable")
            self._catalog_ready.emit(
                None, "The experiment DB is out of reach: showing the file's devices."
            )

    def _take_catalog(self, catalog: Catalog | None, note: str) -> None:
        if catalog is not None:
            self._catalog = catalog
        self.devices_note.setText(note + " Watched ones are marked.")
        self._draw_devices()

    def _add_typed_device(self) -> None:
        name = self.search.text().strip()
        if not name or name in self._catalog.devices():
            return
        if isinstance(self._catalog, StaticCatalog):
            self._catalog = self._catalog.merged({name: []})
        self.search.clear()
        self._draw_devices(select=name)

    def _all_devices(self) -> list[str]:
        named = set(self.draft.watched())
        return sorted(set(self._catalog.devices()) | named)

    def _draw_devices(self, *_: object, select: str | None = None) -> None:
        keep = select or self.selected_device()
        needle = self.search.text().strip().lower()
        watched = self.draft.watched()
        self.device_list.clear()
        for device in self._all_devices():
            if needle and needle not in device.lower():
                continue
            item = QTreeWidgetItem([device, "watched" if device in watched else ""])
            item.setForeground(1, QBrush(QColor(MUTED)))
            self.device_list.addTopLevelItem(item)
            if device == keep:
                self.device_list.setCurrentItem(item)
        ports = self.draft.ports()
        external = QTreeWidgetItem([EXTERNAL, "watched" if ports else ""])
        for column in (0, 1):
            external.setForeground(column, QBrush(QColor(MUTED)))
        self.device_list.addTopLevelItem(external)
        if keep == EXTERNAL:
            self.device_list.setCurrentItem(external)

    def selected_device(self) -> str | None:
        """The device picked in B1."""
        item = self.device_list.currentItem()
        return None if item is None else item.text(0)

    # -- B2 variables ---------------------------------------------------------

    def _build_variables(self) -> None:
        self.variables_box = QGroupBox("Variables")
        self.watched_only = QCheckBox("Watched only")
        self.variable_list = QTreeWidget()
        self.variable_list.setHeaderHidden(True)
        self.variable_list.setColumnCount(2)
        self.variable_list.setRootIsDecorated(False)
        self.variable_list.setColumnWidth(0, 140)
        box = QVBoxLayout(self.variables_box)
        box.addWidget(self.watched_only)
        box.addWidget(self.variable_list, 1)
        box.addWidget(
            footnote(
                "Watched ones name their interlock. Tick several for a settings check."
            )
        )
        self.watched_only.toggled.connect(lambda *_: self._draw_variables())
        self.variable_list.currentItemChanged.connect(
            lambda *_: self._variable_picked()
        )
        self.variable_list.itemChanged.connect(lambda *_: self._update_reading())

    def _draw_variables(self) -> None:
        device = self.selected_device()
        self.variable_list.blockSignals(True)
        self.variable_list.clear()
        if device == EXTERNAL:
            for port, names in sorted(self.draft.ports().items()):
                item = QTreeWidgetItem([f"UDP {port}", ", ".join(names)])
                item.setData(0, Qt.ItemDataRole.UserRole, port)
                self.variable_list.addTopLevelItem(item)
        elif device is not None:
            watched = self.draft.watched().get(device, {})
            variables = sorted(set(self._catalog.variables(device)) | set(watched))
            for variable in variables:
                if self.watched_only.isChecked() and variable not in watched:
                    continue
                item = QTreeWidgetItem([variable, ""])
                item.setData(0, Qt.ItemDataRole.UserRole, variable)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.setForeground(1, QBrush(QColor(MUTED)))
                self.variable_list.addTopLevelItem(item)
        self.variable_list.blockSignals(False)
        self._start_preview(device)
        self._refresh_values()
        self._variable_picked()

    def selected_variable(self) -> str | None:
        """The variable picked in B2."""
        item = self.variable_list.currentItem()
        if item is None or self.selected_device() == EXTERNAL:
            return None
        return item.data(0, Qt.ItemDataRole.UserRole)

    def ticked_variables(self) -> list[str]:
        """The variables ticked in B2 (a settings check holds them)."""
        return [
            self.variable_list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
            for i in range(self.variable_list.topLevelItemCount())
            if self.variable_list.topLevelItem(i).checkState(0) == Qt.CheckState.Checked
        ]

    def _start_preview(self, device: str | None) -> None:
        if device == self._preview_device:
            return
        if self._preview is not None:
            self._preview.stop()
            self._preview = None
        self._preview_device = device
        if device is None or device == EXTERNAL or self._make_preview is None:
            return
        if self.engine.reading(device).updated is not None:
            return  # watched already: the engine has its values
        variables = set(self._catalog.variables(device))
        if variables:
            self._preview = self._make_preview({device: variables})
            self._preview.start()

    def live_value(self, device: str, variable: str) -> Any:
        """*device*'s latest *variable*, from the engine or the preview (``None``: none)."""
        value = self.engine.reading(device).values.get(variable)
        if (
            value is None
            and self._preview is not None
            and device == self._preview_device
        ):
            value = self._preview.reading(device).values.get(variable)
        return value

    def _refresh_values(self) -> None:
        device = self.selected_device()
        if device is None or device == EXTERNAL:
            return
        watched = self.draft.watched().get(device, {})
        for i in range(self.variable_list.topLevelItemCount()):
            item = self.variable_list.topLevelItem(i)
            variable = item.data(0, Qt.ItemDataRole.UserRole)
            value = self.live_value(device, variable)
            text = (
                "—"
                if value is None
                else f"{value:.4g}"
                if isinstance(value, float)
                else str(value)
            )
            names = watched.get(variable)
            item.setText(1, text + (f" · {', '.join(names)}" if names else ""))
        self._update_reading()

    def _variable_picked(self) -> None:
        if self._editing is None:
            self._suggest_kind()
        self._update_title()
        self._update_reading()

    # -- B3 the check ---------------------------------------------------------

    def _build_check(self) -> None:
        self.check_box = QGroupBox("Check")
        self.kind = QComboBox()
        for label, _ in KINDS:
            self.kind.addItem(label)
        self.pages = QStackedWidget()
        for build in (
            self._threshold_page,
            self._alignment_page,
            self._holds_page,
            self._stale_page,
            self._external_page,
        ):
            self.pages.addWidget(build())
        self._build_check_bottom()

    def _threshold_page(self) -> QWidget:
        threshold = QWidget()
        form = QFormLayout(threshold)
        self.operator = QComboBox()
        self.operator.addItems(OPERATORS)
        self.limit = _number_field()
        row = QHBoxLayout()
        row.addWidget(self.operator)
        row.addWidget(self.limit, 1)
        form.addRow("Unsafe if", row)
        self.hysteresis = _number_field("0")
        form.addRow("Hysteresis", self.hysteresis)
        return threshold

    def _alignment_page(self) -> QWidget:
        alignment = QWidget()
        form = QFormLayout(alignment)
        self.target_device = QComboBox()
        self.target_device.setEditable(True)
        self.target_variable = QComboBox()
        self.target_variable.setEditable(True)
        self.tolerance = _number_field()
        self.align_hysteresis = _number_field("0")
        form.addRow("Target device", self.target_device)
        form.addRow("Target variable", self.target_variable)
        form.addRow("Tolerance", self.tolerance)
        form.addRow("Hysteresis", self.align_hysteresis)
        return alignment

    def _holds_page(self) -> QWidget:
        holds = QWidget()
        form = QFormLayout(holds)
        self.holds = QLabel()
        self.holds.setWordWrap(True)
        form.addRow("Holds", self.holds)
        return holds

    def _stale_page(self) -> QWidget:
        stale = QWidget()
        form = QFormLayout(stale)
        self.max_age = _number_field()
        self.max_age.setPlaceholderText("empty: trip on a disconnect only")
        form.addRow("Max age (s)", self.max_age)
        return stale

    def _external_page(self) -> QWidget:
        external = QWidget()
        form = QFormLayout(external)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(5050)
        self.source_name = QLineEdit()
        self.source_name.setPlaceholderText("stage controller")
        self.packet_age = _number_field()
        self.packet_age.setPlaceholderText("empty: the sender sends only on a change")
        form.addRow("UDP port", self.port)
        form.addRow("Sender", self.source_name)
        form.addRow("Max age (s)", self.packet_age)
        return external

    def _build_check_bottom(self) -> None:
        self.add_to = QComboBox()
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("Name of the new interlock")
        right = QFormLayout()
        right.addRow("Kind", self.kind)
        right.addRow("Add to", self.add_to)
        right.addRow("Name", self.new_name)
        fields = QHBoxLayout()
        fields.addWidget(self.pages, 1)
        fields.addLayout(right, 1)

        self.reading = QLabel()
        self.reading.setObjectName("footnote")
        self.update_button = QPushButton("Update selected")
        self.add_button = QPushButton("Add check")
        self.add_button.setObjectName("primary")
        buttons = QHBoxLayout()
        buttons.addWidget(self.reading, 1)
        buttons.addWidget(self.update_button)
        buttons.addWidget(self.add_button)

        box = QVBoxLayout(self.check_box)
        box.addLayout(fields)
        box.addLayout(buttons)
        box.addWidget(
            footnote('"Add to" an existing interlock groups the check with it.')
        )

        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.add_to.currentIndexChanged.connect(self._add_to_changed)
        self.target_device.currentTextChanged.connect(self._draw_target_variables)
        for field in (self.limit, self.hysteresis, self.tolerance, self.max_age):
            field.textChanged.connect(lambda *_: self._update_reading())
        self.operator.currentIndexChanged.connect(lambda *_: self._update_reading())
        self.add_button.clicked.connect(self.add_check)
        self.update_button.clicked.connect(self.update_check)

    def _schema_kind(self) -> str:
        return KINDS[self.kind.currentIndex()][1]

    def _set_kind(self, kind: str) -> None:
        index = next(i for i, (_, k) in enumerate(KINDS) if k == kind)
        self.kind.setCurrentIndex(index)

    def _suggest_kind(self) -> None:
        if self.selected_device() == EXTERNAL:
            self._set_kind("external")
            item = self.variable_list.currentItem()
            if item is not None:
                self.port.setValue(int(item.data(0, Qt.ItemDataRole.UserRole)))
        elif self._schema_kind() == "external":
            self._set_kind("threshold")

    def _kind_changed(self) -> None:
        self.pages.setCurrentIndex(self.kind.currentIndex())
        if self._schema_kind() == "alignment":
            self._draw_target_devices()
        self._update_title()
        self._update_reading()

    def _draw_target_devices(self) -> None:
        device = self.selected_device() or ""
        current = self.target_device.currentText() or device
        self.target_device.blockSignals(True)
        self.target_device.clear()
        self.target_device.addItems(self._all_devices())
        self.target_device.setCurrentText(current)
        self.target_device.blockSignals(False)
        self._draw_target_variables()

    def _draw_target_variables(self, *_: object) -> None:
        device = self.target_device.currentText()
        current = self.target_variable.currentText()
        self.target_variable.clear()
        self.target_variable.addItems(self._catalog.variables(device))
        self.target_variable.setCurrentText(current)

    def _add_to_changed(self) -> None:
        self.new_name.setEnabled(self.add_to.currentText() == NEW_INTERLOCK)

    def _update_title(self) -> None:
        device, variable = self.selected_device(), self.selected_variable()
        kind = self._schema_kind()
        if kind == "external":
            title = "Check on an external signal"
        elif kind in ("stale", "config_change"):
            title = f"Check on {device}" if device and device != EXTERNAL else "Check"
        elif device and variable:
            title = f"Check on {device} {variable}"
        else:
            title = "Check: pick a device and a variable"
        if self._editing is not None:
            title += f" — editing {self._editing[0]}"
        self.check_box.setTitle(title)
        self.holds.setText(
            ", ".join(self.ticked_variables()) or "tick the settings in Variables"
        )

    def form_check(self) -> dict[str, Any]:
        """The check B3 describes now, as plain data (may not validate yet)."""
        build = {
            "threshold": self._threshold_check,
            "alignment": self._alignment_check,
            "config_change": self._holds_check,
            "stale": self._stale_check,
            "external": self._external_check,
        }[self._schema_kind()]
        return {k: v for k, v in build().items() if v is not None}

    def _threshold_check(self) -> dict[str, Any]:
        return {
            "kind": "threshold",
            "device": self.selected_device() or "",
            "variable": self.selected_variable() or "",
            "operator": self.operator.currentText(),
            "limit": _number(self.limit),
            "hysteresis": _number(self.hysteresis) or None,
        }

    def _alignment_check(self) -> dict[str, Any]:
        device = self.selected_device() or ""
        target = self.target_device.currentText()
        return {
            "kind": "alignment",
            "device": device,
            "variable": self.selected_variable() or "",
            "target_device": target if target and target != device else None,
            "target_variable": self.target_variable.currentText(),
            "tolerance": _number(self.tolerance),
            "hysteresis": _number(self.align_hysteresis) or None,
        }

    def _holds_check(self) -> dict[str, Any]:
        return {
            "kind": "config_change",
            "device": self.selected_device() or "",
            "variables": self.ticked_variables(),
        }

    def _stale_check(self) -> dict[str, Any]:
        return {
            "kind": "stale",
            "device": self.selected_device() or "",
            "max_age_s": _number(self.max_age),
        }

    def _external_check(self) -> dict[str, Any]:
        return {
            "kind": "external",
            "port": self.port.value(),
            "source": self.source_name.text().strip() or None,
            "max_age_s": _number(self.packet_age),
        }

    def _update_reading(self) -> None:
        """B3's "Reads now" line: the check evaluated on the live values."""
        self._update_title()
        try:
            condition = _CONDITION.validate_python(self.form_check())
        except ValidationError:
            self.reading.setText("Fill in the check to see what it reads now.")
            return
        if condition.kind not in ("threshold", "alignment"):
            self.reading.setText(describe(condition))
            return
        value = self.live_value(condition.device, condition.variable)
        target = None
        if condition.kind == "alignment":
            target = self.live_value(
                condition.target_device or condition.device, condition.target_variable
            )
        verdict = evaluate(condition, Inputs(value=value, target=target), tripped=False)
        if value is None:
            self.reading.setText(f"{describe(condition)} · no live value yet")
            return
        word = verdict.diagnostic if verdict.unsafe else "safe"
        shown = f"{value:.4g}" if isinstance(value, float) else str(value)
        self.reading.setText(f"Reads now: {shown} → {word}")

    def _validated(self) -> dict[str, Any] | None:
        check = self.form_check()
        try:
            _CONDITION.validate_python(check)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"][1:]) or "check"
            self._notify(f"Not added: {where}: {first['msg']}")
            return None
        return check

    def add_check(self) -> None:
        """Add B3's check to the interlock chosen in "Add to"."""
        check = self._validated()
        if check is None:
            return
        to = self.add_to.currentText()
        try:
            name, index = self.draft.add_check(
                check,
                None if to == NEW_INTERLOCK else to,
                new_name=self.new_name.text(),
            )
        except ValueError as exc:
            self._notify(f"Not added: {exc}")
            return
        self.new_name.clear()
        self._notify(f"Added to {name}: {check_line(check)}")
        self._editing = (name, index)
        self._redraw_all()

    def update_check(self) -> None:
        """Replace the check selected in B4 with B3's."""
        if self._editing is None:
            self._notify("Select a check under Interlocks to update it.")
            return
        check = self._validated()
        if check is None:
            return
        self.draft.replace_check(*self._editing, check)
        self._notify(f"Updated {self._editing[0]}: {check_line(check)}")
        self._redraw_all()

    def load_check(self, name: str, index: int) -> None:
        """Show check *index* of *name* in B1–B3, for editing."""
        check = self.draft.check(name, index)
        self._editing = (name, index)
        kind = check["kind"]
        device = EXTERNAL if kind == "external" else check.get("device")
        self._draw_devices(select=device)
        self._draw_variables()
        self._set_kind(kind)
        if kind in ("threshold", "alignment"):
            self._select_variable(check["variable"])
        if kind == "threshold":
            self.operator.setCurrentText(check["operator"])
            self.limit.setText(f"{check['limit']:g}")
            self.hysteresis.setText(f"{check.get('hysteresis', 0):g}")
        elif kind == "alignment":
            self.target_device.setCurrentText(
                check.get("target_device") or check["device"]
            )
            self.target_variable.setCurrentText(check["target_variable"])
            self.tolerance.setText(f"{check['tolerance']:g}")
            self.align_hysteresis.setText(f"{check.get('hysteresis', 0):g}")
        elif kind == "config_change":
            for i in range(self.variable_list.topLevelItemCount()):
                item = self.variable_list.topLevelItem(i)
                held = item.data(0, Qt.ItemDataRole.UserRole) in check["variables"]
                item.setCheckState(
                    0, Qt.CheckState.Checked if held else Qt.CheckState.Unchecked
                )
        elif kind == "stale":
            self.max_age.setText(
                "" if check.get("max_age_s") is None else f"{check['max_age_s']:g}"
            )
        elif kind == "external":
            self.port.setValue(int(check["port"]))
            self.source_name.setText(check.get("source", ""))
            age = check.get("max_age_s")
            self.packet_age.setText("" if age is None else f"{age:g}")
        self.add_to.setCurrentText(name)
        self._update_reading()

    def _select_variable(self, variable: str) -> None:
        for i in range(self.variable_list.topLevelItemCount()):
            item = self.variable_list.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == variable:
                self.variable_list.setCurrentItem(item)
                return

    # -- B4 interlocks --------------------------------------------------------

    def _build_interlocks(self) -> None:
        self.interlocks_box = QGroupBox("Interlocks")
        self.tree = _InterlockTree()
        self.tree.setHeaderHidden(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        box = QVBoxLayout(self.interlocks_box)
        box.addWidget(self.tree, 1)
        box.addWidget(
            footnote(
                "Tick = armed. Drag a check onto another interlock to regroup. "
                "Right-click to rename, set settle time, ungroup or delete."
            )
        )
        self.tree.itemChanged.connect(self._tree_ticked)
        self.tree.currentItemChanged.connect(lambda *_: self._tree_selected())
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.moved.connect(self._regroup)

    def _draw_interlocks(self) -> None:
        expanded = {
            self.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
            for i in range(self.tree.topLevelItemCount())
            if self.tree.topLevelItem(i).isExpanded()
        }
        self.tree.blockSignals(True)
        self.tree.clear()
        for entry in self.draft.interlocks:
            settle = entry.get("settle_s")
            label = entry["name"] + (f"   · settle {settle:g} s" if settle else "")
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, entry["name"])
            item.setFlags(
                (
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsDropEnabled
                )
                & ~Qt.ItemFlag.ItemIsDragEnabled
            )
            enabled = entry.get("enabled", True)
            item.setCheckState(
                0, Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
            )
            if not enabled:
                item.setForeground(0, QBrush(QColor(MUTED)))
            for index, check in enumerate(entry["conditions"]):
                child = QTreeWidgetItem([check_line(check)])
                child.setData(0, Qt.ItemDataRole.UserRole, index)
                child.setFlags(
                    (child.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                    & ~Qt.ItemFlag.ItemIsDropEnabled
                )
                item.addChild(child)
                if self._editing == (entry["name"], index):
                    self.tree.setCurrentItem(child)
            self.tree.addTopLevelItem(item)
            item.setExpanded(
                entry["name"] in expanded
                or len(entry["conditions"]) > 1
                or (self._editing is not None and self._editing[0] == entry["name"])
            )
        self.tree.blockSignals(False)

    def _tree_ticked(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.parent() is not None:
            return
        name = item.data(0, Qt.ItemDataRole.UserRole)
        self.draft.set_enabled(name, item.checkState(0) == Qt.CheckState.Checked)
        self._redraw_all()

    def _tree_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None or item.parent() is None:
            return
        parent = item.parent()
        assert parent is not None
        self.load_check(
            parent.data(0, Qt.ItemDataRole.UserRole), parent.indexOfChild(item)
        )

    def _regroup(self, name: str, index: int, to: str) -> None:
        if name == to:
            return
        self._editing = self.draft.move_check(name, index, to)
        self._notify(f"Moved a check from {name} to {to}")
        self._redraw_all()

    def _tree_menu(self, point: Any) -> None:
        item = self.tree.itemAt(point)
        if item is None:
            return
        menu = QMenu(self)
        if item.parent() is None:
            name = item.data(0, Qt.ItemDataRole.UserRole)
            menu.addAction("Rename…", lambda: self._rename(name))
            menu.addAction("Settle time…", lambda: self._settle(name))
            menu.addAction(
                "Ungroup", lambda: self._edit(lambda: self.draft.ungroup(name))
            )
            menu.addAction(
                "Move up", lambda: self._edit(lambda: self.draft.move(name, -1))
            )
            menu.addAction(
                "Move down", lambda: self._edit(lambda: self.draft.move(name, 1))
            )
            menu.addSeparator()
            menu.addAction(
                "Delete interlock", lambda: self._edit(lambda: self.draft.remove(name))
            )
        else:
            parent = item.parent()
            assert parent is not None
            name = parent.data(0, Qt.ItemDataRole.UserRole)
            index = parent.indexOfChild(item)
            move = menu.addMenu("Move to")
            assert move is not None
            for other in self.draft.names():
                if other != name:
                    move.addAction(other, lambda o=other: self._regroup(name, index, o))
            menu.addAction(
                "Remove check",
                lambda: self._edit(lambda: self.draft.remove_check(name, index)),
            )
        menu.exec(self.tree.viewport().mapToGlobal(point))

    def _edit(self, change: Callable[[], object]) -> None:
        try:
            change()
        except (KeyError, ValueError) as exc:
            self._notify(str(exc))
        self._editing = None
        self._redraw_all()

    def _rename(self, name: str) -> None:
        new, ok = QInputDialog.getText(self, "Rename interlock", "Name", text=name)
        if ok:
            self._edit(lambda: self.draft.rename(name, new))

    def _settle(self, name: str) -> None:
        entry = next(e for e in self.draft.interlocks if e["name"] == name)
        seconds, ok = QInputDialog.getDouble(
            self,
            "Settle time",
            "Seconds clear before OK",
            entry.get("settle_s", 0.0),
            0.0,
            3600.0,
            1,
        )
        if ok:
            self._edit(lambda: self.draft.set_settle(name, seconds))

    # -- B5 the save bar ------------------------------------------------------

    def _build_save_bar(self) -> None:
        self.save_bar = QWidget()
        self.configuration = QComboBox()
        self.configuration.setMinimumWidth(160)
        open_button = QPushButton("Open…")
        revert = QPushButton("Revert")
        save_as = QPushButton("Save as…")
        self.save_button = QPushButton("Save and reload")
        self.save_button.setObjectName("primary")
        bar = QHBoxLayout(self.save_bar)
        bar.setContentsMargins(0, 4, 0, 0)
        bar.addWidget(QLabel("Configuration"))
        bar.addWidget(self.configuration)
        bar.addWidget(open_button)
        bar.addStretch(1)
        bar.addWidget(revert)
        bar.addWidget(save_as)
        bar.addWidget(self.save_button)
        self.configuration.activated.connect(self._configuration_picked)
        open_button.clicked.connect(self.open_file)
        revert.clicked.connect(self.revert)
        save_as.clicked.connect(self.save_as)
        self.save_button.clicked.connect(self.save)

    def folder(self) -> Path | None:
        """The folder of the running file: where its sibling configurations are."""
        return None if self.engine.path is None else self.engine.path.parent

    def _draw_configurations(self) -> None:
        self.configuration.blockSignals(True)
        self.configuration.clear()
        folder = self.folder()
        if folder is not None and folder.is_dir():
            for path in sorted(folder.glob("*.yaml")):
                self.configuration.addItem(path.stem, str(path))
        if self.engine.path is not None:
            self.configuration.setCurrentText(self.engine.path.stem)
        self.configuration.blockSignals(False)

    def _configuration_picked(self, index: int) -> None:
        path = self.configuration.itemData(index)
        if path and Path(path) != self.engine.path:
            self.switch_to(Path(path))

    def _discard_ok(self) -> bool:
        if not self.draft.dirty:
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes", "Discard the changes in the builder?"
        )
        return answer == QMessageBox.StandardButton.Yes

    def switch_to(self, path: Path) -> None:
        """Run the configuration at *path* (the builder's draft starts over from it)."""
        if not self._discard_ok():
            self._draw_configurations()
            return
        self._reload(path, "Opened")

    def _reload(self, path: Path, what: str) -> None:
        """Reload onto *path* off the window's thread; the draft follows when it is done."""
        self._notify(f"{what} {path.name}; reloading…")

        def finished(message: str) -> None:
            self._notify(f"{what} {path.name}; {message}")
            self.sync_from_engine(force=True)

        self._reloader.reload(path, finished)

    def open_file(self) -> None:
        """Pick a configuration file to run."""
        start = str(self.folder() or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open configuration", start, "YAML (*.yaml)"
        )
        if path:
            self.switch_to(Path(path))

    def revert(self) -> None:
        """Throw away the builder's changes."""
        self.draft.revert()
        self._editing = None
        self._redraw_all()
        self._notify("Reverted to the running configuration")

    def _confirm_diff(self, path: Path | None) -> bool:
        try:
            diff = self.draft.diff(path)
        except ValidationError as exc:
            QMessageBox.warning(
                self, "Not saved", f"The interlocks do not validate:\n\n{exc}"
            )
            return False
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Save {path.name if path else 'configuration'}")
        text = QPlainTextEdit(diff or "No changes.")
        text.setObjectName("log")
        text.setReadOnly(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel("These changes will be written, then the server reloads:")
        )
        layout.addWidget(text)
        layout.addWidget(buttons)
        dialog.resize(640, 420)
        return dialog.exec() == QDialog.DialogCode.Accepted

    def save(self) -> None:
        """Validate, show the diff, write the running file and reload."""
        path = self.engine.path
        if path is None:
            self.save_as()
            return
        if self._confirm_diff(path):
            self._write(path)

    def save_as(self) -> None:
        """Save the draft as a new configuration beside the running one, and run it."""
        folder = self.folder()
        if folder is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save configuration", str(Path.home()), "YAML (*.yaml)"
            )
            if not chosen:
                return
            path = Path(chosen)
        else:
            name, ok = QInputDialog.getText(self, "Save as", "Configuration name")
            if not ok or not name.strip():
                return
            path = folder / f"{name.strip()}.yaml"
            if path.exists():
                answer = QMessageBox.question(self, "Replace", f"Replace {path.name}?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
        if self._confirm_diff(path):
            self._write(path)

    def _write(self, path: Path) -> None:
        try:
            target = save_target(path)
            self.draft.save(target)
        except (ValidationError, OSError, SaveRefused) as exc:
            QMessageBox.warning(self, "Not saved", str(exc))
            return
        self._reload(target, f"Saved {target.name}")

    # -- keeping up -----------------------------------------------------------

    def sync_from_engine(self, force: bool = False) -> None:
        """Start the draft over from the running configuration (unless it holds edits)."""
        if self.draft.dirty and not force:
            return
        self.draft = Draft(self.engine.interlocks)
        self._editing = None
        self._redraw_all()

    def _redraw_all(self) -> None:
        names = self.draft.names()
        current = self.add_to.currentText()
        self.add_to.blockSignals(True)
        self.add_to.clear()
        self.add_to.addItems([*names, NEW_INTERLOCK])
        self.add_to.setCurrentText(current if current in names else NEW_INTERLOCK)
        self.add_to.blockSignals(False)
        self._add_to_changed()
        self._draw_devices()
        self._draw_interlocks()
        self._draw_configurations()
        self._update_title()
        self.changed.emit()

    def close_tab(self) -> None:
        """Stop the preview subscription (the window is closing)."""
        self._start_preview(None)


__all__ = ["EXTERNAL", "KINDS", "NEW_INTERLOCK", "BuilderTab", "check_line"]
