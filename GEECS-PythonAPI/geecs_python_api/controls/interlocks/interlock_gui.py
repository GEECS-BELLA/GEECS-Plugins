"""Tkinter operator interface for a configured interlock constructor."""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections.abc import Callable, Iterable
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..interface.geecs_database import GeecsDatabase
from .interlock_constructor import InterlockConstructor
from .interlock_config import load_configuration, save_configuration
from .monitor_conditions import (
    AlignmentCheck,
    ConfigurationChangeCheck,
    ExternalStatusCheck,
    MultiCheck,
    ThresholdCheck,
)

logger = logging.getLogger(__name__)


def _local_bind_host() -> str:
    """Return the local IPv4 address used for outbound network traffic."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        probe.close()


def _load_experiment_names() -> list[str]:
    """Load experiment names using the configured GEECS database connection."""
    db = GeecsDatabase._get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(f"SELECT name FROM {GeecsDatabase.name}.expt ORDER BY name;")
        return [row["name"] for row in cursor.fetchall()]
    finally:
        GeecsDatabase._close_db(db, cursor)


class _ToolTip:
    """Display a short explanation while the pointer rests over a widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _schedule(self, _event: Any = None) -> None:
        self._hide()
        self.after_id = self.widget.after(500, self._show)

    def _show(self) -> None:
        if self.window is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.window,
            text=self.text,
            justify="left",
            background="#fffbe6",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
        ).pack()

    def _hide(self, _event: Any = None) -> None:
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class InterlockGUI:
    """Display and control the lifecycle of an ``InterlockConstructor``.

    The GUI does not evaluate checks itself. The constructor's monitor threads
    remain the source of truth for safety state and diagnostics.
    """

    def __init__(
        self,
        interlock: InterlockConstructor,
        armable_checks: Iterable[Any] = (),
        title: str | None = None,
        refresh_ms: int = 250,
    ) -> None:
        self.interlock = interlock
        self.armable_checks = tuple(armable_checks)
        self.refresh_ms = refresh_ms
        self._worker: threading.Thread | None = None
        self._device_variables: dict[str, list[str]] = {}
        self._device_display_to_alias: dict[str, str] = {}
        self._device_alias_to_display: dict[str, str] = {}
        self._condition_rows: dict[str, tuple[Any, str, ttk.Entry]] = {}
        self._condition_catalog: dict[str, Any] = {}
        self._leaf_condition_rows: dict[str, Any] = {}

        self.root = tk.Tk()
        self.root.title(title or f"{interlock.exp_name} Interlocks")
        self.root.geometry("980x620")
        self.root.minsize(760, 460)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_var = tk.StringVar(value="Stopped")
        self.overall_status_var = tk.StringVar(value="UNKNOWN")
        self.overall_status_detail_var = tk.StringVar(value="No monitor state yet")
        self._build_widgets()
        self._refresh()

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(16, 14, 16, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text=f"{self.interlock.exp_name} interlocks",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var).grid(
            row=0, column=1, padx=(12, 0), sticky="e"
        )
        help_button = ttk.Button(header, text="Help", command=self._show_help)
        help_button.grid(row=0, column=2, padx=(12, 0), sticky="e")
        _ToolTip(help_button, "Open a quick guide to the interlock workflow.")

        indicator = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        indicator.grid(row=1, column=0, sticky="ew")
        indicator.columnconfigure(2, weight=1)
        self.status_lamp = tk.Canvas(
            indicator, width=30, height=30, highlightthickness=0
        )
        self.status_lamp.grid(row=0, column=0, padx=(0, 8))
        self.status_lamp_id = self.status_lamp.create_oval(
            4, 4, 26, 26, fill="#9a7415", outline="#5f490d", width=2
        )
        ttk.Label(
            indicator,
            textvariable=self.overall_status_var,
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(indicator, textvariable=self.overall_status_detail_var).grid(
            row=0, column=2, padx=(12, 0), sticky="w"
        )
        _ToolTip(
            self.status_lamp,
            "Green means all monitors are safe. Red means an interlock is active. "
            "Amber means the state is unknown or the server is stopped.",
        )

        body = ttk.Notebook(self.root, padding=(16, 4, 16, 12))
        body.grid(row=2, column=0, sticky="nsew")
        self.root.rowconfigure(2, weight=1)
        monitors_tab = ttk.Frame(body, padding=(0, 4, 0, 0))
        devices_tab = ttk.Frame(body, padding=(0, 4, 0, 0))
        body.add(monitors_tab, text="Interlocks")
        body.add(devices_tab, text="Devices")
        self._build_monitors_tab(monitors_tab)
        self._build_devices_tab(devices_tab)

    def _build_monitors_tab(self, body: ttk.Frame) -> None:
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        columns = ("monitor", "state", "diagnostic")
        self.monitor_table = ttk.Treeview(body, columns=columns, show="headings")
        self.monitor_table.heading("monitor", text="Monitor")
        self.monitor_table.heading("state", text="State")
        self.monitor_table.heading("diagnostic", text="Diagnostic")
        self.monitor_table.column("monitor", width=210, anchor="w")
        self.monitor_table.column("state", width=100, anchor="center")
        self.monitor_table.column("diagnostic", width=420, anchor="w")
        self.monitor_table.tag_configure("safe", foreground="#176b3a")
        self.monitor_table.tag_configure("unsafe", foreground="#a12622")
        self.monitor_table.tag_configure("unknown", foreground="#6b5d16")
        self.monitor_table.tag_configure("leaf", foreground="#555555")
        self.monitor_table.bind("<Double-1>", self._on_monitor_table_double_click)
        self.monitor_table.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            body, orient="vertical", command=self.monitor_table.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.monitor_table.configure(yscrollcommand=scrollbar.set)
        monitor_controls = ttk.Frame(body)
        monitor_controls.grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="ew")
        create_multi_button = ttk.Button(
            monitor_controls,
            text="Create multi-check",
            command=self._open_multi_check_dialog,
        )
        create_multi_button.pack(side="left", padx=(0, 8))
        _ToolTip(
            create_multi_button, "Combine two or more existing checks with OR logic."
        )
        remove_monitor_button = ttk.Button(
            monitor_controls,
            text="Remove selected interlock",
            command=self._remove_selected_monitor,
        )
        remove_monitor_button.pack(side="left")
        _ToolTip(
            remove_monitor_button,
            "Remove the selected monitor while the server is stopped.",
        )
        edit_monitor_button = ttk.Button(
            monitor_controls,
            text="Edit selected interlock",
            command=self._edit_selected_monitor,
        )
        edit_monitor_button.pack(side="left", padx=(8, 0))
        _ToolTip(
            edit_monitor_button,
            "Edit a single check's variable/operator/threshold, a multi-check's "
            "member checks, or a configuration-change check's watched variables.",
        )
        add_external_button = ttk.Button(
            monitor_controls,
            text="Add external signal",
            command=self._open_add_external_signal_dialog,
        )
        add_external_button.pack(side="left", padx=(8, 0))
        _ToolTip(
            add_external_button,
            "Listen for a one-shot UDP safe/warning packet from a non-GEECS "
            "source (e.g. a motion controller on another computer) and add it "
            "as an interlock.",
        )

    def _build_devices_tab(self, body: ttk.Frame) -> None:
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        device_frame = ttk.LabelFrame(body, text="Available devices", padding=8)
        device_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        self.device_list = tk.Listbox(device_frame, exportselection=False, width=32)
        self.device_list.grid(row=0, column=0, sticky="nsew")
        self.device_list.bind("<<ListboxSelect>>", self._on_device_selected)
        add_device_button = ttk.Button(
            device_frame, text="Add device", command=self._open_add_device_dialog
        )
        add_device_button.grid(row=1, column=0, pady=(8, 0), sticky="ew")
        _ToolTip(
            add_device_button,
            "Register a device and create its first threshold or alignment check.",
        )
        remove_device_button = ttk.Button(
            device_frame,
            text="Remove selected device",
            command=self._remove_selected_device,
        )
        remove_device_button.grid(row=2, column=0, pady=(6, 0), sticky="ew")
        _ToolTip(
            remove_device_button,
            "Remove the selected device and its related monitors while stopped.",
        )

        details = ttk.Frame(body)
        details.grid(row=0, column=1, sticky="nsew")
        details.columnconfigure(0, weight=1)
        details.rowconfigure(1, weight=1)
        self.selected_device_var = tk.StringVar(value="Select a device")
        self._selected_device_alias: str | None = None
        ttk.Label(
            details,
            textvariable=self.selected_device_var,
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        variable_frame = ttk.LabelFrame(
            details, text="Interlockable variables", padding=8
        )
        variable_frame.grid(row=1, column=0, sticky="nsew")
        variable_frame.columnconfigure(0, weight=1)
        variable_frame.rowconfigure(0, weight=1)
        columns = ("variable", "current", "target", "interlock")
        self.variable_table = ttk.Treeview(
            variable_frame, columns=columns, show="headings", selectmode="browse"
        )
        for column, heading, width in (
            ("variable", "Variable", 180),
            ("current", "Current", 160),
            ("target", "Target / threshold", 220),
            ("interlock", "Interlock", 220),
        ):
            self.variable_table.heading(column, text=heading)
            self.variable_table.column(column, width=width, anchor="w")
        self.variable_table.grid(row=0, column=0, sticky="nsew")
        variable_scrollbar = ttk.Scrollbar(
            variable_frame, orient="vertical", command=self.variable_table.yview
        )
        variable_scrollbar.grid(row=0, column=1, sticky="ns")
        self.variable_table.configure(yscrollcommand=variable_scrollbar.set)
        self.variable_table.bind("<Double-1>", self._edit_device_condition)
        self._device_condition_rows: dict[str, tuple[Any, str]] = {}
        variable_entry_frame = ttk.Frame(variable_frame)
        variable_entry_frame.grid(
            row=1, column=0, columnspan=2, pady=(8, 0), sticky="ew"
        )
        variable_entry_frame.columnconfigure(0, weight=1)
        self.additional_variable_var = tk.StringVar()
        self.additional_variable_box = ttk.Combobox(
            variable_entry_frame,
            textvariable=self.additional_variable_var,
            state="normal",
        )
        self.additional_variable_box.grid(row=0, column=0, sticky="ew")
        subscribe_button = ttk.Button(
            variable_entry_frame,
            text="Subscribe variable",
            command=self._add_variable_to_selected_device,
        )
        subscribe_button.grid(row=0, column=1, padx=(8, 0))
        _ToolTip(
            subscribe_button, "Add this variable to the selected device subscription."
        )

        settings = ttk.LabelFrame(details, text="Condition settings", padding=8)
        settings.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="Condition").grid(row=0, column=0, sticky="w")
        ttk.Label(settings, text="Value").grid(row=0, column=1, sticky="w")
        ttk.Label(
            settings,
            text="Edit a threshold or tolerance, then apply and re-arm if needed.",
        ).grid(row=0, column=2, padx=(12, 0), sticky="w")
        self.condition_settings = settings

        controls = ttk.Frame(self.root, padding=(16, 0, 16, 16))
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(4, weight=1)
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        _ToolTip(self.start_button, "Subscribe devices and start the interlock server.")
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=1, padx=(0, 8))
        _ToolTip(self.stop_button, "Stop the server and unsubscribe devices.")
        self.arm_button = ttk.Button(controls, text="Arm settings", command=self.arm)
        self.arm_button.grid(row=0, column=2, padx=(0, 8))
        _ToolTip(
            self.arm_button,
            "Capture current configuration values as the safe baseline.",
        )
        self.apply_button = ttk.Button(
            controls, text="Apply device settings", command=self.apply_device_settings
        )
        self.apply_button.grid(row=0, column=3, padx=(0, 8))
        _ToolTip(
            self.apply_button, "Apply edited thresholds or tolerances to the checks."
        )
        ttk.Button(
            controls, text="Save configuration", command=self.save_configuration
        ).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(
            controls, text="Import configuration", command=self.import_configuration
        ).grid(row=0, column=5, padx=(0, 8))
        ttk.Label(
            controls,
            text=f"{self.interlock.host}:{self.interlock.port}",
        ).grid(row=0, column=6, sticky="e")

    def _show_help(self) -> None:
        """Show the interlock GUI quick-start guide."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Interlock GUI Help")
        help_window.geometry("620x420")
        help_window.transient(self.root)
        help_window.grab_set()
        help_window.columnconfigure(0, weight=1)
        help_window.rowconfigure(0, weight=1)
        help_text = tk.Text(help_window, wrap="word", padx=14, pady=14)
        help_text.grid(row=0, column=0, sticky="nsew")
        help_text.insert(
            "1.0",
            "Interlock GUI\n\n"
            "1. Add devices from the Devices tab and choose the variables to subscribe to.\n"
            "2. Create threshold or alignment checks for the device.\n"
            "3. Use Create multi-check to combine checks with OR logic.\n"
            "4. Press Start to subscribe devices and run monitor threads.\n"
            "5. Press Arm settings after startup for configuration-change checks.\n"
            "6. Use Save configuration and Import configuration to reuse setups.\n\n"
            "Safety behavior\n"
            "Missing or stale device data fails safe by making the affected check unsafe. "
            "Stop the server before removing devices, monitors, or importing a configuration.",
        )
        help_text.configure(state="disabled")
        ttk.Button(help_window, text="Close", command=help_window.destroy).grid(
            row=1, column=0, padx=14, pady=10, sticky="e"
        )

    def _set_controls(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.arm_button.configure(
            state="normal" if running and self.armable_checks else "disabled"
        )
        self.apply_button.configure(
            state="normal" if self._condition_rows else "disabled"
        )

    def _run_lifecycle(self, action: Any, success_message: str) -> None:
        def worker() -> None:
            try:
                action()
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Interlock lifecycle action failed")
                error_message = str(error)
                self.root.after(
                    0, lambda: messagebox.showerror("Interlock error", error_message)
                )
            else:
                self.root.after(0, lambda: self.status_var.set(success_message))

        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def start(self) -> None:
        """Start device subscriptions and monitor/server threads."""
        self.status_var.set("Starting...")
        self._set_controls(running=True)
        self._run_lifecycle(self.interlock.start, "Running")

    def stop(self) -> None:
        """Stop monitor/server threads and unsubscribe devices."""
        self.status_var.set("Stopping...")
        self._run_lifecycle(self.interlock.stop, "Stopped")
        self._set_controls(running=False)

    def arm(self) -> None:
        """Capture current configuration values for all armable checks."""
        try:
            for check in self.armable_checks:
                check.arm_interlock()
        except (RuntimeError, ValueError) as error:
            messagebox.showerror("Cannot arm interlocks", str(error))
            return
        self.status_var.set("Running - settings armed")

    def save_configuration(self) -> None:
        """Save the current devices and monitor checks to a JSON file."""
        path = filedialog.asksaveasfilename(
            title="Save interlock configuration",
            initialfile=f"{self.interlock.exp_name} interlocks.json",
            defaultextension=".json",
            filetypes=[("JSON configuration", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_configuration(self.interlock, path)
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("Cannot save configuration", str(error))
            return
        self.status_var.set(f"Configuration saved: {path}")

    def import_configuration(self) -> None:
        """Import devices and monitors from a JSON configuration file."""
        if self.interlock.server is not None:
            messagebox.showerror(
                "Cannot import configuration",
                "Stop the interlock before importing a configuration.",
            )
            return
        path = filedialog.askopenfilename(
            title="Import interlock configuration",
            filetypes=[("JSON configuration", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            load_configuration(self.interlock, path)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            messagebox.showerror("Cannot import configuration", str(error))
            return
        self.armable_checks = tuple(
            condition
            for monitor, _, _ in self.interlock.monitor_registrations.values()
            for condition in self._iter_conditions(monitor)
            if isinstance(condition, ConfigurationChangeCheck)
        )
        self._selected_device_alias = None
        self.status_var.set(f"Configuration imported: {path}")
        self._refresh()

    def _iter_conditions(self, condition: Any) -> Iterable[Any]:
        if isinstance(condition, MultiCheck):
            for child in condition.conditions:
                yield from self._iter_conditions(child)
        else:
            yield condition

    def _configured_conditions(self) -> list[Any]:
        conditions = []
        for check_func, _, _ in self.interlock.monitor_registrations.values():
            conditions.extend(self._iter_conditions(check_func))
        return conditions

    def _conditions_for_device(self, alias: str) -> list[tuple[str, Any]]:
        matches = []
        for monitor_name, (check, _, _) in self.interlock.monitor_registrations.items():
            for condition in self._iter_conditions(check):
                if getattr(condition, "device_name", None) == alias:
                    matches.append((monitor_name, condition))
        return matches

    def _first_alias_for_device_name(self, device_name: str) -> str | None:
        """Return the first registered alias for a physical GEECS device name.

        A device name can be registered under several aliases (one per interlock
        role), so this is only a best-effort lookup for prefilling the "add
        device" form - not a reliable reverse mapping.
        """
        for alias, name in self.interlock.device_group.device_names.items():
            if name == device_name:
                return alias
        return None

    @staticmethod
    def _condition_variable_text(condition: Any) -> str:
        if isinstance(condition, ThresholdCheck):
            return condition.variable_name
        if isinstance(condition, AlignmentCheck):
            values = condition.value_variable
            return ", ".join(values) if isinstance(values, list) else values
        if isinstance(condition, ConfigurationChangeCheck):
            return ", ".join(condition.variables)
        return ""

    def _condition_target_text(self, condition: Any) -> str:
        if isinstance(condition, ThresholdCheck):
            return f"{condition.operator} {condition.threshold}"
        if isinstance(condition, AlignmentCheck):
            tolerances = condition.tolerance
            if not isinstance(tolerances, list):
                tolerances = [tolerances]
            return "tolerance: " + ", ".join(str(value) for value in tolerances)
        if isinstance(condition, ConfigurationChangeCheck):
            return "armed" if condition.baseline else "not armed"
        return ""

    def _edit_device_condition(self, event: Any) -> None:
        row_id = self.variable_table.identify_row(event.y)
        column = self.variable_table.identify_column(event.x)
        if not row_id or column != "#3" or row_id not in self._device_condition_rows:
            return
        condition, _ = self._device_condition_rows[row_id]
        if isinstance(condition, ConfigurationChangeCheck):
            self._open_configuration_change_edit_dialog(
                self._condition_description(condition), condition
            )
            return
        if not isinstance(condition, (ThresholdCheck, AlignmentCheck)):
            return
        entry = ttk.Entry(self.variable_table)
        entry.insert(0, self._condition_edit_value(condition))
        bbox = self.variable_table.bbox(row_id, column)
        if not bbox:
            return
        entry.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        entry.focus_set()

        def commit(_event: Any = None) -> None:
            try:
                self._apply_condition_edit(condition, entry.get())
            except ValueError as error:
                messagebox.showerror("Invalid target", str(error))
            else:
                entry.destroy()
                self._refresh()
                self._on_device_selected()

        entry.bind("<Return>", commit)
        entry.bind("<Escape>", lambda _event: entry.destroy())
        entry.bind("<FocusOut>", commit)

    @staticmethod
    def _condition_edit_value(condition: Any) -> str:
        if isinstance(condition, ThresholdCheck):
            return str(condition.threshold)
        tolerances = condition.tolerance
        return (
            ", ".join(str(value) for value in tolerances)
            if isinstance(tolerances, list)
            else str(tolerances)
        )

    @staticmethod
    def _apply_condition_edit(condition: Any, text: str) -> None:
        if isinstance(condition, ThresholdCheck):
            condition.threshold = float(text)
            return
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
        if not values:
            raise ValueError("Enter one or more numeric tolerance values")
        condition.tolerance = (
            values if isinstance(condition.tolerance, list) else values[0]
        )

    def _rebuild_condition_settings(self, device_name: str) -> None:
        for child in self.condition_settings.winfo_children():
            if int(child.grid_info().get("row", 0)) > 0:
                child.destroy()
        self._condition_rows.clear()
        row = 1
        for condition in self._configured_conditions():
            if getattr(condition, "device_name", None) != device_name:
                continue
            if isinstance(condition, ConfigurationChangeCheck):
                ttk.Label(
                    self.condition_settings,
                    text=f"configuration change: {', '.join(condition.variables)}",
                ).grid(row=row, column=0, sticky="w", pady=2)
                ttk.Button(
                    self.condition_settings,
                    text="Edit watched variables...",
                    command=lambda cond=condition: self._open_configuration_change_edit_dialog(
                        self._condition_description(cond), cond
                    ),
                ).grid(row=row, column=1, sticky="w", pady=2)
                ttk.Label(
                    self.condition_settings,
                    text="armed" if condition.baseline else "not armed",
                ).grid(row=row, column=2, padx=(12, 0), sticky="w")
                row += 1
                continue
            if isinstance(condition, ThresholdCheck):
                label = f"{condition.variable_name} {condition.operator}"
                value = condition.threshold
            elif isinstance(condition, AlignmentCheck):
                label = f"{condition.value_variable} tolerance"
                value = condition.tolerance
            else:
                continue
            ttk.Label(self.condition_settings, text=label).grid(
                row=row, column=0, sticky="w", pady=2
            )
            entry = ttk.Entry(self.condition_settings, width=16)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, sticky="w", pady=2)
            ttk.Label(
                self.condition_settings,
                text="threshold"
                if isinstance(condition, ThresholdCheck)
                else "tolerance",
            ).grid(row=row, column=2, padx=(12, 0), sticky="w")
            self._condition_rows[f"{id(condition)}"] = (condition, label, entry)
            row += 1
        self._set_controls(self.interlock.server is not None)

    def _on_device_selected(self, _event: Any = None) -> None:
        selection = self.device_list.curselection()
        if not selection:
            return
        display_name = self.device_list.get(selection[0])
        device_name = self._device_display_to_alias.get(display_name, display_name)
        self._selected_device_alias = device_name
        self.selected_device_var.set(display_name)
        variables = self._device_variables.get(device_name, [])
        device = self.interlock.device_group.get_device(device_name)
        available_variables = set(variables)
        if device is not None:
            available_variables.update(getattr(device, "dev_vars", {}).keys())
        self.additional_variable_box.configure(values=sorted(available_variables))
        self.additional_variable_var.set("")
        for item in self.variable_table.get_children():
            self.variable_table.delete(item)
        self._device_condition_rows.clear()
        configured_variables = set()
        for index, (monitor_name, condition) in enumerate(
            self._conditions_for_device(device_name)
        ):
            if isinstance(condition, AlignmentCheck):
                value_variables = condition.value_variable
            elif isinstance(condition, ThresholdCheck):
                value_variables = [condition.variable_name]
            elif isinstance(condition, ConfigurationChangeCheck):
                value_variables = list(condition.variables)
            else:
                # Unknown/custom condition types (e.g. CustomCheck) don't map to a
                # fixed variable name - show the row without current-value data
                # instead of guessing at an attribute that may not exist.
                value_variables = []
            if isinstance(value_variables, str):
                value_variables = [value_variables]
            current_values = [
                self.interlock.device_group.get_value(device_name, variable)
                for variable in value_variables
            ]
            row_id = f"condition-{index}"
            self._device_condition_rows[row_id] = (condition, monitor_name)
            configured_variables.update(value_variables)
            self.variable_table.insert(
                "",
                "end",
                iid=row_id,
                values=(
                    self._condition_variable_text(condition),
                    self._display(current_values[0])
                    if len(current_values) == 1
                    else ", ".join(self._display(value) for value in current_values),
                    self._condition_target_text(condition),
                    monitor_name,
                ),
            )
        for variable in variables:
            if variable in configured_variables:
                continue
            current = self.interlock.device_group.get_value(device_name, variable)
            self.variable_table.insert(
                "",
                "end",
                iid=f"variable-{variable}",
                values=(variable, self._display(current), "", "Not interlocked"),
            )
        self._rebuild_condition_settings(device_name)

    def _add_variable_to_selected_device(self) -> None:
        """Subscribe an additional variable on the selected device."""
        alias = self._selected_device_alias
        variable = self.additional_variable_var.get().strip()
        if alias is None:
            messagebox.showinfo("Subscribe variable", "Select a device first.")
            return
        if not variable:
            messagebox.showerror("Subscribe variable", "Enter or select a variable.")
            return
        try:
            if not self.interlock.device_group.add_variables(alias, [variable]):
                raise RuntimeError(f"Subscription failed for variable '{variable}'")
        except (KeyError, RuntimeError) as error:
            messagebox.showerror("Cannot subscribe variable", str(error))
            return
        self.additional_variable_var.set("")
        self._refresh()
        self._on_device_selected()

    def _open_add_device_dialog(self) -> None:
        """Open the form used to register a device and its first condition."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add interlock device")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        full_name = tk.StringVar()
        alias = tk.StringVar()
        variable = tk.StringVar()
        selected_variable = tk.StringVar()
        selected_alignment_target = tk.StringVar()
        interlock_type = tk.StringVar(value="Threshold")
        operator = tk.StringVar(value="<")
        target = tk.StringVar()
        variables: list[str] = []
        catalog_devices = self.interlock.device_group.experiment_info.get("devices", {})

        ttk.Label(dialog, text="GEECS device name").grid(
            row=0, column=0, padx=10, pady=6, sticky="w"
        )
        device_name_box = ttk.Combobox(
            dialog,
            textvariable=full_name,
            values=tuple(sorted(catalog_devices)),
            width=40,
        )
        device_name_box.grid(row=0, column=1, padx=10, pady=6, sticky="ew")
        ttk.Label(dialog, text="Interlock alias").grid(
            row=1, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=alias, width=42).grid(
            row=1, column=1, padx=10, pady=6, sticky="ew"
        )

        variable_frame = ttk.LabelFrame(
            dialog, text="Variables to subscribe", padding=6
        )
        variable_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
        variable_frame.columnconfigure(0, weight=1)
        variable_entry = ttk.Entry(variable_frame, textvariable=variable)
        variable_entry.grid(row=0, column=0, sticky="ew")
        variable_list = tk.Listbox(variable_frame, height=4, exportselection=False)
        variable_list.grid(row=1, column=0, columnspan=2, pady=(6, 0), sticky="ew")

        def add_variable() -> None:
            name = variable.get().strip()
            if name and name not in variables:
                variables.append(name)
                variable_list.insert(tk.END, name)
                variable.set("")
                refresh_variable_choices()

        def remove_variable() -> None:
            selected = variable_list.curselection()
            if selected:
                variables.pop(selected[0])
                variable_list.delete(selected[0])
                refresh_variable_choices()

        ttk.Button(variable_frame, text="Add variable", command=add_variable).grid(
            row=0, column=1, padx=(6, 0)
        )
        ttk.Button(
            variable_frame, text="Remove selected", command=remove_variable
        ).grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="w")

        def load_existing_device(_event: Any = None) -> None:
            selected_name = full_name.get().strip()
            existing_alias = self._first_alias_for_device_name(selected_name)
            if existing_alias is not None:
                alias.set(existing_alias)
            variables.clear()
            variable_list.delete(0, tk.END)
            if existing_alias is not None:
                selected_variables = self._device_variables.get(existing_alias, [])
            else:
                selected_variables = list(catalog_devices.get(selected_name, {}))
            for existing_variable in selected_variables:
                variables.append(existing_variable)
                variable_list.insert(tk.END, existing_variable)
            refresh_variable_choices()

        device_name_box.bind("<<ComboboxSelected>>", load_existing_device)

        condition_frame = ttk.LabelFrame(dialog, text="Interlock condition", padding=6)
        condition_frame.grid(
            row=3, column=0, columnspan=2, padx=10, pady=6, sticky="ew"
        )
        condition_frame.columnconfigure(1, weight=1)
        ttk.Label(condition_frame, text="Type").grid(row=0, column=0, sticky="w")
        type_box = ttk.Combobox(
            condition_frame,
            textvariable=interlock_type,
            values=("Threshold", "Alignment"),
            state="readonly",
        )
        type_box.grid(row=0, column=1, sticky="ew")
        ttk.Label(condition_frame, text="Variable").grid(row=1, column=0, sticky="w")
        variable_box = ttk.Combobox(
            condition_frame, textvariable=selected_variable, state="normal"
        )
        variable_box.grid(row=1, column=1, sticky="ew")
        ttk.Label(condition_frame, text="Operator").grid(row=2, column=0, sticky="w")
        operator_box = ttk.Combobox(
            condition_frame,
            textvariable=operator,
            values=("<", ">", "<=", ">=", "==", "!="),
            state="readonly",
        )
        operator_box.grid(row=2, column=1, sticky="ew")
        ttk.Label(condition_frame, text="Target / threshold").grid(
            row=3, column=0, sticky="w"
        )
        ttk.Entry(condition_frame, textvariable=target).grid(
            row=3, column=1, sticky="ew"
        )
        ttk.Label(condition_frame, text="Alignment target variable").grid(
            row=4, column=0, sticky="w"
        )
        alignment_target_box = ttk.Combobox(
            condition_frame,
            textvariable=selected_alignment_target,
            state="normal",
        )
        alignment_target_box.grid(row=4, column=1, sticky="ew")

        def refresh_variable_choices() -> None:
            variable_box.configure(values=variables)
            alignment_target_box.configure(values=variables)
            if variables and not selected_variable.get():
                selected_variable.set(variables[0])
            if variables and not selected_alignment_target.get():
                selected_alignment_target.set(variables[0])

        def refresh_type_fields(_event: Any = None) -> None:
            is_alignment = interlock_type.get() == "Alignment"
            operator_box.configure(state="disabled" if is_alignment else "readonly")
            alignment_target_box.configure(
                state="readonly" if is_alignment else "disabled"
            )
            if is_alignment:
                selected_alignment_target.set(variables[0] if variables else "")

        type_box.bind("<<ComboboxSelected>>", refresh_type_fields)

        def save_device() -> None:
            name = full_name.get().strip()
            selected_variable_name = selected_variable.get().strip()
            selected_target_variable = selected_alignment_target.get().strip()
            selected_alias = alias.get().strip() or name
            try:
                if self.interlock.server is not None:
                    raise RuntimeError("Stop the interlock before adding a device.")
                if not name or not variables or not selected_variable_name:
                    raise ValueError(
                        "Device name, variables, and a monitored variable are required."
                    )
                if selected_alias in self.interlock.device_group.devices:
                    raise ValueError(f"Alias '{selected_alias}' is already registered.")
                device = self.interlock.add_device(selected_alias, name, variables)
                device.use_alias_in_TCP_subscription = False
                if interlock_type.get() == "Alignment":
                    value_variables = [
                        value.strip()
                        for value in selected_variable_name.split(",")
                        if value.strip()
                    ]
                    target_variables = [
                        value.strip()
                        for value in selected_target_variable.split(",")
                        if value.strip()
                    ]
                    tolerances = [
                        float(value.strip())
                        for value in target.get().split(",")
                        if value.strip()
                    ]
                    if (
                        not value_variables
                        or len(value_variables) != len(target_variables)
                        or len(value_variables) != len(tolerances)
                    ):
                        raise ValueError(
                            "Alignment requires matching value variables, target variables, and tolerances."
                        )
                    if any(
                        value == target_name
                        for value, target_name in zip(value_variables, target_variables)
                    ):
                        raise ValueError(
                            "Each alignment value variable needs a different target variable."
                        )
                    check = AlignmentCheck(
                        self.interlock.device_group,
                        selected_alias,
                        value_variables,
                        target_variables,
                        tolerances,
                    )
                else:
                    numeric_target = float(target.get())
                    check = ThresholdCheck(
                        self.interlock.device_group,
                        selected_alias,
                        selected_variable_name,
                        numeric_target,
                        operator=operator.get(),
                    )
                self.interlock.add_monitor(
                    f"{name} {selected_variable_name} interlock", check
                )
            except (RuntimeError, ValueError) as error:
                messagebox.showerror("Cannot add device", str(error), parent=dialog)
                return
            dialog.destroy()
            self._refresh()

        ttk.Button(dialog, text="Add device", command=save_device).grid(
            row=5, column=0, padx=10, pady=10, sticky="w"
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=5, column=1, padx=10, pady=10, sticky="e"
        )
        refresh_type_fields()

    def _remove_selected_monitor(self) -> None:
        """Remove the selected monitor registration after confirmation."""
        selection = self.monitor_table.selection()
        if not selection:
            messagebox.showinfo("Remove interlock", "Select an interlock first.")
            return
        name = selection[0]
        if name in self._leaf_condition_rows:
            messagebox.showinfo(
                "Remove interlock",
                "This is a member of a multi-check. Use 'Edit selected interlock' "
                "on the multi-check to uncheck it instead.",
            )
            return
        if not messagebox.askyesno("Remove interlock", f"Remove '{name}'?"):
            return
        registration = self.interlock.monitor_registrations.get(name)
        removed_check = registration[0] if registration else None
        try:
            self.interlock.remove_monitor(name)
            self.armable_checks = tuple(
                check
                for check in self.armable_checks
                if check in self._configured_conditions()
            )
            if removed_check is not None:
                self._prune_orphaned_external_listeners(removed_check)
        except (KeyError, RuntimeError) as error:
            messagebox.showerror("Cannot remove interlock", str(error))

    def _prune_orphaned_external_listeners(self, removed_check: Any) -> None:
        """Drop external listeners no longer backing any remaining condition."""
        removed_listeners = {
            condition.listener
            for condition in self._iter_conditions(removed_check)
            if isinstance(condition, ExternalStatusCheck)
        }
        if not removed_listeners:
            return
        still_used = {
            condition.listener
            for condition in self._configured_conditions()
            if isinstance(condition, ExternalStatusCheck)
        }
        orphaned = removed_listeners - still_used
        if orphaned:
            self.interlock.external_listeners = [
                listener
                for listener in self.interlock.external_listeners
                if listener not in orphaned
            ]

    def _remove_selected_device(self) -> None:
        """Remove the selected device and its related monitor registrations."""
        if self.interlock.server is not None:
            messagebox.showerror(
                "Cannot remove device", "Stop the interlock before removing a device."
            )
            return
        if self._selected_device_alias is None:
            messagebox.showinfo("Remove device", "Select a device first.")
            return
        alias = self._selected_device_alias
        display_name = self._device_alias_to_display.get(alias, alias)
        if not messagebox.askyesno(
            "Remove device", f"Remove '{display_name}' and its interlocks?"
        ):
            return
        try:
            for name, (check, _, _) in list(
                self.interlock.monitor_registrations.items()
            ):
                if any(
                    getattr(condition, "device_name", None) == alias
                    for condition in self._iter_conditions(check)
                ):
                    self.interlock.remove_monitor(name)
            self.interlock.device_group.remove_device(alias)
            self.armable_checks = tuple(
                check
                for check in self.armable_checks
                if getattr(check, "device_name", None) != alias
            )
            self._selected_device_alias = None
            self._refresh()
        except (KeyError, RuntimeError) as error:
            messagebox.showerror("Cannot remove device", str(error))

    def _on_monitor_table_double_click(self, event: Any) -> None:
        """Open the edit dialog for whichever interlock (or member) row was double-clicked."""
        if self.interlock.server is not None:
            return
        row_id = self.monitor_table.identify_row(event.y)
        if not row_id:
            return
        self.monitor_table.selection_set(row_id)
        self._edit_selected_monitor()

    def _edit_selected_monitor(self) -> None:
        """Edit the condition behind the selected interlock or interlock member."""
        if self.interlock.server is not None:
            messagebox.showerror(
                "Cannot edit interlock",
                "Stop the interlock before editing an interlock.",
            )
            return
        selection = self.monitor_table.selection()
        if not selection:
            messagebox.showinfo("Edit interlock", "Select an interlock first.")
            return
        row_id = selection[0]
        if row_id in self._leaf_condition_rows:
            condition = self._leaf_condition_rows[row_id]
            title = self._condition_description(condition)
        else:
            registration = self.interlock.monitor_registrations.get(row_id)
            if registration is None:
                return
            condition = registration[0]
            title = row_id
        if not self._open_condition_edit_dialog(title, condition):
            messagebox.showinfo(
                "Edit interlock", f"'{title}' cannot be edited from this dialog."
            )

    def _open_condition_edit_dialog(
        self,
        title: str,
        condition: Any,
        on_apply: Callable[[], None] | None = None,
    ) -> bool:
        """Open whichever edit dialog matches ``condition``'s type.

        Returns
        -------
            False if the condition type has no editor, so the caller can report that.
        """
        if isinstance(condition, MultiCheck):
            self._open_multi_check_edit_dialog(title, condition, on_apply=on_apply)
        elif isinstance(condition, ThresholdCheck):
            self._open_threshold_edit_dialog(title, condition, on_apply=on_apply)
        elif isinstance(condition, AlignmentCheck):
            self._open_alignment_edit_dialog(title, condition, on_apply=on_apply)
        elif isinstance(condition, ConfigurationChangeCheck):
            self._open_configuration_change_edit_dialog(
                title, condition, on_apply=on_apply
            )
        elif isinstance(condition, ExternalStatusCheck):
            self._open_external_status_edit_dialog(title, condition, on_apply=on_apply)
        else:
            return False
        return True

    def _after_condition_edit(self) -> None:
        """Refresh views and prompt for re-arming after an interlock is edited."""
        self.status_var.set("Settings changed - re-arm before relying on interlock")
        self._refresh()
        self._on_device_selected()

    def _open_multi_check_edit_dialog(
        self,
        monitor_name: str,
        multi_check: MultiCheck,
        on_apply: Callable[[], None] | None = None,
    ) -> None:
        """Edit which leaf conditions belong to a multi-check, or edit one in place."""
        conditions = self._configured_conditions()
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {monitor_name}")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(
            dialog,
            text="Select conditions to include (OR logic). Use Edit to change a "
            "condition's own settings.",
        ).pack(padx=10, pady=(10, 5), anchor="w")
        checks = []
        selected_ids = {id(condition) for condition in multi_check.conditions}
        for index, condition in enumerate(conditions):
            checked = tk.BooleanVar(value=id(condition) in selected_ids)
            checks.append((condition, checked))
            row = ttk.Frame(dialog)
            row.pack(padx=10, pady=1, fill="x")
            checkbutton = ttk.Checkbutton(
                row,
                text=self._condition_description(condition),
                variable=checked,
            )
            checkbutton.pack(side="left", anchor="w")

            def make_edit_handler(
                cond: Any = condition, widget: ttk.Checkbutton = checkbutton
            ) -> Callable[[], None]:
                def refresh_label() -> None:
                    widget.configure(text=self._condition_description(cond))

                def handler() -> None:
                    if not self._open_condition_edit_dialog(
                        self._condition_description(cond),
                        cond,
                        on_apply=refresh_label,
                    ):
                        messagebox.showinfo(
                            "Edit interlock",
                            "This condition type cannot be edited here.",
                            parent=dialog,
                        )

                return handler

            if isinstance(
                condition,
                (
                    ThresholdCheck,
                    AlignmentCheck,
                    ConfigurationChangeCheck,
                    ExternalStatusCheck,
                    MultiCheck,
                ),
            ):
                ttk.Button(row, text="Edit", command=make_edit_handler()).pack(
                    side="right"
                )

        def apply() -> None:
            selected_conditions = [
                condition for condition, checked in checks if checked.get()
            ]
            if len(selected_conditions) < 2:
                messagebox.showerror(
                    "Cannot edit multi-check",
                    "Select at least two conditions.",
                    parent=dialog,
                )
                return
            multi_check.conditions = selected_conditions
            dialog.destroy()
            self._after_condition_edit()
            if on_apply:
                on_apply()

        ttk.Button(dialog, text="Apply", command=apply).pack(
            side="left", padx=10, pady=10
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(
            side="right", padx=10, pady=10
        )

    def _edit_dialog_variable_choices(self, device_alias: str) -> list[str]:
        variables = set(self._device_variables.get(device_alias, []))
        device = self.interlock.device_group.get_device(device_alias)
        if device is not None:
            variables.update(getattr(device, "dev_vars", {}).keys())
        return sorted(variables)

    def _open_threshold_edit_dialog(
        self,
        monitor_name: str,
        condition: ThresholdCheck,
        on_apply: Callable[[], None] | None = None,
    ) -> None:
        """Edit a single threshold check's variable, operator, and threshold."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {monitor_name}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        device_display = self._device_alias_to_display.get(
            condition.device_name, condition.device_name
        )
        ttk.Label(dialog, text="Device").grid(
            row=0, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Label(dialog, text=device_display).grid(
            row=0, column=1, padx=10, pady=6, sticky="w"
        )

        variable = tk.StringVar(value=condition.variable_name)
        ttk.Label(dialog, text="Variable").grid(
            row=1, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Combobox(
            dialog,
            textvariable=variable,
            values=self._edit_dialog_variable_choices(condition.device_name),
        ).grid(row=1, column=1, padx=10, pady=6, sticky="ew")

        operator = tk.StringVar(value=condition.operator)
        ttk.Label(dialog, text="Operator").grid(
            row=2, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Combobox(
            dialog,
            textvariable=operator,
            values=tuple(condition.operators.keys()),
            state="readonly",
        ).grid(row=2, column=1, padx=10, pady=6, sticky="ew")

        threshold = tk.StringVar(value=str(condition.threshold))
        ttk.Label(dialog, text="Threshold").grid(
            row=3, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=threshold).grid(
            row=3, column=1, padx=10, pady=6, sticky="ew"
        )

        def apply() -> None:
            variable_name = variable.get().strip()
            if not variable_name:
                messagebox.showerror(
                    "Cannot edit interlock", "Enter a variable.", parent=dialog
                )
                return
            try:
                threshold_value = float(threshold.get())
            except ValueError:
                messagebox.showerror(
                    "Cannot edit interlock",
                    "Threshold must be numeric.",
                    parent=dialog,
                )
                return
            condition.variable_name = variable_name
            condition.operator = operator.get()
            condition.operator_fn = condition.operators[operator.get()]
            condition.threshold = threshold_value
            dialog.destroy()
            self._after_condition_edit()
            if on_apply:
                on_apply()

        ttk.Button(dialog, text="Apply", command=apply).grid(
            row=4, column=0, padx=10, pady=10, sticky="w"
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=4, column=1, padx=10, pady=10, sticky="e"
        )

    def _open_alignment_edit_dialog(
        self,
        monitor_name: str,
        condition: AlignmentCheck,
        on_apply: Callable[[], None] | None = None,
    ) -> None:
        """Edit a single alignment check's value/target variables and tolerance(s)."""

        def as_list(value: Any) -> list[Any]:
            return list(value) if isinstance(value, list) else [value]

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {monitor_name}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        device_display = self._device_alias_to_display.get(
            condition.device_name, condition.device_name
        )
        ttk.Label(dialog, text="Device").grid(
            row=0, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Label(dialog, text=device_display).grid(
            row=0, column=1, padx=10, pady=6, sticky="w"
        )

        value_variables = tk.StringVar(
            value=", ".join(str(v) for v in as_list(condition.value_variable))
        )
        ttk.Label(dialog, text="Value variable(s)").grid(
            row=1, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=value_variables).grid(
            row=1, column=1, padx=10, pady=6, sticky="ew"
        )

        target_variables = tk.StringVar(
            value=", ".join(str(v) for v in as_list(condition.target_variable))
        )
        ttk.Label(dialog, text="Target variable(s)").grid(
            row=2, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=target_variables).grid(
            row=2, column=1, padx=10, pady=6, sticky="ew"
        )

        tolerances = tk.StringVar(
            value=", ".join(str(v) for v in as_list(condition.tolerance))
        )
        ttk.Label(dialog, text="Tolerance(s)").grid(
            row=3, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=tolerances).grid(
            row=3, column=1, padx=10, pady=6, sticky="ew"
        )
        ttk.Label(
            dialog,
            text="For multi-axis alignment, use matching comma-separated lists.",
        ).grid(row=4, column=0, columnspan=2, padx=10, sticky="w")

        def apply() -> None:
            values = [v.strip() for v in value_variables.get().split(",") if v.strip()]
            targets = [
                v.strip() for v in target_variables.get().split(",") if v.strip()
            ]
            try:
                tolerance_values = [
                    float(v.strip()) for v in tolerances.get().split(",") if v.strip()
                ]
            except ValueError:
                messagebox.showerror(
                    "Cannot edit interlock",
                    "Tolerances must be numeric.",
                    parent=dialog,
                )
                return
            if (
                not values
                or len(values) != len(targets)
                or len(values) != len(tolerance_values)
            ):
                messagebox.showerror(
                    "Cannot edit interlock",
                    "Value variables, target variables, and tolerances must have "
                    "matching counts.",
                    parent=dialog,
                )
                return
            was_multi = isinstance(condition.value_variable, list) or len(values) > 1
            condition.value_variable = values if was_multi else values[0]
            condition.target_variable = targets if was_multi else targets[0]
            condition.tolerance = tolerance_values if was_multi else tolerance_values[0]
            dialog.destroy()
            self._after_condition_edit()
            if on_apply:
                on_apply()

        ttk.Button(dialog, text="Apply", command=apply).grid(
            row=5, column=0, padx=10, pady=10, sticky="w"
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=5, column=1, padx=10, pady=10, sticky="e"
        )

    def _open_configuration_change_edit_dialog(
        self,
        monitor_name: str,
        condition: ConfigurationChangeCheck,
        on_apply: Callable[[], None] | None = None,
    ) -> None:
        """Edit which variables a configuration-change check watches."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {monitor_name}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        device_display = self._device_alias_to_display.get(
            condition.device_name, condition.device_name
        )
        ttk.Label(dialog, text="Device").grid(
            row=0, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Label(dialog, text=device_display).grid(
            row=0, column=1, padx=10, pady=6, sticky="w"
        )

        variables = tk.StringVar(value=", ".join(condition.variables))
        ttk.Label(dialog, text="Watched variables").grid(
            row=1, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=variables).grid(
            row=1, column=1, padx=10, pady=6, sticky="ew"
        )
        ttk.Label(
            dialog,
            text="Editing this clears the armed baseline; re-arm before relying on it.",
        ).grid(row=2, column=0, columnspan=2, padx=10, sticky="w")

        def apply() -> None:
            variable_names = [
                v.strip() for v in variables.get().split(",") if v.strip()
            ]
            if not variable_names:
                messagebox.showerror(
                    "Cannot edit interlock",
                    "Enter at least one variable.",
                    parent=dialog,
                )
                return
            condition.variables = variable_names
            condition.baseline = {}
            condition.last_failure_info = "Interlock not armed"
            dialog.destroy()
            self._after_condition_edit()
            if on_apply:
                on_apply()

        ttk.Button(dialog, text="Apply", command=apply).grid(
            row=3, column=0, padx=10, pady=10, sticky="w"
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=3, column=1, padx=10, pady=10, sticky="e"
        )

    def _open_external_status_edit_dialog(
        self,
        monitor_name: str,
        condition: ExternalStatusCheck,
        on_apply: Callable[[], None] | None = None,
    ) -> None:
        """Edit a non-GEECS external signal's label, listen host, and UDP port.

        The listener isn't bound while the interlock is stopped (which editing
        already requires), so these fields can be changed directly - they take
        effect the next time the interlock is started.
        """
        listener = condition.listener
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {monitor_name}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        label = tk.StringVar(value=condition.label)
        ttk.Label(dialog, text="Label").grid(
            row=0, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=label).grid(
            row=0, column=1, padx=10, pady=6, sticky="ew"
        )

        host = tk.StringVar(value=listener.host)
        ttk.Label(dialog, text="Listen host").grid(
            row=1, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=host).grid(
            row=1, column=1, padx=10, pady=6, sticky="ew"
        )
        ttk.Label(
            dialog,
            text="Blank/0.0.0.0 = all local interfaces. Must be an address on\n"
            "THIS computer, never the sender's IP.",
            justify="left",
            foreground="#555555",
        ).grid(row=1, column=2, padx=(8, 10), sticky="w")

        port = tk.StringVar(value=str(listener.port))
        ttk.Label(dialog, text="UDP port").grid(
            row=2, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=port).grid(
            row=2, column=1, padx=10, pady=6, sticky="ew"
        )

        ttk.Label(
            dialog,
            text=listener.get_diagnostic_info(),
            foreground="#555555",
        ).grid(row=3, column=0, columnspan=2, padx=10, pady=(6, 0), sticky="w")

        def apply() -> None:
            label_text = label.get().strip()
            host_text = host.get().strip() or "0.0.0.0"
            if not label_text:
                messagebox.showerror(
                    "Cannot edit interlock", "Enter a label.", parent=dialog
                )
                return
            try:
                port_value = int(port.get())
            except ValueError:
                messagebox.showerror(
                    "Cannot edit interlock", "Port must be an integer.", parent=dialog
                )
                return
            condition.label = label_text
            listener.name = label_text
            listener.host = host_text
            listener.port = port_value
            dialog.destroy()
            self._after_condition_edit()
            if on_apply:
                on_apply()

        ttk.Button(dialog, text="Apply", command=apply).grid(
            row=4, column=0, padx=10, pady=10, sticky="w"
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=4, column=1, padx=10, pady=10, sticky="e"
        )

    def _condition_description(self, condition: Any) -> str:
        if isinstance(condition, ThresholdCheck):
            return (
                f"{condition.device_name}: {condition.variable_name} "
                f"{condition.operator} {condition.threshold}"
            )
        if isinstance(condition, AlignmentCheck):
            return (
                f"{condition.device_name}: {condition.value_variable} aligned to "
                f"{condition.target_variable} +/- {condition.tolerance}"
            )
        if isinstance(condition, ConfigurationChangeCheck):
            variables = ", ".join(condition.variables)
            return f"{condition.device_name}: configuration change on {variables}"
        if isinstance(condition, ExternalStatusCheck):
            return f"External signal: {condition.label}"
        if isinstance(condition, MultiCheck):
            return f"Combined check ({len(condition.conditions)} members)"
        return type(condition).__name__

    def _leaf_current_value_text(self, condition: Any) -> str:
        """Return a short "current value" label for a MultiCheck member row."""
        if isinstance(condition, ThresholdCheck):
            value = self.interlock.device_group.get_value(
                condition.device_name, condition.variable_name
            )
            return f"current: {self._display(value)}"
        if isinstance(condition, AlignmentCheck):
            variables = (
                condition.value_variable
                if isinstance(condition.value_variable, list)
                else [condition.value_variable]
            )
            values = [
                self.interlock.device_group.get_value(condition.device_name, variable)
                for variable in variables
            ]
            return "current: " + ", ".join(self._display(value) for value in values)
        if isinstance(condition, ConfigurationChangeCheck):
            return "armed" if condition.baseline else "not armed"
        if isinstance(condition, ExternalStatusCheck):
            state = condition.listener.get_state()
            if state is None:
                return "no data yet"
            return "WARNING" if state else "safe"
        return ""

    def _open_multi_check_dialog(self) -> None:
        """Open a form for combining existing conditions with OR logic."""
        conditions = []
        for check in self.interlock.monitor_registrations.values():
            conditions.extend(self._iter_conditions(check[0]))
        if len(conditions) < 2:
            messagebox.showinfo(
                "Create multi-check",
                "At least two threshold or alignment checks are required.",
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Create multi-check")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)
        ttk.Label(dialog, text="Multi-check name").grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        name = tk.StringVar(value="Combined interlock")
        ttk.Entry(dialog, textvariable=name, width=48).grid(
            row=1, column=0, padx=10, sticky="ew"
        )
        checks_frame = ttk.LabelFrame(dialog, text="Conditions to combine", padding=8)
        checks_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        selected = []
        self._condition_catalog.clear()
        for index, condition in enumerate(conditions):
            condition_id = f"condition-{index}"
            self._condition_catalog[condition_id] = condition
            checked = tk.BooleanVar(value=True)
            selected.append((condition_id, checked))
            ttk.Checkbutton(
                checks_frame,
                text=self._condition_description(condition),
                variable=checked,
            ).pack(anchor="w")

        def create() -> None:
            selected_conditions = [
                self._condition_catalog[condition_id]
                for condition_id, checked in selected
                if checked.get()
            ]
            if len(selected_conditions) < 2:
                messagebox.showerror(
                    "Cannot create multi-check",
                    "Select at least two conditions.",
                    parent=dialog,
                )
                return
            monitor_name = name.get().strip()
            if not monitor_name:
                messagebox.showerror(
                    "Cannot create multi-check",
                    "Enter a monitor name.",
                    parent=dialog,
                )
                return
            try:
                self.interlock.add_monitor(
                    monitor_name, MultiCheck(selected_conditions), interval=0.1
                )
            except ValueError as error:
                messagebox.showerror(
                    "Cannot create multi-check", str(error), parent=dialog
                )
                return
            dialog.destroy()

        ttk.Button(dialog, text="Create", command=create).grid(
            row=3, column=0, padx=10, pady=(0, 10), sticky="w"
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=3, column=0, padx=10, pady=(0, 10), sticky="e"
        )

    def _open_add_external_signal_dialog(self) -> None:
        """Open a form to listen for a non-GEECS UDP safe/warning signal as an interlock."""
        if self.interlock.server is not None:
            messagebox.showerror(
                "Cannot add external signal",
                "Stop the interlock before adding an external signal.",
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Add external signal")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        ttk.Label(
            dialog,
            text="Listens for a one-shot UDP packet (e.g. from a motion controller\n"
            'on another computer) containing "safe" or "warning", and uses it as\n'
            "an interlock condition. No GEECS device is involved.",
            justify="left",
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 6), sticky="w")

        label = tk.StringVar(value="external-signal")
        ttk.Label(dialog, text="Label").grid(
            row=1, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=label).grid(
            row=1, column=1, padx=10, pady=6, sticky="ew"
        )

        host = tk.StringVar(value="")
        ttk.Label(dialog, text="Listen host").grid(
            row=2, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=host).grid(
            row=2, column=1, padx=10, pady=6, sticky="ew"
        )
        ttk.Label(
            dialog,
            text="Leave blank (all local interfaces) unless you specifically need to\n"
            "bind one - this must be an address on THIS computer, never the\n"
            "sender's IP.",
            justify="left",
            foreground="#555555",
        ).grid(row=2, column=2, padx=(8, 10), sticky="w")

        port = tk.StringVar()
        ttk.Label(dialog, text="UDP port").grid(
            row=3, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=port).grid(
            row=3, column=1, padx=10, pady=6, sticky="ew"
        )

        monitor_name = tk.StringVar()
        ttk.Label(dialog, text="Interlock name").grid(
            row=4, column=0, padx=10, pady=6, sticky="w"
        )
        ttk.Entry(dialog, textvariable=monitor_name).grid(
            row=4, column=1, padx=10, pady=6, sticky="ew"
        )
        ttk.Label(
            dialog,
            text="Leave blank to name the interlock after the label.",
        ).grid(row=5, column=0, columnspan=2, padx=10, sticky="w")

        def add() -> None:
            label_text = label.get().strip()
            host_text = host.get().strip() or "0.0.0.0"
            name_text = monitor_name.get().strip() or (
                f"{label_text} interlock" if label_text else ""
            )
            if not label_text:
                messagebox.showerror(
                    "Cannot add external signal", "Enter a label.", parent=dialog
                )
                return
            if not name_text:
                messagebox.showerror(
                    "Cannot add external signal",
                    "Enter an interlock name.",
                    parent=dialog,
                )
                return
            if name_text in self.interlock.monitor_registrations:
                messagebox.showerror(
                    "Cannot add external signal",
                    f"An interlock named '{name_text}' already exists.",
                    parent=dialog,
                )
                return
            try:
                port_value = int(port.get())
            except ValueError:
                messagebox.showerror(
                    "Cannot add external signal",
                    "Port must be an integer.",
                    parent=dialog,
                )
                return
            listener = self.interlock.add_external_listener(
                port=port_value, host=host_text, name=label_text
            )
            check = ExternalStatusCheck(listener, label=label_text)
            self.interlock.add_monitor(name_text, check)
            dialog.destroy()
            self._refresh()

        ttk.Button(dialog, text="Add", command=add).grid(
            row=6, column=0, padx=10, pady=(6, 10), sticky="w"
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=6, column=1, padx=10, pady=(6, 10), sticky="e"
        )

    @staticmethod
    def _display(value: Any) -> str:
        return "N/A" if value is None else str(value)

    def _target_for(self, device_name: str, variable_name: str) -> str:
        targets = []
        for condition in self._configured_conditions():
            if isinstance(condition, ThresholdCheck):
                if (
                    condition.device_name == device_name
                    and condition.variable_name == variable_name
                ):
                    targets.append(f"{condition.operator} {condition.threshold}")
            elif (
                isinstance(condition, AlignmentCheck)
                and condition.device_name == device_name
                and condition.value_variable == variable_name
            ):
                target = self.interlock.device_group.get_value(
                    device_name, condition.target_variable
                )
                targets.append(f"{self._display(target)} +/- {condition.tolerance}")
        return "; ".join(targets)

    def apply_device_settings(self) -> None:
        """Apply numeric threshold and tolerance edits to configured checks."""
        try:
            for condition, _, entry in self._condition_rows.values():
                value = float(entry.get())
                if isinstance(condition, ThresholdCheck):
                    condition.threshold = value
                elif isinstance(condition, AlignmentCheck):
                    condition.tolerance = value
        except ValueError:
            messagebox.showerror(
                "Invalid setting", "Thresholds and tolerances must be numeric."
            )
            return
        self.status_var.set("Settings changed - re-arm before relying on interlock")
        messagebox.showinfo(
            "Settings applied",
            "Condition settings updated. Press Arm settings to capture the new configuration.",
        )

    def _refresh(self) -> None:
        running = self.interlock.server is not None
        self._set_controls(running)
        self._device_variables = self.interlock.device_group.device_variables.copy()
        self._device_display_to_alias = {}
        self._device_alias_to_display = {}
        for alias in self._device_variables:
            device_name = self.interlock.device_group.device_names.get(alias, alias)
            # The same physical GEECS device can be registered under several
            # aliases (e.g. one alias per interlock role on the same camera), so
            # the device name alone is not a unique key here - key by alias and
            # fold the device name in only for a human-readable label.
            display_name = alias if alias == device_name else f"{alias} ({device_name})"
            self._device_display_to_alias[display_name] = alias
            self._device_alias_to_display[alias] = display_name
        device_names = list(self._device_display_to_alias)
        existing_devices = list(self.device_list.get(0, tk.END))
        if existing_devices != device_names:
            self.device_list.delete(0, tk.END)
            for device_name in device_names:
                self.device_list.insert(tk.END, device_name)
            if device_names and not self.device_list.curselection():
                self.device_list.selection_set(0)
                self._on_device_selected()
        selected_device = self._selected_device_alias
        if selected_device in self._device_variables:
            for row_id, (
                condition,
                monitor_name,
            ) in self._device_condition_rows.items():
                if self.variable_table.exists(row_id):
                    if isinstance(condition, AlignmentCheck):
                        value_variables = condition.value_variable
                    elif isinstance(condition, ThresholdCheck):
                        value_variables = [condition.variable_name]
                    elif isinstance(condition, ConfigurationChangeCheck):
                        value_variables = list(condition.variables)
                    else:
                        value_variables = []
                    if isinstance(value_variables, str):
                        value_variables = [value_variables]
                    current_values = [
                        self.interlock.device_group.get_value(selected_device, variable)
                        for variable in value_variables
                    ]
                    self.variable_table.item(
                        row_id,
                        values=(
                            self._condition_variable_text(condition),
                            self._display(current_values[0])
                            if len(current_values) == 1
                            else ", ".join(
                                self._display(value) for value in current_values
                            ),
                            self._condition_target_text(condition),
                            monitor_name,
                        ),
                    )
            for variable in self._device_variables[selected_device]:
                row_id = f"variable-{variable}"
                if self.variable_table.exists(row_id):
                    current = self.interlock.device_group.get_value(
                        selected_device, variable
                    )
                    self.variable_table.item(
                        row_id,
                        values=(
                            variable,
                            self._display(current),
                            "",
                            "Not interlocked",
                        ),
                    )
        states = self.interlock.get_all_monitors()
        if not states:
            lamp_color, status, detail = "#9a7415", "UNKNOWN", "No monitor state yet"
        elif any(states.values()):
            lamp_color, status, detail = (
                "#c9322b",
                "INTERLOCK ACTIVE",
                "Unsafe monitor condition",
            )
        else:
            lamp_color, status, detail = "#20844a", "SAFE", "All monitors clear"
        self.status_lamp.itemconfigure(self.status_lamp_id, fill=lamp_color)
        self.overall_status_var.set(status)
        self.overall_status_detail_var.set(detail)
        current_rows = set(self.monitor_table.get_children())
        for name, (check, _, _) in self.interlock.monitor_registrations.items():
            state = self.interlock.get_monitor_state(name)
            if state is True:
                label, tag = "UNSAFE", "unsafe"
            elif state is False:
                label, tag = "SAFE", "safe"
            else:
                label, tag = "UNKNOWN", "unknown"
            values = (name, label, self.interlock.get_diagnostic_info(name))
            if name in current_rows:
                self.monitor_table.item(name, values=values, tags=(tag,))
                current_rows.remove(name)
            else:
                self.monitor_table.insert(
                    "", "end", iid=name, values=values, tags=(tag,), open=True
                )
            if isinstance(check, MultiCheck):
                self._sync_leaf_rows(name, check.conditions)
            else:
                for child in list(self.monitor_table.get_children(name)):
                    self._remove_tree_row(child)
        for item in current_rows:
            self._remove_tree_row(item)
        self.root.after(self.refresh_ms, self._refresh)

    def _sync_leaf_rows(self, parent_id: str, conditions: list[Any]) -> None:
        """Mirror a MultiCheck's member conditions as child rows under its tree item.

        Makes the interlocks a multi-check combines visible directly in the
        Interlocks tab (they were previously only reachable via the Devices tab),
        and lets each one be selected and edited on its own.
        """
        existing = set(self.monitor_table.get_children(parent_id))
        for index, condition in enumerate(conditions):
            row_id = f"{parent_id}::leaf::{index}"
            values = (
                f"    ↳ {self._condition_description(condition)}",
                "",
                self._leaf_current_value_text(condition),
            )
            if row_id in existing:
                self.monitor_table.item(row_id, values=values, tags=("leaf",))
                existing.discard(row_id)
            else:
                self.monitor_table.insert(
                    parent_id,
                    "end",
                    iid=row_id,
                    values=values,
                    tags=("leaf",),
                    open=True,
                )
            self._leaf_condition_rows[row_id] = condition
            if isinstance(condition, MultiCheck):
                self._sync_leaf_rows(row_id, condition.conditions)
            else:
                for child in list(self.monitor_table.get_children(row_id)):
                    self._remove_tree_row(child)
        for row_id in existing:
            self._remove_tree_row(row_id)

    def _remove_tree_row(self, row_id: str) -> None:
        """Delete a monitor-table row (and any children), cleaning up leaf tracking."""
        for child in list(self.monitor_table.get_children(row_id)):
            self._remove_tree_row(child)
        self.monitor_table.delete(row_id)
        self._leaf_condition_rows.pop(row_id, None)

    def _on_close(self) -> None:
        if self.interlock.server is not None:
            self.stop()
            self.root.after(100, self._finish_close)
        else:
            self._finish_close()

    def _finish_close(self) -> None:
        self.root.destroy()

    def run(self) -> None:
        """Enter the Tkinter event loop."""
        self.root.mainloop()


def launch_interlock_gui(
    interlock: InterlockConstructor,
    armable_checks: Iterable[Any] = (),
    title: str | None = None,
) -> None:
    """Create and run an :class:`InterlockGUI`."""
    InterlockGUI(interlock, armable_checks, title=title).run()


def launch_general_interlock_gui(
    host: str | None = None,
    port: int | None = None,
    staleness_timeout_ms: float = 5000,
    title: str = "GEECS Interlock Builder",
) -> None:
    """Prompt for an experiment, then open a blank interlock builder GUI.

    ``host`` and ``port`` fall back to the ``SERVER_IP`` / ``SERVER_PORT``
    environment variables (e.g. loaded from a ``.env`` file). The port must be
    provided one way or the other; the host falls back to this machine's
    outbound interface address.
    """
    bind_host = host or os.getenv("SERVER_IP") or _local_bind_host()
    if port is None:
        env_port = os.getenv("SERVER_PORT")
        if not env_port:
            raise ValueError(
                "No interlock server port given: pass port= or set SERVER_PORT"
            )
        port = int(env_port)
    prompt = tk.Tk()
    prompt.title("Select GEECS experiment")
    prompt.geometry("460x170")
    prompt.resizable(False, False)
    prompt.attributes("-topmost", True)
    prompt.columnconfigure(0, weight=1)

    selected_experiment = tk.StringVar()
    ttk.Label(
        prompt,
        text="Choose the GEECS experiment for this interlock:",
    ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")
    experiment_box = ttk.Combobox(
        prompt,
        textvariable=selected_experiment,
        values=(),
        state="normal",
        width=48,
    )
    experiment_box.grid(row=1, column=0, padx=16, sticky="ew")
    error_var = tk.StringVar()
    ttk.Label(prompt, textvariable=error_var, foreground="#a12622").grid(
        row=2, column=0, padx=16, pady=(5, 0), sticky="w"
    )

    try:
        experiments = _load_experiment_names()
    except Exception:
        logger.exception("Unable to load GEECS experiments")
        experiments = []
        error_var.set("Experiment list unavailable; enter an experiment name manually.")
    experiment_box.configure(values=experiments)
    if experiments:
        experiment_box.current(0)
    experiment_box.focus_set()

    def open_selected_experiment() -> None:
        experiment = selected_experiment.get().strip()
        if not experiment:
            error_var.set("Select an experiment first.")
            return
        try:
            interlock = InterlockConstructor(
                experiment,
                host=bind_host,
                port=port,
                staleness_timeout_ms=staleness_timeout_ms,
            )
        except Exception as error:
            logger.exception("Unable to initialize experiment '%s'", experiment)
            error_var.set(f"Could not open experiment: {error}")
            return
        prompt.destroy()
        InterlockGUI(interlock, title=f"{title} - {experiment}").run()

    ttk.Button(prompt, text="Continue", command=open_selected_experiment).grid(
        row=3, column=0, padx=16, pady=12, sticky="e"
    )
    prompt.protocol("WM_DELETE_WINDOW", prompt.destroy)
    prompt.bind("<Return>", lambda _event: open_selected_experiment())
    prompt.mainloop()
