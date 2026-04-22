"""Toggle-able YAML preview panel for the Config File Editor GUI.

Displays a live, read-only YAML serialization of the current editor
state.  Includes a "Copy to Clipboard" button.

This module is consumed by ``config_editor_window.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import yaml

from .config_io import sanitize_for_yaml
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class YamlPreviewPanel(QWidget):
    """Read-only panel showing live YAML serialization of the config.

    The text is displayed in a monospace font and is selectable for
    manual copy.  A toolbar button copies the full YAML to the system
    clipboard.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the panel layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._copy_btn = QPushButton("Copy to Clipboard")
        self._copy_btn.clicked.connect(self._on_copy)
        toolbar.addWidget(self._copy_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # YAML text display
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)

        # Monospace font — prefer Consolas on Windows, fallback to Courier New
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        if not font.exactMatch():
            font = QFont("Courier New", 10)
            font.setStyleHint(QFont.Monospace)
        self._text_edit.setFont(font)

        layout.addWidget(self._text_edit, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_preview(self, config_dict: Dict[str, Any]) -> None:
        """Serialize the config dict to YAML and display it.

        Parameters
        ----------
        config_dict : dict
            The configuration dictionary to serialize.
        """
        try:
            clean_dict = sanitize_for_yaml(config_dict)
            yaml_text = yaml.safe_dump(
                clean_dict,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
                allow_unicode=True,
            )
            self._text_edit.setPlainText(yaml_text)
        except Exception as exc:
            logger.warning("Failed to serialize config to YAML: %s", exc)
            self._text_edit.setPlainText(f"# Error serializing YAML:\n# {exc}")

    def clear(self) -> None:
        """Clear the YAML preview text."""
        self._text_edit.clear()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_copy(self) -> None:
        """Copy the full YAML text to the system clipboard."""
        text = self._text_edit.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
                logger.debug("YAML copied to clipboard (%d chars)", len(text))
