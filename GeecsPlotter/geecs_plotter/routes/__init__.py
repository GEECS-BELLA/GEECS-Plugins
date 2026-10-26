"""The routes: ``pages`` (the HTML) and ``api`` (the JSON), over one :class:`Context`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.templating import Jinja2Templates

from geecs_plotter import scans

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@dataclass(frozen=True)
class Context:
    """What every route needs: where the data is, and the templates."""

    experiment: str
    base_directory: Optional[Path]
    templates: Jinja2Templates

    def folder(self, day: date, scan: int) -> Path:
        """The scan's folder, or a 404 naming what was looked for."""
        found = scans.scan_folder(
            day, scan, experiment=self.experiment, base_path=self.base_directory
        )
        if found is None:
            raise HTTPException(
                status_code=404, detail=f"no Scan{scan:03d} on {day.isoformat()}"
            )
        return found
