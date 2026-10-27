"""The routes: ``pages`` (the HTML) and ``api`` (the JSON), over one :class:`Context`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import HTTPException, Request
from fastapi.templating import Jinja2Templates
from geecs_data_utils.folder_catalog import FolderScanCatalog
from geecs_data_utils.tiled_catalog import RunDetail

from geecs_plotter import scans
from geecs_plotter.settings import SettingsStore

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@dataclass(frozen=True)
class Context:
    """What every route needs: experiment, catalog, templates, settings store."""

    experiment: str
    templates: Jinja2Templates
    catalog: FolderScanCatalog
    store: SettingsStore

    def run(self, day: date, scan: int) -> RunDetail:
        """The scan, or a 404 naming what was looked for."""
        found = scans.find_scan(
            day, scan, experiment=self.experiment, catalog=self.catalog
        )
        if found is None:
            raise HTTPException(
                status_code=404, detail=f"no Scan{scan:03d} on {day.isoformat()}"
            )
        return found


async def read_form(request: Request) -> dict[str, str]:
    """The urlencoded body as a dict (the page posts no files)."""
    body = (await request.body()).decode("utf-8")
    return dict(parse_qsl(body, keep_blank_values=True))
