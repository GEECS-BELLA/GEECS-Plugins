"""Writers for GeecsBluesky post-run analysis artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from geecs_bluesky.analysis.models import AnalysisInvocationMetadata, FeatureRow

FEATURE_TABLE_BASENAME = "features"


class AnalysisArtifactWriter:
    """Create sidecar analysis artifacts under a day-level analysis directory."""

    def __init__(
        self,
        *,
        day_analysis_dir: Path,
        scan_number: int,
        analyzer_id: str,
        invocation_id: str | None = None,
    ) -> None:
        self.day_analysis_dir = Path(day_analysis_dir)
        self.scan_number = scan_number
        self.analyzer_id = analyzer_id
        self.invocation_id = invocation_id or default_invocation_id()
        self.output_dir = (
            self.day_analysis_dir
            / f"Scan{scan_number:03d}"
            / analyzer_id
            / self.invocation_id
        )
        self.assets_dir = self.output_dir / "assets"

    def prepare(self) -> None:
        """Create the analysis invocation directory.

        The day-level parent must already exist. This prevents analysis code
        from accidentally creating a missing day tree or raw ``scans/ScanXXX``
        folder while still allowing it to create products under ``analysis/``.
        """
        if not self.day_analysis_dir.parent.exists():
            raise FileNotFoundError(
                f"Day folder does not exist: {self.day_analysis_dir.parent}"
            )
        if self.day_analysis_dir.name != "analysis":
            raise ValueError(
                "day_analysis_dir must be the day-level analysis directory"
            )

        self.day_analysis_dir.mkdir(exist_ok=True)
        scan_dir = self.day_analysis_dir / f"Scan{self.scan_number:03d}"
        scan_dir.mkdir(exist_ok=True)
        analyzer_dir = scan_dir / self.analyzer_id
        analyzer_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=False)
        self.assets_dir.mkdir()

    def write_metadata(self, metadata: AnalysisInvocationMetadata) -> Path:
        """Write ``analysis_metadata.json`` atomically."""
        path = self.output_dir / "analysis_metadata.json"
        _write_json_atomic(path, metadata.model_dump(mode="json", exclude_none=True))
        return path

    def write_features_jsonl(self, rows: list[FeatureRow]) -> Path:
        """Write a dependency-light JSONL feature table."""
        path = self.output_dir / f"{FEATURE_TABLE_BASENAME}.jsonl"
        payload = [row.to_flat_dict() for row in rows]
        _write_jsonl_atomic(path, payload)
        return path

    def write_features_parquet(self, rows: list[FeatureRow]) -> Path:
        """Write a Parquet feature table when pandas/pyarrow are installed."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Writing parquet analysis results requires pandas with a "
                "parquet engine such as pyarrow."
            ) from exc

        path = self.output_dir / f"{FEATURE_TABLE_BASENAME}.parquet"
        table = pd.DataFrame([row.to_flat_dict() for row in rows])
        _validate_required_columns(table.columns)
        tmp_path = path.with_suffix(".parquet.tmp")
        try:
            table.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return path

    def asset_path(self, filename: str) -> Path:
        """Return a path below this invocation's derived-asset directory."""
        path = self.assets_dir / filename
        if path.parent != self.assets_dir:
            raise ValueError("derived asset filename must not contain path components")
        return path


def default_invocation_id(now: datetime | None = None) -> str:
    """Return a UTC timestamp suitable for an invocation directory name."""
    timestamp = now or datetime.now(timezone.utc)
    return timestamp.strftime("%Y%m%dT%H%M%SZ")


def required_feature_columns() -> tuple[str, ...]:
    """Return columns every feature table must contain."""
    return (
        "raw_run_uid",
        "event_uid",
        "scan_number",
        "scan_event_index",
        "shot_number",
        "device",
        "data_key",
        "datum_id",
        "asset_spec",
        "analyzer_id",
        "status",
        "error_message",
        "elapsed_s",
    )


def _validate_required_columns(columns) -> None:
    missing = [column for column in required_feature_columns() if column not in columns]
    if missing:
        raise ValueError(f"feature table is missing required columns: {missing}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    if rows:
        _validate_required_columns(rows[0].keys())
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp:
        for row in rows:
            json.dump(row, tmp, sort_keys=True)
            tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
