"""Derived Bluesky run documents for post-run analysis sidecars."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from event_model import compose_run

from geecs_bluesky.analysis.models import AnalysisInvocationMetadata
from geecs_bluesky.assets.tiled_readback import read_tiled_config

Document = tuple[str, dict[str, Any]]


def build_analysis_run_documents(
    metadata: AnalysisInvocationMetadata,
    *,
    output_dir: Path,
) -> list[Document]:
    """Build a small derived Bluesky run that points at analysis sidecars."""
    feature_table = metadata.feature_table or "features.jsonl"
    effective_output_dir = Path(metadata.analysis_output_dir or output_dir)
    metadata_path = effective_output_dir / "analysis_metadata.json"
    feature_table_path = effective_output_dir / feature_table
    run_bundle = compose_run(
        metadata={
            "purpose": "geecs_bluesky_analysis",
            "analysis_kind": "post_run_sidecar",
            "analysis_id": metadata.analysis_id,
            "analyzer_id": metadata.analyzer_id,
            "analyzer_name": metadata.analyzer_name,
            "analyzer_version": metadata.analyzer_version,
            "analysis_scope": metadata.analysis_scope.value,
            "analysis_of": metadata.raw_run_uid,
            "raw_run_uid": metadata.raw_run_uid,
            "scan_number": metadata.scan_number,
            "experiment": metadata.experiment,
            "analysis_output_dir": str(effective_output_dir),
            "analysis_metadata_path": str(metadata_path),
            "feature_table_path": str(feature_table_path),
            "schema_version": metadata.schema_version,
        }
    )
    metadata.derived_run_uid = run_bundle.start_doc["uid"]
    descriptor_bundle = run_bundle.compose_descriptor(
        name="analysis",
        data_keys=_analysis_data_keys(),
        object_keys={"analysis": list(_analysis_data_keys())},
        configuration={},
    )
    event = descriptor_bundle.compose_event(
        data={
            "analysis_id": metadata.analysis_id,
            "raw_run_uid": metadata.raw_run_uid,
            "analyzer_id": metadata.analyzer_id,
            "analysis_scope": metadata.analysis_scope.value,
            "analysis_output_dir": str(output_dir),
            "analysis_metadata_path": str(metadata_path),
            "feature_table_path": str(feature_table_path),
            "input_count": len(metadata.inputs),
            "derived_asset_count": len(metadata.outputs),
        },
        timestamps={key: run_bundle.start_doc["time"] for key in _analysis_data_keys()},
    )
    stop = run_bundle.compose_stop()
    return [
        ("start", run_bundle.start_doc),
        ("descriptor", descriptor_bundle.descriptor_doc),
        ("event", event),
        ("stop", stop),
    ]


def publish_documents(
    documents: Iterable[Document],
    callback: Callable[[str, dict[str, Any]], None],
) -> None:
    """Publish documents to a Bluesky document callback."""
    for name, doc in documents:
        callback(name, doc)


def publish_analysis_run_to_tiled(
    documents: Iterable[Document],
    *,
    tiled_uri: str | None = None,
    tiled_api_key: str | None = None,
) -> None:
    """Publish derived analysis run documents to the configured Tiled catalog."""
    if tiled_uri is None:
        tiled_uri, tiled_api_key = read_tiled_config()
    if not tiled_uri:
        raise RuntimeError(
            "No Tiled URI given and none found in "
            "~/.config/geecs_python_api/config.ini [tiled]"
        )

    try:
        from bluesky.callbacks.tiled_writer import TiledWriter
        from tiled.client import from_uri
    except ImportError as exc:  # pragma: no cover - depends on optional tiled extra
        raise RuntimeError(
            "tiled is not installed; install the GeecsBluesky tiled extra "
            "before publishing analysis runs"
        ) from exc

    writer = TiledWriter(from_uri(tiled_uri, api_key=tiled_api_key))
    publish_documents(documents, writer)


def _analysis_data_keys() -> dict[str, dict[str, Any]]:
    return {
        "analysis_id": {
            "source": "geecs_bluesky://analysis/id",
            "dtype": "string",
            "shape": [],
        },
        "raw_run_uid": {
            "source": "geecs_bluesky://analysis/raw_run_uid",
            "dtype": "string",
            "shape": [],
        },
        "analyzer_id": {
            "source": "geecs_bluesky://analysis/analyzer_id",
            "dtype": "string",
            "shape": [],
        },
        "analysis_scope": {
            "source": "geecs_bluesky://analysis/scope",
            "dtype": "string",
            "shape": [],
        },
        "analysis_output_dir": {
            "source": "geecs_bluesky://analysis/output_dir",
            "dtype": "string",
            "shape": [],
        },
        "analysis_metadata_path": {
            "source": "geecs_bluesky://analysis/metadata_path",
            "dtype": "string",
            "shape": [],
        },
        "feature_table_path": {
            "source": "geecs_bluesky://analysis/feature_table_path",
            "dtype": "string",
            "shape": [],
        },
        "input_count": {
            "source": "geecs_bluesky://analysis/input_count",
            "dtype": "integer",
            "shape": [],
        },
        "derived_asset_count": {
            "source": "geecs_bluesky://analysis/derived_asset_count",
            "dtype": "integer",
            "shape": [],
        },
    }
