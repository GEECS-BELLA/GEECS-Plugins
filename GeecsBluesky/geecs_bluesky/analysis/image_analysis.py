"""Adapters from ImageAnalysis analyzers to GeecsBluesky analysis results."""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Any

import numpy as np

from geecs_bluesky.analysis.models import AnalysisResult, AnalysisScope, InputAssetRef
from geecs_data_utils.config_roots import scan_analysis_config
from geecs_data_utils.io.array1d import Data1DResult

DEFAULT_GEECS_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini").expanduser()


def resolve_analysis_config_dir(
    config_dir: str | Path | None = None,
    *,
    geecs_config_path: str | Path = DEFAULT_GEECS_CONFIG_PATH,
) -> Path:
    """Resolve the unified Scan/ImageAnalysis config directory.

    Parameters
    ----------
    config_dir : str or Path, optional
        Explicit config directory. When supplied, it wins over all defaults.
    geecs_config_path : str or Path, optional
        Shared GEECS user config path to inspect when no explicit directory
        is provided.

    Returns
    -------
    Path
        Directory that should be passed to ``image_analysis.config`` loaders.

    Raises
    ------
    ValueError
        If no unified analysis config directory can be resolved.
    """
    if config_dir is not None:
        return Path(config_dir).expanduser()

    geecs_config = Path(geecs_config_path).expanduser()
    if (
        geecs_config == DEFAULT_GEECS_CONFIG_PATH
        and scan_analysis_config.base_dir is not None
    ):
        return scan_analysis_config.base_dir

    if config_path := _config_dir_from_geecs_config(geecs_config):
        return config_path

    raise ValueError(
        "Could not resolve unified analysis config directory. Set "
        "SCAN_ANALYSIS_CONFIG_DIR, pass config_dir explicitly, or configure "
        "scan_analysis_configs_path in "
        f"{geecs_config}."
    )


def _config_dir_from_geecs_config(config_path: Path) -> Path | None:
    """Read the analysis config root from the shared GEECS user config."""
    if not config_path.exists():
        return None

    config = ConfigParser()
    config.read(config_path)
    if not config.has_section("Paths"):
        return None

    configured = config["Paths"].get("scan_analysis_configs_path")
    if not configured:
        return None
    path = Path(configured).expanduser()
    if path.exists():
        return path

    return None


def resolve_image_analysis_config_dir(
    config_dir: str | Path | None = None,
    *,
    geecs_config_path: str | Path = DEFAULT_GEECS_CONFIG_PATH,
) -> Path:
    """Resolve the config directory for ImageAnalysis loaders.

    This compatibility alias preserves the old ImageAnalysis wording while
    resolving only through the unified scan-analysis config root.
    """
    return resolve_analysis_config_dir(
        config_dir,
        geecs_config_path=geecs_config_path,
    )


class ImageAnalyzerAdapter:
    """Wrap an ImageAnalysis-style analyzer for the GeecsBluesky runner.

    The wrapped object only needs to expose ``analyze_image(image,
    auxiliary_data=None)`` and return an object with either ``feature_scalars()``
    or a ``scalars`` mapping. This keeps GeecsBluesky free of hard imports on
    concrete ImageAnalysis analyzer classes.
    """

    def __init__(
        self,
        image_analyzer: Any,
        *,
        analyzer_id: str,
        analyzer_name: str | None = None,
        analyzer_version: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.image_analyzer = image_analyzer
        self.analyzer_id = analyzer_id
        self.analyzer_name = analyzer_name or type(image_analyzer).__name__
        self.analyzer_version = analyzer_version
        self.analysis_scope = AnalysisScope.EVENT
        self._config = dict(config or {})

    def describe_config(self) -> dict[str, Any]:
        """Return the serializable analyzer configuration."""
        return dict(self._config)

    def analyze(
        self,
        data: Any,
        *,
        asset: InputAssetRef,
        output_dir: Path,
    ) -> AnalysisResult:
        """Analyze one filled image/array and return portable feature scalars."""
        _ = output_dir
        analyzer_input, auxiliary_data, data_metadata = _coerce_analyzer_input(
            data,
            image_analyzer=self.image_analyzer,
        )
        if data_metadata and hasattr(self.image_analyzer, "data_metadata"):
            self.image_analyzer.data_metadata = data_metadata
        auxiliary_data.update(
            {
                "raw_run_uid": asset.raw_run_uid,
                "event_uid": asset.event_uid,
                "scan_event_index": asset.scan_event_index,
                "datum_id": asset.datum_id,
                "device": asset.device,
                "data_key": asset.data_key,
            }
        )
        result = self.image_analyzer.analyze_image(
            analyzer_input,
            auxiliary_data=auxiliary_data,
        )
        return AnalysisResult(features=_feature_scalars(result))


def _coerce_analyzer_input(
    data: Any,
    *,
    image_analyzer: Any,
) -> tuple[Any, dict[str, Any], dict[str, str]]:
    """Return array-like data plus ImageAnalysis auxiliary metadata."""
    if not isinstance(data, Data1DResult):
        return _select_line_columns(data, image_analyzer=image_analyzer), {}, {}

    auxiliary_data: dict[str, Any] = {}
    if getattr(data, "auxiliary_column_data", None):
        auxiliary_data["_aux_columns"] = dict(data.auxiliary_column_data)

    metadata = {
        "x_units": getattr(data, "x_units", None),
        "y_units": getattr(data, "y_units", None),
        "x_label": getattr(data, "x_label", None),
        "y_label": getattr(data, "y_label", None),
    }
    data_metadata = {key: value for key, value in metadata.items() if value is not None}
    return (
        _select_line_columns(data.data, image_analyzer=image_analyzer),
        auxiliary_data,
        data_metadata,
    )


def _select_line_columns(data: Any, *, image_analyzer: Any) -> Any:
    """Apply a 1D analyzer's configured x/y columns to loaded text arrays."""
    line_config = getattr(image_analyzer, "line_config", None)
    if line_config is None:
        return data
    array = np.asarray(data)
    if array.ndim != 2 or array.shape[1] <= 2:
        return data

    data_loading = line_config.data_loading
    return array[:, [data_loading.x_column, data_loading.y_column]]


def _feature_scalars(result: Any) -> dict[str, Any]:
    if hasattr(result, "feature_scalars"):
        return dict(result.feature_scalars())
    return dict(getattr(result, "scalars", {}))
