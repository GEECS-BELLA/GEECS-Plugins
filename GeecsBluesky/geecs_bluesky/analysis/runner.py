"""Minimal execution seam for post-run analysis over filled assets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

from geecs_bluesky.analysis.models import (
    AnalysisInvocationMetadata,
    AnalysisResult,
    AnalysisScope,
    DerivedAssetRef,
    FeatureRow,
    InputAssetRef,
)
from geecs_bluesky.analysis.writer import AnalysisArtifactWriter


class AnalyzerProtocol(Protocol):
    """Algorithm boundary used by the GeecsBluesky post-run runner."""

    analyzer_id: str
    analyzer_name: str
    analyzer_version: str | None
    analysis_scope: AnalysisScope

    def describe_config(self) -> dict[str, Any]:
        """Return a serializable analyzer configuration."""
        ...

    def analyze(
        self,
        data: Any,
        *,
        asset: InputAssetRef,
        output_dir: Path,
    ) -> AnalysisResult:
        """Analyze one filled asset and return scalar/derived outputs."""
        ...


class FilledAsset:
    """Filled input data plus its raw-run identity."""

    def __init__(self, *, ref: InputAssetRef, data: Any) -> None:
        self.ref = ref
        self.data = data


class AnalysisRunner:
    """Run one analyzer over a collection of already-filled assets."""

    def __init__(self, *, writer: AnalysisArtifactWriter) -> None:
        self.writer = writer

    def run(
        self,
        *,
        analyzer: AnalyzerProtocol,
        assets: Iterable[FilledAsset],
        metadata: AnalysisInvocationMetadata,
        feature_table_format: Literal["jsonl", "parquet"] = "jsonl",
    ) -> AnalysisInvocationMetadata:
        """Execute *analyzer* and persist feature rows plus provenance metadata."""
        self.writer.prepare()
        asset_list = list(assets)
        scan_context = _prepare_scan_context(
            analyzer,
            assets=asset_list,
            output_dir=self.writer.assets_dir,
        )
        rows: list[FeatureRow] = []
        outputs: list[DerivedAssetRef] = []
        inputs: list[InputAssetRef] = []

        for filled_asset in asset_list:
            inputs.append(filled_asset.ref)
            started = perf_counter()
            try:
                result = _analyze_asset(
                    analyzer,
                    filled_asset.data,
                    asset=filled_asset.ref,
                    output_dir=self.writer.assets_dir,
                    scan_context=scan_context,
                )
            except Exception as exc:
                result = AnalysisResult.failed(exc)
            if result.elapsed_s is None:
                result.elapsed_s = perf_counter() - started

            outputs.extend(result.derived_assets)
            rows.append(_feature_row(filled_asset.ref, analyzer, result))

        outputs.extend(
            _finalize_scan_context(
                analyzer,
                scan_context=scan_context,
                output_dir=self.writer.assets_dir,
            )
        )
        if feature_table_format == "jsonl":
            feature_path = self.writer.write_features_jsonl(rows)
        elif feature_table_format == "parquet":
            feature_path = self.writer.write_features_parquet(rows)
        else:
            raise ValueError(
                f"unsupported feature table format: {feature_table_format}"
            )

        metadata.inputs = inputs
        metadata.outputs = outputs
        metadata.analysis_scope = _analysis_scope(analyzer)
        metadata.analysis_output_dir = str(self.writer.output_dir)
        metadata.feature_table = feature_path.name
        self.writer.write_metadata(metadata)
        return metadata


def _prepare_scan_context(
    analyzer: AnalyzerProtocol,
    *,
    assets: list[FilledAsset],
    output_dir: Path,
) -> Any:
    prepare = getattr(analyzer, "prepare_scan_context", None)
    if prepare is None:
        return None
    return prepare(assets=assets, output_dir=output_dir)


def _analyze_asset(
    analyzer: AnalyzerProtocol,
    data: Any,
    *,
    asset: InputAssetRef,
    output_dir: Path,
    scan_context: Any,
) -> AnalysisResult:
    analyze_with_context = getattr(analyzer, "analyze_with_scan_context", None)
    if analyze_with_context is not None:
        return analyze_with_context(
            data,
            asset=asset,
            output_dir=output_dir,
            scan_context=scan_context,
        )
    return analyzer.analyze(data, asset=asset, output_dir=output_dir)


def _finalize_scan_context(
    analyzer: AnalyzerProtocol,
    *,
    scan_context: Any,
    output_dir: Path,
) -> list[DerivedAssetRef]:
    finalize = getattr(analyzer, "finalize_scan_context", None)
    if finalize is None:
        return []
    return list(finalize(scan_context=scan_context, output_dir=output_dir))


def _analysis_scope(analyzer: AnalyzerProtocol) -> AnalysisScope:
    value = getattr(analyzer, "analysis_scope", AnalysisScope.EVENT)
    return AnalysisScope(value)


def _feature_row(
    asset: InputAssetRef,
    analyzer: AnalyzerProtocol,
    result: AnalysisResult,
) -> FeatureRow:
    return FeatureRow(
        raw_run_uid=asset.raw_run_uid,
        event_uid=asset.event_uid,
        scan_number=asset.scan_number,
        scan_event_index=asset.scan_event_index,
        shot_number=asset.shot_number,
        device=asset.device,
        data_key=asset.data_key,
        datum_id=asset.datum_id,
        asset_spec=asset.asset_spec,
        analyzer_id=analyzer.analyzer_id,
        status=result.status,
        error_message=result.error_message,
        elapsed_s=result.elapsed_s,
        features=result.features,
        derived_assets={
            output.asset_id: output.relative_path for output in result.derived_assets
        },
    )
