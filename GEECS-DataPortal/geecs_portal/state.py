"""The portal's shared state: one :class:`PortalState` on ``app.state``."""

from __future__ import annotations

import dataclasses
import importlib
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
from fastapi import HTTPException
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from geecs_web_theme.web import root_of

from geecs_data_utils.scan_info import analysis_folder_for
from geecs_data_utils.scan_paths import ScanPaths
from geecs_data_utils.tiled_catalog import RunDetail, ScanCatalog

from geecs_portal import analysis_runs, resources
from geecs_portal.cache import ShotDataCache
from geecs_portal.routes.common import _figure_kwargs, _resolved_folder

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _DiagInfo:
    """A loadable diagnostic's run-side facts (the selector cache's value)."""

    #: The data device the wrapper reads: ``scan.device``, else the name
    #: (``data_device_name or device_name`` in ScanAnalysis's wrapper).
    device: str
    #: Per-analyzer output directory under ``analysis/ScanNNN/``.
    output_name: str
    #: A run deletes or rewrites data files (the ``.himg`` compaction):
    #: the tab asks for the scan number first and the start endpoint
    #: refuses without it.
    destructive: bool = False
    #: Which document this is — ``recipe`` (an ``AnalysisRecipe``; a
    #: version-2 file loads as one since the v2 document was deleted,
    #: #1039) or ``job`` (an ``.himg`` job) — so a catalog reader can
    #: tell a measurement from a data-management step without opening
    #: the YAML.
    kind: str = "unknown"

    @classmethod
    def from_diagnostic(cls, diag) -> "_DiagInfo":
        # Either format, through the names both documents carry.
        return cls(
            device=str(diag.data_folder),
            output_name=str(diag.effective_output_name),
            destructive=bool(getattr(diag, "destructive", False)),
            kind=_DOCUMENT_KINDS.get(type(diag).__name__, "unknown"),
        )


#: Document class → the catalog's ``kind`` word (by name: this module
#: imports no schema class).
_DOCUMENT_KINDS = {
    "AnalysisRecipe": "recipe",
    "HimgJob": "job",
}


@dataclasses.dataclass(frozen=True)
class _GroupInfo:
    """A loadable analysis group's facts for the catalog."""

    #: Path-like id relative to ``groups/`` without the suffix
    #: (``HTU/all``) — unique, unlike the bare stem.
    id: str
    name: str
    description: str
    #: The resolved members in run order (priority ascending): enabled
    #: refs only, every one a loadable diagnostic — what a run would get.
    members: tuple[str, ...]


@dataclasses.dataclass
class PortalState:
    """What :func:`~geecs_portal.app.create_app` builds once and every route reads."""

    catalog: ScanCatalog
    templates: Jinja2Templates
    runner: analysis_runs.AnalysisRunner
    data_cache: ShotDataCache
    factory: analysis_runs.AnalyzerFactory
    default_experiment: str = ""
    processing_config_dir: Optional[Path] = None
    analysis_factory: Optional[analysis_runs.AnalyzerFactory] = None
    logbook_base: str = ""
    #: Sending needs an address the PORTAL can reach, not one the browser
    #: can: the call is server-to-server (``geecs_portal.logbook_send``),
    #: which is what keeps it working on a plain-HTTP page and keeps CORS
    #: out of the logbook.  A path-shaped ``--logbook-url`` is a fact about
    #: the browser's front door and names no host this process can dial, so
    #: it links but does not send, and the page hides the button.
    logbook_send_base: str = ""
    #: The client the plot send uses (``None`` = one per send); injected
    #: by tests so the conversation runs against an in-process logbook.
    logbook_client: Optional[httpx.Client] = None
    #: Whether GEECSplotter's page is mounted at ``/plot`` (then it is the
    #: Plot tab; without it the scan page has none).
    plot_page: bool = False
    config_editor_enabled: bool = False  # set when the editor router mounts
    # (fingerprint → valid names): re-validated only when the tree
    # changes — discovery lists every YAML stem, but legacy flat camera
    # configs in the same tree don't LOAD as diagnostics, and offering
    # one puts an unfixable broken image in front of the operator
    # (found live: UNCLASSIFIED/UC_Amp4_IR_input.yaml, 2026-09-01).
    processing_cache: dict = dataclasses.field(default_factory=dict)

    def logbook_url(
        self, request: Request, detail: RunDetail, run_day: Optional[date]
    ) -> str:
        """The run's entry in the scan logbook, or "" when there is none to link.

        The logbook's day page anchors each scan card by its folder name
        (``#Scan012``); the portal knows the logbook's base URL and the
        day, so the link is built here without importing the logbook — a
        peer view layer in its own process, reached by URL like any other
        page. A path-shaped base (``/log``) is same-origin and takes this
        app's own prefix; an absolute one is used verbatim. The logbook
        serves the default experiment alone, and scan numbers restart
        daily per experiment, so a run from another experiment gets no
        link rather than a wrong one.
        """
        summary = detail.summary
        if not (self.logbook_base and run_day and summary.scan_number):
            return ""
        if summary.experiment != self.default_experiment:
            return ""
        base = (
            root_of(request) + self.logbook_base
            if self.logbook_base.startswith("/")
            else self.logbook_base
        )
        return f"{base}/day/{run_day.isoformat()}#Scan{summary.scan_number:03d}"

    def logbook_send_allowed(self, detail) -> bool:
        """Whether a send to the logbook is possible for this run at all.

        An absolute address to dial, and a run of this portal's experiment
        (the logbook serves one experiment's share; scan numbers restart
        per experiment). The destination day and scan are the caller's:
        the start document's for a rendered plot, the resolved folder's
        tag for an artifact (see the send route).
        """
        return bool(
            self.logbook_send_base
            and detail.summary.experiment == self.default_experiment
        )

    def logbook_sendable(self, detail, run_day) -> bool:
        """Whether this run can receive a plot — the link's rule, plus an address."""
        return bool(
            self.logbook_send_allowed(detail) and run_day and detail.summary.scan_number
        )

    def load_run(self, uid: str):
        """Load one run, mapping failures to honest HTTP status codes.

        ``KeyError`` is the fakes' and the Tiled client's unknown-uid
        signal → 404.  Anything else (connection errors, unconfigured
        URI) means the catalog itself is unavailable → 503, so an outage
        never reads as "run not found" for runs that exist.
        """
        try:
            return self.catalog.load_run(uid)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"run not found: {exc}"
            ) from exc
        except Exception as exc:
            logger.warning("catalog load_run failed: %s", exc)
            raise HTTPException(
                status_code=503, detail=f"catalog unavailable: {exc}"
            ) from exc

    def ephemeral_module(self):
        """The portal's backend router, or the feature's 404 ladder.

        The one place the "configured? installed?" preamble lives for
        the processing selector and the rendered view alike.
        """
        if self.processing_config_dir is None:
            raise HTTPException(
                status_code=404,
                detail="processing is not configured on this portal "
                "(start it with --processing-configs)",
            )
        try:
            # Check the optional runtime even when the router is cached.
            importlib.import_module("geecs_analysis.engine.compiled")
            ephemeral = importlib.import_module("geecs_portal.processing")
        except ImportError as exc:
            raise HTTPException(
                status_code=404,
                detail="processing needs the portal's 'analysis' extra "
                "(pip install geecs-data-portal[analysis])",
            ) from exc
        return ephemeral

    def render_processing_figure(self, array, processing: str, render: dict) -> bytes:
        """The rendered view of ONE shot: the analyzer draws its own result.

        Same write-free router as ``apply_processing``: supported recipes
        use core measurements and object-API figures; unported recipes keep
        the legacy ephemeral renderer. Same status ladder as the
        pixel path: unknown diagnostic 404, denylisted / invalid config
        400, analyzer failure 400, and "ran but cannot be drawn"
        (``RenderError``) 404 — the pixel path's "render failed" /
        "produces no processed image" code.
        """
        ephemeral = self.ephemeral_module()
        try:
            (fig,) = ephemeral.render_diagnostic_ephemeral(
                processing,
                [array],
                config_dir=self.processing_config_dir,
                **_figure_kwargs(render),
            )
            return resources.figure_png(fig)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=404, detail=f"no diagnostic: {exc}"
            ) from exc
        except ephemeral.RenderError as exc:
            raise HTTPException(
                status_code=404, detail=f"render failed: {exc}"
            ) from exc
        except ValueError as exc:  # denylisted, or invalid config
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"processing failed: {exc}"
            ) from exc

    def render_frame_figure(self, array, render: dict) -> bytes:
        """The rendered view of an averaged image: base renderer, no overlays."""
        ephemeral = self.ephemeral_module()
        try:
            return resources.figure_png(
                ephemeral.render_frame_figure(array, **_figure_kwargs(render))
            )
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"render failed: {exc}"
            ) from exc

    def apply_processing(self, arrays: list, processing: str) -> list:
        """Ephemeral-process *arrays* → the analyzers' processed images.

        One compiled recipe for the batch, or the retained legacy ephemeral
        route for an unsupported recipe. Refusals map onto the
        endpoint ladder: unknown diagnostic → 404, denylisted/miswired
        → 400, analyzer failure → 400 honestly — never a 500.
        """
        ephemeral = self.ephemeral_module()
        try:
            processed = ephemeral.process_images(
                processing, arrays, config_dir=self.processing_config_dir
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=404, detail=f"no diagnostic: {exc}"
            ) from exc
        except ValueError as exc:  # denylisted, or invalid config
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"processing failed: {exc}"
            ) from exc
        if any(image is None for image in processed):
            raise HTTPException(
                status_code=404,
                detail=f"diagnostic {processing!r} produces no processed image",
            )
        return processed

    def list_day(
        self, experiment: str, day: date, filter_text: str
    ) -> tuple[list, str]:
        """The day's runs (newest first), filtered — ONE implementation.

        Shared by the day page, the JSON day listing and both jump
        routes so the surfaces cannot drift.  A catalog failure is
        returned, not raised (``(runs, error)``): the page renders it
        inline, the API maps it to 503, the jump degrades to the day.
        """
        try:
            runs = list(self.catalog.list_runs(experiment, day))
        except Exception as exc:  # noqa: BLE001 — surface, don't 500
            logger.warning("day listing failed: %s", exc)
            return [], str(exc)
        needle = filter_text.strip().lower()
        if needle:
            runs = [run for run in runs if needle in run.filter_text()]
        return runs, ""

    def neighbours(self, uid: str, experiment: str, run_day: Optional[date]):
        """The scan steppers: ``(prev_uid, next_uid, day_runs)``.

        The day's listing (newest first) feeds both the rail's scan
        dropdown and the stepper neighbours — previous = older, next =
        newer.  A listing failure (or a uid missing from its own day)
        just hides them: ``("", "", [])`` — never sinks the page.
        """
        if run_day is None:
            return "", "", []
        try:
            day_runs = list(self.catalog.list_runs(experiment, run_day))
            day_uids = [run.uid for run in day_runs]
            position = day_uids.index(uid)
        except Exception as exc:  # noqa: BLE001 — stepper is optional
            logger.warning("neighbour listing failed: %s", exc)
            return "", "", []
        next_uid = day_uids[position - 1] if position > 0 else ""
        prev_uid = day_uids[position + 1] if position + 1 < len(day_uids) else ""
        return prev_uid, next_uid, day_runs

    def processing_names(self) -> list[str]:
        """LOADABLE diagnostic IDs for the processing selector (``[]`` = hidden)."""
        return list(self.processing_infos())

    def processing_infos(self) -> dict[str, _DiagInfo]:
        """LOADABLE diagnostics → their run-side facts (``{}`` = hidden).

        Explicit-opt-in: no configured tree, no feature (never the
        global fallback). Degrades to empty — never errors a page —
        when the ``analysis`` extra is not installed or the tree
        cannot be listed. Each discovered stem is validated with a
        real ``load_diagnostic`` (cached against the tree's YAML
        mtimes); invalid ones are dropped with an INFO log naming the
        file, so a legacy config is a log line, not a broken image.
        """
        if self.processing_config_dir is None:
            return {}
        try:
            processing_api = self.ephemeral_module()
            list_diagnostics = processing_api.list_diagnostics
            load_diagnostic = processing_api.load_diagnostic
        except HTTPException:
            return {}
        # Fingerprint BEFORE listing: a YAML landing between the two
        # scans then costs one harmless revalidation, instead of a
        # cache entry permanently missing it. mtime+size so same-second
        # edits are caught even on coarse-mtime SMB-mounted trees.
        tree = Path(self.processing_config_dir) / "analyzers"
        try:
            fingerprint = frozenset(
                (str(path), stat.st_mtime, stat.st_size)
                for pattern in ("*.yaml", "*.yml")
                for path in tree.rglob(pattern)
                for stat in (path.stat(),)  # one syscall, untearable tuple
            )
        except OSError:
            fingerprint = None
        if fingerprint is not None:
            cached = self.processing_cache.get(fingerprint)  # .get: a racing
            if cached is not None:  # clear() must degrade to revalidate
                return cached
        try:
            names = list_diagnostics(config_dir=self.processing_config_dir)
        except Exception as exc:  # noqa: BLE001 — unlistable tree = no selector
            logger.debug("processing configs unavailable: %s", exc)
            return {}
        valid: dict[str, _DiagInfo] = {}
        for name in names:
            try:
                diag = load_diagnostic(name, config_dir=self.processing_config_dir)
                valid[name] = _DiagInfo.from_diagnostic(diag)
            except Exception as exc:  # noqa: BLE001 — one bad YAML must not hide the rest
                logger.info(
                    "processing selector: skipping %r — not a loadable "
                    "unified diagnostic (%s)",
                    name,
                    exc,
                )
        if fingerprint is not None:
            self.processing_cache.clear()  # one entry: the current tree state
            self.processing_cache[fingerprint] = valid
        return valid

    def group_infos(self) -> list[_GroupInfo]:
        """LOADABLE analysis groups under the configured tree (``[]`` = none).

        The analyzer listing's sibling for the catalog, through
        ScanAnalysis's own loader: ``discover_groups`` finds the files
        (one walk, the runtime's rule for the two accepted names) and
        ``load_analysis_group`` validates and RESOLVES each one — so a
        group is listed with exactly the members a run would get, and a
        file that would fail at run time (bad YAML, an invalid document,
        a ref no diagnostic answers to) is dropped with an INFO log
        naming it, as the per-run listing drops an invalid diagnostic.
        Uncached: groups are few and the route is on demand.
        """
        if self.processing_config_dir is None:
            return []
        try:
            from scan_analysis.config.analysis_group_loader import (
                discover_groups,
                load_analysis_group,
            )
        except ImportError:
            return []
        root = Path(self.processing_config_dir)
        try:
            index = discover_groups(root)
        except (FileNotFoundError, OSError):
            return []
        groups: list[_GroupInfo] = []
        groups_dir = root / "groups"
        # The index keys each file twice (stem + path-like); list each FILE once.
        for path in sorted(set(index.values()), key=str):
            try:
                loaded = load_analysis_group(path)
            except (ValueError, FileNotFoundError, OSError) as exc:
                logger.info("analysis catalog: skipping group %s (%s)", path, exc)
                continue
            groups.append(
                _GroupInfo(
                    id=path.relative_to(groups_dir).with_suffix("").as_posix(),
                    name=loaded.name,
                    description=loaded.description or "",
                    members=tuple(r.id for r in loaded.analyzers),
                )
            )
        return groups

    def analysis_enabled(self) -> bool:
        """Whether the scan page should offer the Analysis tab at all."""
        try:
            self.analysis_available()
        except HTTPException:
            return False
        return True

    def analysis_enabled_for(self, folder: Optional[Path]) -> bool:
        """The Analysis tab's gate: the feature on AND a resolvable scan folder.

        Runs and the artifact listing are per scan folder — the page and
        ``/api/run`` read the same answer.
        """
        return folder is not None and self.analysis_enabled()

    def analysis_available(self) -> None:
        """404 unless the run feature is configured AND installed."""
        if self.processing_config_dir is None:
            raise HTTPException(
                status_code=404,
                detail="analysis runs are not configured on this portal "
                "(start it with --processing-configs)",
            )
        if self.analysis_factory is None:
            try:
                import scan_analysis  # noqa: F401 — the real factory's need
            except ImportError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="analysis runs need the portal's 'analysis' extra "
                    "(pip install geecs-data-portal[analysis])",
                ) from exc

    def analysis_context(self, uid: str, day: str):
        """(detail, scan folder, analysis folder, ScanTag) or 404.

        The tag is parsed FROM THE RESOLVED FOLDER (``ScanPaths(folder=…)``,
        read-only), never rebuilt from the start doc: the folder's day
        is the claim-time day, the start doc's ``time`` is stamped later
        — a scan claimed at 23:59:58 and opened at 00:00:01 would
        otherwise run the analyzer on the NEXT day's same-numbered scan
        (and TZ / experiment-spelling drift would do the same).
        """
        detail = self.load_run(uid)
        _, folder = _resolved_folder(detail, day)
        if folder is None:
            raise HTTPException(status_code=404, detail="scan folder not resolvable")
        try:
            tag = ScanPaths(folder=folder, read_mode=True).get_tag()
        except (ValueError, OSError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"scan folder is not a canonical scans/ScanNNN path: {exc}",
            ) from exc
        if tag is None:
            raise HTTPException(status_code=404, detail="scan folder has no tag")
        return detail, folder, analysis_folder_for(folder), tag


def get_state(request: Request) -> PortalState:
    """The app's :class:`PortalState` — the routers' one ``Depends`` accessor."""
    return request.app.state.portal
