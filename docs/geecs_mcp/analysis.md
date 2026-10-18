# The analysis domain

The analysis domain closes the loop the scan service opens: after a scan
runs, an agent can start the standard analysis on it, wait for the job,
and read what it produced — the numbers and the figures. All of it goes
through the [Data Portal](../web_services/data_portal.md)'s HTTP API
(`:8200`): `run_analysis` is the same job the portal's **Analysis tab**
starts, on the same [ScanAnalysis](../scan_analysis/overview.md) documents,
so an operator watching the tab sees the agent's run as it happens. The
server runs no analysis of its own.

A scan is named by **scan number and day** (ISO, today when omitted). The
portal's jump route turns that into the run; a number the day does not
hold is `not_found` — never the day's newest run by mistake.

## The verbs

| Tool | Class | What it does |
|---|---|---|
| `list_analyzers` | R | Every analysis document the portal can run: `id`, `device`, `output_dir`, `destructive`, `kind`. The ids are what `run_analysis` takes |
| `list_analysis_groups` | R | The analysis groups as written, with their members in run order. Not runnable as one unit: run the members in turn |
| `get_scan_analysis(scan_number, day?)` | R | The portal's analysis picture of one scan: per analyzer, whether it applies, its job record (log cut to a 40-line tail), its files and artifacts |
| `run_analysis(scan_number, analyzer, day?, confirm?)` | W | Start one analyzer on one scan. Returns at once with the portal's job record (usually `queued`) |
| `wait_for_analysis(scan_number, analyzer, day?, timeout_s=90)` | R | Block until that job leaves `queued` / `running` (cap 110 s); returns `state` (`done` / `failed` / `no_data` / `unknown`), the error, artifacts and files |
| `get_analysis_artifact(scan_number, path, day?, max_bytes=65536)` | R | Read one non-image artifact: a `*_summary_*.json` sidecar comes back parsed as `data`, a text table as `text`. Capped (ceiling 256 KiB) with `truncated` and `size_bytes` when cut |
| `get_scan_scalars(scan_number, columns, day?, x?, bincfg?, filters?)` | R | Per-bin centres, error bands and counts for up to four scalar columns — the Plot tab's numbers, without its figure or notebook snippet |
| `get_scan_figure(scan_number, label?, day?, thumbnail=false)` | R | A figure **reference**: the figure path, analyzer, kind and the portal's artifact URL. `thumbnail=true` adds a ≤768 px JPEG as an MCP image block |

## Running analysis

The portal runs **one job per scan at a time**: a second request while
one is active is `policy_refusal` carrying the active job. An analyzer id
the scan has no diagnostic for is `not_found` with the catalog's ids. A
**destructive** kind needs `confirm=<scan_number>` — the same typed
confirmation the Analysis tab asks a person for. The agent passes it; the
server never fills it in, and without it the portal's refusal comes back
as `invalid_request`.

`wait_for_analysis` polls the same listing. A timeout is a result
(`finished: false`) — the agent calls again. If the portal restarted and
has no job record, the state is `unknown` with the files on disk listed,
never guessed as done.

`run_analysis` takes no `operator` argument: the portal's analysis route
has no field for one, so the job is not attributed to the supervising
person today.

## Reading results

The analysis listing gives job status and artifact descriptors, not
measured values. The loop decides on numbers, so two bounded reads cover
them:

- `get_analysis_artifact` reads what an analyzer wrote — the summary
  sidecar, a per-shot table — by the `path` the listing shows. Images are
  refused (use `get_scan_figure`); a path outside the scan's analysis
  folder is the portal's `not_found`.
- `get_scan_scalars` asks the portal for binned statistics over the
  scan's scalar data. `bincfg` and `filters` are the portal's own JSON
  strings, passed through unparsed, so a binning an operator set up in
  the Plot tab is reproducible from its URL; without `bincfg` the
  portal's default binning applies (the scan variable's steps on a
  stepped scan). More than four columns, or an unknown one, is the
  portal's refusal.

Neither read interprets a column or a file: the meaning stays with the
portal and with the analysis recipe.

Figures follow the payload doctrine strictly: image bytes never ride the
agent's context by default (a single full-size PNG once swamped a
model's context window). A reference's URL is the portal's own artifact
URL, so a client fetches the original bytes directly; the opt-in
thumbnail is the only image the model sees, and it is returned as an
image block rather than text. The same `figure` path is what
[`post_logbook_entry`](logbook.md) takes to attach the figure to an
entry.

## A typical sequence

```text
list_analyzers                                → ids, incl. "Amp2Input"
run_analysis(12, "Amp2Input")                 → ok, job: queued
wait_for_analysis(12, "Amp2Input")            → finished, state: done, artifacts
get_analysis_artifact(12, "<summary>.json")   → data: {...}
get_scan_scalars(12, ["<column>"])            → bins, centres, bands
get_scan_figure(12)                           → figure reference (portal URL)
run_analysis(999, "Amp2Input")                → not_found (the day has no scan 999)
```
