# The logbook domain

The logbook domain is one verb: `post_logbook_entry` writes the agent's
write-up of a scan into the [logbook](../web_services/logbook.md)'s
**scans** book (GeecsLogbook, `:8400`), filed on the scan's card, with
the figures it cites attached. It is a write (class **W** in the
[safety model](overview.md#the-safety-model)).

## The verb

| Tool | Class | What it does |
|---|---|---|
| `post_logbook_entry(day, body_md, operator, scan?, figures?, payload?)` | W | Create an `agent_analysis` **draft** in the scans book; the portal attaches each figure. Returns `entry_id`, `url` and one `attachments` row per figure |

- `day` is `YYYY-MM-DD`; `scan` files the entry on that scan's card (it
  must be a scan the portal holds that day; omit it for a day-level
  entry).
- `body_md` is Markdown. The script behind a quoted number goes inline as
  a fenced block — attachments are images only.
- `figures` is a list of `{scan_number, path, day?, caption?}`, where
  `path` is a `figure` that [`get_scan_figure`](analysis.md) returned. A
  figure's `day` defaults to the **entry's** day.
- `payload`, optional, is the logbook's analysis payload
  (`{kind: "analysis", analyzer, metrics, figures}`) — the queryable
  numbers. It passes through as given; the logbook validates it.

## A draft, never kept by the agent

Every entry is created with `kind: agent_analysis` and `status: draft`.
The logbook refuses a non-note kind created as kept, and there is no
keep verb here: a person reads the draft and keeps it (or not) in the
logbook. The entry's author is `agent, admin supervision (<operator>)`,
and the same string is the author on each figure send — a record of who
supervised the agent, not a permission (access is OSPREY's permission
lists and hooks). An empty `operator` is `invalid_request`.

## Figures are references

No image bytes pass through the MCP server or the model's context. Each
figure is a portal artifact reference; for each one the server asks the
**portal** to attach it, and the portal fetches the artifact from the
scan's analysis folder and uploads it to the entry — the same send its
Plot tab uses, asked to append to this entry or to nothing: if the
draft has gone (deleted, or the portal is pointed at a different
logbook), the figure is refused rather than filed in a fresh entry. The
portal also refuses a figure from a run that is not this experiment's or
has no day and scan number (`not_found`).

## The order is the recovery rule

1. **Before** anything is written: the entry's `scan` is found in the
   portal's catalog for that day, and every figure's run is resolved and
   its path found among that run's listed images. A bad reference — a
   scan the day does not hold, a mistyped figure name — refuses the whole
   call with nothing posted (`not_found`, with the figures that exist).
2. The entry is created.
3. Each figure is sent through the portal.

Once the entry exists the call is **ok whatever the figures do**. Each
figure gets a row in `attachments` — `attached: true` with its `link`, or
`attached: false` with the service's `error_kind` and message — and
`complete: false` plus a `note` say when any failed. The `entry_id` is
never lost. **Do not post again** after a partial result: that makes a
second entry. The draft can be completed by hand on the logbook page.

## A typical sequence

```text
wait_for_analysis(12, "Amp2Input")       → done
get_scan_figure(12)                      → figure: "<path>.png"
post_logbook_entry("2026-10-16", body_md, operator=…, scan=12,
                   figures=[{scan_number: 12, path: "<path>.png"}])
                                         → ok, entry_id, url, attachments, complete: true
```
