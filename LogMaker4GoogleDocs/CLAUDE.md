# LogMaker4GoogleDocs — Developer Context for Claude

Google Docs/Drive/Sheets API wrapper for automated experiment logs. Creates
per-day Google Docs from templates, fills in placeholders, uploads images to
Drive, and inserts them into a 2×2 table within each scan's log entry.

Used by ScanAnalysis for automatic post-scan figure upload. Optional everywhere —
missing it causes silent skips, not errors.

## Package Layout

```
logmaker_4_googledocs/
  docgen.py          # All public functions — the entire API surface
  config.ini         # Must contain: [DEFAULT] script = <Apps Script ID>
  credentials.json   # OAuth client secrets — GITIGNORED
  token.pickle       # Cached OAuth tokens — GITIGNORED, auto-generated
  # Experiment INI files also gitignored:
  # HTUparameters.ini, HTTparaeters.ini — contain Drive folder IDs
```

## Authentication & Service

### `establishService(apiservice, apiversion) -> Resource`

Creates (or returns cached) authenticated Google API client.

```python
svc = establishService("script", "v1")    # Apps Script
svc = establishService("drive", "v3")     # Drive
svc = establishService("docs", "v1")      # Docs
svc = establishService("sheets", "v4")    # Sheets
```

- **Cached** in `_service_cache` — one client per (service, version) per process
- **Token refresh** — handles expired tokens transparently; rewrites `token.pickle`
- **First run** — opens browser for OAuth consent; subsequent runs use pickle
- **Requires** `credentials.json` in the module directory — raises
  `FileNotFoundError` if missing
- OAuth scopes: documents, drive, spreadsheets

### Experiment INI Files

Per-experiment config files (`HTUparameters.ini`, `HTTparaeters.ini`) hold
Drive IDs. Gitignored — must be created manually on each machine. Keys used:

```ini
[DEFAULT]
LogID = <Google Doc ID of today's log>     # Written by createExperimentLog
ImageParentFolderID = <Drive folder ID>    # Parent for per-day image subfolders
LogTemplateID = ...
LogFolderID = ...
```

`experiment_mapping` in `docgen.py` maps experiment names to INI files:
`"Undulator"` → `HTUparameters.ini`, `"Thomson"` → `HTTparaeters.ini`.

## Key Functions

### Document Management

**`createExperimentLog(logtempID, tempfolderID, logfolderID, logfilename, argconfig, servicevar)`**

Creates or finds a daily Google Doc by copying a template. Writes the Doc ID
to `argconfig` INI file under `DEFAULT/LogID`. Called once per day.

**`appendToLog(templateID, documentID, search, servicevar)`**

Appends a template block to the doc unless `search` string is already present.
Used to add per-scan sections.

**`findAndReplace(documentID, placeholdersandvalues, servicevar)`**

Replaces `{{placeholder}}` tokens. `placeholdersandvalues` is a `ConfigParser`
whose `DEFAULT` section maps bare key names to replacement values.

### Image Upload

**`insertImageToExperimentLog(scanNumber, row, column, image_path, documentID=None, experiment='Undulator') -> bool`**

The main convenience function for ScanAnalysis integration:

1. Resolves `documentID` from experiment INI if not provided
2. Determines image folder:
   - If `ImageParentFolderID` is in the experiment INI → `get_or_create_folder()`
     creates a per-day subfolder (e.g., `"2025-06-12"`)
   - Else → `_FALLBACK_IMAGE_FOLDER` (staging folder, may be purged)
3. Scales PNG to ~4.75 inches @ 100 DPI via `scale_image()`
4. Uploads via `uploadImage()`
5. Calls Apps Script `insertImageToTableCell()` to insert into the doc

`row` and `column` are 0-based indices into the 2×2 table for that scan.

**`uploadImage(localimagepath, destinationID) -> str | None`**

Uploads to Drive folder. PNG files are pre-scaled; GIFs uploaded as-is.
Returns Drive file ID or `None` on failure.

### Caching

```python
_service_cache = {}        # (apiservice, apiversion) → (credentials, Resource)
_folder_id_cache = {}      # (parent_folder_id, folder_name) → folder_id
```

`get_or_create_folder(parent_folder_id, folder_name)` is cached — only queries
Drive once per (parent, name) per process. Safe for all-day runs.

## Integration with ScanAnalysis

`ScanAnalysis/scan_analysis/gdoc_upload.py` wraps `insertImageToExperimentLog`:

```python
upload_summary_to_gdoc(
    display_files,        # List of paths; uploads display_files[-1]
    scan_number,
    gdoc_slot,            # 0-3 → (row, col) in the 2×2 table
    document_id=None,     # None reads from INI
    experiment="Undulator",
)
```

`gdoc_slot` mapping:
```
0 → row=0, col=0    1 → row=0, col=1
2 → row=1, col=0    3 → row=1, col=1
```

This is set per-analyzer in the scan analysis YAML config. Omitting `gdoc_slot`
puts the analyzer into hyperlink mode (future PR).

## Apps Script Dependency

Several functions call a deployed Google Apps Script project via the Script API.
The `SCRIPT_ID` is stored in `config.ini`. The script handles:

- `createOrFindDoc` — Create or find daily doc from template
- `appendScanEntry` — Add a new scan block from template
- `insertImageToTableCell` — Insert a Drive image into a table cell by scan number

The Apps Script is deployed separately and is **not** in this repo. Changes to
the script require deployment via the Google Apps Script editor.

## Future: Hyperlinks Feature

`upload_display_files_and_link()` and `append_link_to_scan()` are implemented
but require the Apps Script function `appendLinkToScan(documentID, scanNumber,
label, url)` to be deployed first. This will enable uploading all analyzer
outputs as clickable hyperlinks in the "Additional diagnostics" cell rather than
embedding images in table slots.

## Typical Day Workflow

```python
from logmaker_4_googledocs import docgen as g

# 1. Authenticate
svc = g.establishService("script", "v1")

# 2. Create today's log (once per day)
doc_id = g.createExperimentLog(
    LOG_TEMPLATE_ID, TEMPLATE_FOLDER_ID, LOG_FOLDER_ID,
    f"Experiment Log {date.today()}", config_path, svc
)

# 3. Per scan: add section + fill placeholders
g.appendToLog(SCAN_TEMPLATE_ID, doc_id, search=f"Scan {scan_num:03d}", servicevar=svc)
cfg = configparser.ConfigParser()
cfg["DEFAULT"] = {"ScanNumber": f"{scan_num:03d}", "Description": description}
g.findAndReplace(doc_id, cfg, servicevar=svc)

# 4. After analysis: insert image (called by ScanAnalysis automatically)
g.insertImageToExperimentLog(
    scanNumber=scan_num, row=0, column=0,
    image_path="/path/to/summary.png",
    documentID=doc_id,
)
```

## Error Handling

- Google API / Apps Script errors are logged but not re-raised — workflows
  continue even if upload fails
- `None` or `False` return values signal failure
- Missing `credentials.json` raises `FileNotFoundError` immediately on first
  `establishService()` call
- Token expiry handled transparently within `establishService()`

## Setup on a New Machine

1. Copy `credentials.json` (OAuth client) into the module directory
2. Create `HTUparameters.ini` (or experiment equivalent) with Drive folder IDs
3. Run `establishService("drive", "v3")` once — browser opens for OAuth consent
4. `token.pickle` is written; subsequent runs are non-interactive
