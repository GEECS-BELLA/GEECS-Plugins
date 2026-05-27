# ConfigFileGUI

**Point-and-click editor for the YAML files that drive Image Analysis and
Scan Analysis.**

ConfigFileGUI replaces hand-editing YAML in a text editor. It opens a
`scan_analysis_configs/` directory, walks its `analyzers/` and `groups/`
subtrees, and renders a form-driven editor for each file — typed, validated
against the Pydantic schema, and round-trippable on save.

Use it when you want to:

- Author or edit a per-diagnostic analyzer config (a `CameraConfig` for a
  camera, a `Line1DConfig` for a 1D trace).
- Build a **group** — a named collection of analyzer refs that LiveWatch (or
  any other consumer) can dispatch as a unit.
- Sanity-check a config: the editor validates as you type, and the YAML
  preview pane shows the exact bytes that will hit disk on save.

## Launch

```bash
poetry run python ScanAnalysis/ConfigFileGUI/main.py
```

Optionally point the GUI at a specific configs directory:

```bash
poetry run python ScanAnalysis/ConfigFileGUI/main.py \
    --scan-config-dir /path/to/scan_analysis_configs
```

Without `--scan-config-dir` you start on a blank tree and pick a root via
**File → Open Directory…**. The chosen root is remembered between sessions.

## The main window

![ConfigFileGUI initial view, with the analyzer/group tree on the left and
the editor placeholder on the right](
assets/configgui_01_initial.png)

Three panels, left to right:

- **Tree (left, 280 px).** Shows the `analyzers/` and `groups/` subtrees of
  the chosen scan_analysis_configs root, grouped by namespace (e.g. `HTU/`,
  `HTT/`, `PW/`). Single-click a YAML file to load it into the editor.
  The path label above the tree shows the current root; the `Browse…`
  button switches roots. Refresh, Rename, and New Config sit along the
  bottom.
- **Editor (centre, 820 px).** Renders the typed editor for whichever file
  you have selected — an *analyzer editor* for files under `analyzers/`,
  a *group editor* for files under `groups/`. The Save button below the
  editor commits the current state to disk (also bound to `Ctrl+S`).
- **YAML preview (right, hidden by default).** Toggle with
  **Tools → Toggle YAML Preview**. Live view of the serialised YAML that a
  save would produce — useful when debugging schema-related surprises.

A status bar at the bottom shows the last-loaded path and any validation
messages.

## Editing an analyzer config

Click any analyzer YAML in the tree. The editor renders the typed form for
that diagnostic:

![ConfigFileGUI showing a loaded camera analyzer config with the Image
section expanded](
assets/configgui_02_analyzer_camera.png)

The top fields (`name`, `image_analyzer`, `kwargs`) are the analyzer's
identity and the dotted import path of the Python class that runs against
each shot. Below them, the **Image** section is the per-frame processing
pipeline:

- `type: camera` or `type: line` chooses which schema applies; switching it
  re-renders the form.
- General settings (description, bit depth) live above the pipeline.
- The collapsible blocks (**Background**, **Vignette**, **Crosshair
  Masking**, **ROI**, **Circular Mask**, **Thresholding**, …) are the
  individual processing steps. Each has its own header checkbox: tick it to
  include the step in the active pipeline, leave it unchecked to omit it.
  The order of steps as they run is the order in the `pipeline.steps` list,
  not the visual order of the panels.

Below the Image section, the **Scan** section configures how this analyzer
is invoked at the scan-orchestration level (priority, mode, output slot).

The editor validates on every change. If a field is invalid (wrong type,
out-of-range value, missing requirement for the chosen method), the status
bar flags it and Save refuses the write.

## YAML preview

Toggle **Tools → Toggle YAML Preview** (`Ctrl+Y` if bound) to open the
right-hand pane:

![ConfigFileGUI with the YAML preview pane open, showing the serialised
representation alongside the form editor](
assets/configgui_03_yaml_preview.png)

The preview updates as you type, so you can confirm exactly what gets
written. A **Copy to Clipboard** button is handy when you want to paste the
config into a script, an issue tracker, or a colleague's editor.

## Editing a group config

Click a YAML under `groups/`. The editor switches to the group-editor view:

![ConfigFileGUI showing a loaded group config with its analyzer roster and
per-entry priority/enable controls](
assets/configgui_04_group.png)

A group is a named list of analyzer references that LiveWatch (and other
consumers) run together as a unit. The editor exposes:

- **`name`** and **`description`** — the human-readable identity.
- **`upload_to_scan-log`** — flag that controls whether each analyzer's
  display files get sent to the Google Doc e-log.
- **`Add analyzer`** — type-ahead picker over every analyzer YAML the tree
  knows about. The picker filters as you type; click **Add** to append.
- **Per-analyzer rows** — a checkbox (include / exclude this run), a
  priority dropdown, and a remove button. Priority can be left unset
  (inherit the analyzer's own `scan.priority`) or overridden per-group.

Each entry is either a bare string (e.g. `Amp4Input`) or a dict (`{ref:
Amp4Output, enabled: false}`). The editor handles the form conversion both
ways; you don't need to worry about which form a row uses.

## Where files live on disk

ConfigFileGUI expects this layout under the root you opened:

```
scan_analysis_configs/
├── analyzers/
│   ├── HTU/
│   │   ├── UC_TopView.yaml      ← CameraConfig / Line1DConfig
│   │   ├── Amp4Input.yaml
│   │   └── …
│   └── HTT/
│       └── …
└── groups/
    ├── HTU/
    │   ├── baseline.yaml         ← AnalysisGroupConfig
    │   └── …
    └── HTT/
        └── …
```

Each YAML round-trips through the editor — load, edit, save — and the
on-disk file stays canonically formatted. Files outside the
`analyzers/`/`groups/` subtree are ignored by the tree but not by the
filesystem; you can keep README files, archives, or experimental drafts
alongside without disturbing the GUI.

## Analysis Preview (optional)

**Tools → Analysis Preview…** opens a dialog that runs the currently-loaded
analyzer against a single test image and shows the result inline. Useful
when developing a config and you don't want to wait for a full scan to find
out whether the pipeline does what you expect. The dialog asks for an
image file and an output directory, then dispatches the analyzer just as
LiveWatch would, surfacing any errors directly.

## See also

- The [end-to-end tutorial](tutorial.md) walks through building one
  CameraConfig in ConfigFileGUI, referencing it from a group, and running
  the group via LiveWatch.
- [LiveWatch](live_watch.md) — the runner that consumes the configs
  ConfigFileGUI produces.
- [Image Analysis overview](../image_analysis/overview.md) — what each
  pipeline step in the editor actually does to an image.
