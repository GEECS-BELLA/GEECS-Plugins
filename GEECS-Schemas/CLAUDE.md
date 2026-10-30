# GEECS-Schemas

The shared configuration vocabulary. Runtime dependencies are Pydantic and
GEST's VOCS model; importing schemas must not require numerical libraries,
Bluesky, devices, Xopt or analysis execution. YAML is serialization, not a
second schema language.

All models inherit `SchemaModel` (`_base.py`), which rejects unknown fields.
Standalone config documents inherit `VersionedSchemaModel`; nested payloads
inherit the enclosing document's version. Export public vocabulary from
`__init__.py`. `SCHEMA_REGISTRY` contains only standalone document kinds,
not every nested model. Changes to registered documents also regenerate the
published JSON schemas and Markdown references through existing tooling.

## Analysis documents

Two documents share `scan_analysis_configs/analyzers/`: the
`AnalysisRecipe` (`analysis/recipe.py`, format 3, the analysis core's
native shape) and the `HimgJob` (`analysis/himg.py`: the `.himg`
data-management jobs, not analyses — a top-level `kind`, its own
`schema_version: 1`). `load_analysis_document` reads a top-level `kind`
as a job (a recipe has none), else validates a recipe, which refuses
another `schema_version` and the retired v2 layout (`analyzer:` +
`image:`); every loader goes through it. `processing_1d` keeps only
`Data1DType` / `Data1DLoading`, the line input's loading section. Every document carries
the same neutral names (`device`, `data_folder`, `effective_output_name`,
`destructive`, `scan.priority`, `source_id`) so a host lists them
without asking which it holds. The recipe types the document's frame —
naming, `input`, `inputs`, `scan`, `figure` (`FigureStyle`, the per-frame
draw) and the summary kinds' option models (`image_grid`, `waterfall`,
`average`, `scalar_fit`, each declaring the frame dimensionality it draws,
which the document checks against the input) — and carries `steps` /
`measure` as registry references (`StepRef` / `MeasureRef`, extra keys
allowed on purpose): the numerical vocabulary is GEECS-Analysis' registry and is
never duplicated here. Add a summary kind's option model here and its
layout in GEECS-Analysis; add a step or measure in GEECS-Analysis alone.

## EPICS device data (the web estate)

`epics/` holds the native-device registry (`NativeDevices`, in
`__init__.py`) and the web estate's device data (`Planning/epics_estate.md`
§2.2). `epics/family.py`: `FamilyFile`, one per IOC record template, its
records by role (`REGISTRY_KIND_FAMILIES`: a registry kind's default
family, for the bridge and the archiver alike); the files are **package data**,
`epics/families/<template>.json`, read by `load_families()` — JSON so the
runtime stays free of PyYAML. `epics/panel_devices.py`: `PanelDevices`,
the configs repo's `<experiment>/panel_devices.yaml` (per-device family,
suffix overrides, labels, `allow`, `note`, `omit`, `hazard`, the archive
opt-ins and opt-outs, the serving `ioc`). Their readers (the PV bridge and
the archiver's registry pass) are on `feat/web-estate`; GeecsCanopy reads
them next. A `NativeDevices` entry may name its family (`template`, one of
its kind's; `family_template` falls back to the kind's default). Both validate against the installed families; tests
pass tiny ones through the validation context (`context={"families":
...}`). The installed families are frozen and shared process-wide; a
family with actions puts every writable role in one, so `allow` decides
every write. A family file changes when its IOC template does;
the pin holding the camera, scope, motor and supply files to GeecsPanels'
own suffix tables (web estate S2) lives on `feat/web-estate`, not here.

## Screens

`screens.py`: `Screen`, a control GUI as a document (`<experiment>/screens/<name>.yaml`,
drawn by GeecsControlRoom): titled groups on a grid, each a kit colour
token (`GroupTone`, never a literal), of widgets of exactly seven kinds
(`WidgetKind`), each a free label and a `Binding` (device + family role +
axis, or a bare PV). No expressions, scripts or conditions, ever. What
depends on the families and instance data (the role exists, the kind fits
its type) is `resolve_screen`, not the model. On a DB device a role is its
GEECS variable name, and `resolve_screen(..., deferred=<the module's roster>)`
binds it without a `Role` (`Bound.deferred`, `Bound.fits`): the page judges
it by the module's word. A device in neither the files nor the roster is
refused. A bare PV is read only; it may carry a writing kind or a flipper's
indicators, drawn disabled. A button may bind `{action: <plan>}` instead,
a plan in the experiment's action library (`resolve_screen(...,
actions=<the plan names>)` refuses another).

## Action plans

`action_plan.py`: `ActionPlanLibrary` (`action_library/actions.yaml`) and
the one step library both executors share, GeecsBluesky's compiler in a
scan and GeecsCanopy's module from a screen button: `flatten` (nested
`run` steps inlined, unknown names and loops refused) and `values_match`
(a `check`'s legacy comparison). A step's semantics change here, once.

## Home layout

`home_layout.py`: `HomeLayout`, the Control Room's launcher
(`<experiment>/home.yaml`): areas of screen names, device-page entries by
family, and the named links (`HomeLink`: scanner, portal, logbook,
config_editor). A link is a word; its URL is a deployment setting, never in
the file. Whether a screen file exists or a family is installed is the
page's check (a disabled entry with the reason), not the model's.

## Sweep payload

`sweep.py` owns the validated JSON payload, not numerical trajectory
construction. `Sweep.trajectory` discriminates axis sweeps and typed patterns
by `kind`; individual axis spacing is another discriminated union. Count is
separate and an empty moving sweep is invalid. Do not impose an arbitrary
axis-count ceiling: five correlated lists are an explicit operator use case.
Correlated axes require equal point counts across range/list/log spacing;
grids allow unequal counts. Lists preserve order and repeats.

`relative` is per axis, distinct from a pseudo positioner's own definition.
The executing plan must restore relative axes after completion/abort;
schema validation does not read baselines or perform restoration. The x2x
variant retains stock semantics: two relative axes, Y traversing half X.

`axis_references()` preserves authored order. `n_steps()` is exact without
allocating positions for axis/square/x2x trajectories and returns `None`
for curved spirals. Never copy Bluesky's geometry here to compute their
count; `geecs_bluesky.trajectory` expands them and owns the exact result.

## Tests

Use the root Poetry environment (`scripts/check.sh GEECS-Schemas`), not a
separate package environment. Test JSON round trips, discriminators,
validation refusals and cheap count behavior. Numerical parity with Bluesky
is tested in GeecsBluesky, which owns those dependencies.
