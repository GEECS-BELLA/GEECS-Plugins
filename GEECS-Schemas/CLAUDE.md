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
