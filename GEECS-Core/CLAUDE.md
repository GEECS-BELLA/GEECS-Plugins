# GEECS-Core — Developer Context for Claude

The GEECS access **library**: UDP/TCP wire protocol (`transport/`), experiment
MySQL DB (`db/`), PV naming contract (`pv_naming`), the one `GeecsError` tree
(`exceptions`), the `FakeGeecsServer` wire-protocol test double (`testing/`),
and the entry-level synchronous `GeecsDevice` client (`client/`).

**Read `DESIGN.md` before adding anything here.** It carries the three
layering rules (dependencies flow downward; `client/` is the only sync/async
bridge; the admission rule — code belongs here only if *every* consumer of
GEECS devices needs it) and the config.ini stance. This package's value is
what it refuses to contain.

Consumers: GeecsCAGateway and GeecsPvaGateway (servers built on it),
GeecsBluesky and GeecsScanner (`GeecsDb`, `pv_naming`, exceptions), and
end-user scripts (`client`). It was extracted from GeecsCAGateway
(2026-08-20) — the operational wire-protocol history ("quirks that bit us")
still lives in `GeecsCAGateway/CLAUDE.md`.

DB metadata the gateways serve (`db/`): a variable's **description** lives
only on the `variable` table (per instance), never on `devicetype_variable`,
so serving one is a `LEFT JOIN variable` + coalesce — the same
`devicetype_variable → variable` inheritance-chain resolution the `:SP` set
flag uses. Reuse that pattern; it is not a one-line "add a column".
`variable.description` is `varchar(1000)` but EPICS `.DESC` caps at **40
characters**: `VariableSpec` clips at 40 with a warning, and the ≤40
"stable identity only" discipline has to be applied at authoring time too,
or Phoebus shows half a sentence. A `.DESC` is a mutable label with **no
history** — anything time-varying or provenance-bearing (a recalibration, a
filter swap) belongs in a git-tracked config overlay or the elog, never
there. The `alias` column is a separate handle from the description and is
kept separate deliberately.

Import hygiene: `import geecs_core.transport` must stay stdlib-only — the
package `__init__` exports the exception tree eagerly and everything heavier
lazily. Do not add eager `db`/`client` imports there.

Testing: fully offline (`poetry run pytest tests -q`) — the fake server plays
the device, `test_geecs_db` fakes the MySQL connector, `asyncio_mode = "auto"`,
`fake_server`-marked tests get an automatic 30 s timeout (conftest), and
`integration`-marked tests (real lab DB) are deselected by default.

Repo-wide conventions apply (root `CLAUDE.md`): Pydantic v2, NumPy docstrings,
type hints, `poetry version` + `CHANGELOG.md` on every code-changing PR.
