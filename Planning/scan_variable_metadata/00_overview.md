# Scan-variable & PV-metadata vocabulary — decisions and deferred work

Status: **first cut landed on `feat/scanvar-pv-metadata`** (2026-07-09,
branched from `feat/vision-v1`). This note captures the reasoning from the
design conversation that produced that branch, and — importantly — records the
pieces we deliberately deferred because they need lab-network / DB work that
can't be done or verified offline.

## What this is about

Three of the vision schemas' recurring questions turned out to be one
question: **what is a device fact vs. an editorial/config decision?** The
vision doc's own principle (§4.2) already answers it — "device facts live
below the configs; limits, units, tolerances, and choices belong to the GEECS
DB and surface as PV metadata from the gateway." The configs above should
*reference or override* device facts, never *restate* them. Everything below
is an application of that one rule.

## Landed in this branch (offline-safe, tested)

1. **`PseudoTarget` → `PseudoComponent`** (`geecs_schemas.scan_variables`).
   Pure clarity rename: a `PseudoScanVariable` *contains* a
   `list[PseudoComponent]`; the old name collided conceptually with the
   element's own `target` field. The legacy vocabulary called these
   *components*. No behavior change; goldens/schema-reference regenerated.

2. **`ScanVariable.confirm`** — an optional `Device:Variable` naming the
   variable that *measures* the result when it differs from the variable being
   *set*. This documents "topology C" (below) in the one place it bites today.
   Additive, defaults to `None` (⇒ today's behavior), validated as a
   `Device:Variable`. **Declared, not yet enforced** by the engine — mirrors
   how the codebase already carries reserved fields (`at_scan_start`, etc.).
   Deliberately **no tolerance field**: the match tolerance is a device fact
   and stays below the configs (DB / gateway PV metadata), per §4.2.

3. **`VariableSpec.description`** (`geecs_ca_gateway.config`) — the model +
   DB-read plumbing for a PV `.DESC`, clipped to 40 chars (EPICS DBR_STRING)
   with a warning. `from_db_metadata` reads `meta.get("description")`, which is
   simply absent today (⇒ `""`, inert). **The live SQL SELECT and the actual
   CA DESC-serving are NOT in this branch** — see Deferred #1/#2.

4. **CaSettable / CaMotor docstring honesty** (`geecs_bluesky.devices.ca`).
   Corrected the misleading framing that `motor` is *the* way to declare a
   positioner and `CaSettable` is unsafe for stages.

## The completion-semantics taxonomy (the substantive finding)

"Motor vs setpoint" was always a fuzzy proxy for the real axis: **the
relationship between the variable you set and the variable that confirms it.**
Three topologies:

- **A — same variable is setpoint *and* an independent measurement.**
  Stages (`Position.Axis N`), steering magnets (`U_S1H:Current`). The GEECS DB
  tolerance is bound to that variable, so `CaMotor`'s readback poll is correct
  and meaningful. This is the only case where CaMotor is clearly load-bearing.

- **B — command echoes itself / no readback.** Pure solenoids. `CaSettable` is
  right; a readback poll would be vacuous (confirming a value against its own
  echo).

- **C — the variable you set ≠ the variable that measures.**
  `U_EMQTripletBipolar` (production!): the catalog's "EMQ1 Current" writes
  `Current_Limit.Ch1` (a *software limit*), while the measured current is a
  separate variable. GEECS binds its set-completion tolerance to the *written*
  variable, so the set "confirms" a value that is trivially correct and the
  real current is **never checked**. **Neither `CaSettable` nor `CaMotor`
  covers this** — CaMotor polls the readback of the *same* variable it set, so
  it cannot confirm a decoupled measurement.

The discrete cousin of C is a solenoid whose truth is a separate DI indicator
(the laser-shutter pattern) — the future `CaShutter`. C-analog (EMQ) and
C-discrete (shutter) are the *same* abstraction ("set X, confirm on Y, with a
match rule"), differing only in the match rule (tolerance vs equality).

### These gaps already exist in production

The topology-C gap is not a regression this work introduces — it is exactly
how MC / legacy geecs-scanner behaves today, and has for years. So closing any
of it is a strict improvement; the worst case of doing nothing is parity with
today. That framing makes each item below a free option, closeable when cheap.

## DESC / metadata-history discipline

- A PV `.DESC` (whether from the DB or served by the gateway) is a **mutable
  label with no history**. Changing it leaves no record of the prior value and
  does not relabel archived samples.
- **Rule:** `.DESC` carries *stable identity only* ("EMQ1 steering current,
  triplet Ch1"), edited freely when wording improves. Anything **time-varying
  or provenance-bearing** (a recalibration, a filter swap) must go where
  history lives — a git-tracked config overlay or the elog — never `.DESC`.
- EPICS shops get metadata history from **version-controlled IOC `.db` files**;
  our gateway swapped that for a live MySQL source, which is where the history
  vanished. Routing history-bearing metadata through git-tracked config
  (`GEECS-Plugins-Configs`, following the `derived_channels.yaml` precedent)
  restores it. Two vessels, different jobs: the **git overlay** is the
  correctable present (`git blame` = the change log); the **Tiled run
  metadata** stamp is the frozen past (welded to each dataset).

## Alias vs. friendly name (the duplication risk)

Keep the DB `alias` **separate** from `.DESC` — alias is a short handle,
description is a one-line sentence. To avoid two naming authorities that
drift, the scan-variable catalog's display name should **default to the DB
alias**, carrying its own name only as an intentional override. Default chain:
**catalog name → DB alias → raw `Device:Variable`**. This turns the catalog
from a second naming system into an *allow-list with optional relabel* — which
is the residue the vision doc wants (allow-listing + overrides + pseudo math).

## Decisions closed (do not re-litigate)

- **Stay on ophyd-async, not classic ophyd.** The gateway unlocked stock EPICS
  *signals*, not *records* — and records (motorRecord) are where classic
  ophyd's value lives. Switching would re-hand-roll the same convergence logic
  on a sunset framework. The genuine "smarts below the client" option is a
  soft-IOC record layer, a much larger separate decision.
- **`motor` kind stays authored, not derived from the DB.** DB tolerance is an
  imperfect proxy for "has an independent readback," and deriving would add a
  DB dependency to positioner construction for a near-no-op distinction.
- **Action plans:** architecture (compile-to-Bluesky-plan-stubs) is right;
  primitives are a deliberately minimal, bug-compatible port. Two real gaps if
  ever extended: `check` is exact-equality only (flaky on analog readbacks),
  and there is **no escape hatch to a registered Bluesky plan** (`run` only
  nests other ActionPlans). Not urgent; noted so YAML doesn't grow a dialect.

## Deferred — needs DB population / write work

The columns exist (verified against the live DB, 2026-07-09) — the blocker is
now *population* and a *join*, not schema. Findings:

| Column | Table | Populated? |
|---|---|---|
| `alias` | `devicetype_variable` (type default) | **0 / 4626** — unused |
| `alias` | `variable` (per-instance override) | 248 / 2641 (~9%) |
| `description` | `devicetype_variable` | **column does not exist** |
| `description` | `variable` | exists `varchar(1000)`, **0 / 2641 populated** |

Two consequences that revise the earlier plan:

- **`description` lives ONLY on `variable`, not `devicetype_variable`.** The
  gateway's SELECTs read `devicetype_variable`; serving descriptions means a
  `LEFT JOIN variable` + coalesce — the same `devicetype_variable → variable`
  inheritance-chain resolution used for the `:SP` set flag (commit
  `fb9d9786`). Reuse that pattern; it is *not* a one-line "add a column."
- **The width mismatch is load-bearing.** `variable.description` is
  `varchar(1000)`; EPICS `.DESC` caps at **40 chars**. `VariableSpec` clips at
  40 with a warning, but the ≤40 "stable identity" discipline must be applied
  at *authoring* time too, or Phoebus shows half a sentence.

1. **Populate + JOIN the description/alias columns.**
   `GeecsDb.get_device_variables` / `get_experiment_device_variables` must
   `LEFT JOIN variable` to pick up `description` (and prefer `variable.alias`
   over the empty `devicetype_variable.alias`), coalescing per the inheritance
   chain. Nothing is populated yet, so the payoff depends on someone writing
   short (≤40 char) descriptions into `variable.description`. The
   `VariableSpec.description` plumbing already consumes `meta["description"]`
   the moment the SELECT provides it.

2. **Actual CA DESC-serving in the gateway.** The gateway serves a raw
   `dict[str, ChannelData]` pvdb; caproto's `ChannelData` has **no `doc`/DESC
   kwarg** (verified, caproto 1.3.0). Serving `.DESC` needs either explicit
   `<pv>.DESC` string channels in the pvdb or the field-aware record
   machinery — a PV_CONTRACT change that needs live CA-client validation.
   `VariableSpec.description` is plumbed and ready; wiring it to a served field
   is a separate on-network PR (update `PV_CONTRACT.md` + a pinned test).

3. **Per-run DESC / apparatus metadata stamp into Tiled.** The run-metadata
   dict (`md`) is already assembled and injected per run
   (`scan_request_runner`, `run_wrapper`). Stamping signal descriptions (and,
   later, apparatus facts like "ND1.0 filter installed") into it freezes them
   onto each dataset — the "frozen past" vessel. Scope decision when built:
   stamp only scan/save-set devices (cheap, incomplete) vs. a whole-apparatus
   snapshot (complete, heavier).

4. **Name-defaults-to-alias resolution.** The catalog display-name default
   chain (catalog name → DB alias → raw `Device:Variable`). The usable alias
   is `variable.alias` (~9% populated); `devicetype_variable.alias` is empty,
   so the same JOIN as #1 is the source. Implement with #1; coverage grows as
   aliases are populated (a raw `Device:Variable` is the honest fallback until
   then).

5. **The topology-C device (`set X, confirm on Y`).** A device whose `set()`
   completes on a *named, possibly different* confirming variable — analog
   match by tolerance, discrete match by equality. Unifies the future
   `CaShutter` (discrete) with the EMQ case (analog) as one abstraction, not
   two.

   **The measured-variable name is now known** (DB lookup, 2026-07-09):
   `U_EMQTripletBipolar` exposes both `Current_Limit.Ch{1,2,3,4}` (set=True on
   Ch1; the software limit the catalog writes) and `Current.Ch{1,2,3,4}`
   (set=False; the measured current). So the mapping is regular:
   `Current_Limit.ChN → Current.ChN`. The catalog exposes Ch1–3 (EMQ1/2/3);
   Ch4 is unused. Writing the `confirm:` entries is ~3 lines per channel:

   ```yaml
   EMQ1 Current:
     target:  U_EMQTripletBipolar:Current_Limit.Ch1
     confirm: U_EMQTripletBipolar:Current.Ch1
   # EMQ2 → .Ch2, EMQ3 → .Ch3
   ```

   **The confirmation is doubly vacuous today, and there is a prerequisite.**
   Every one of these variables has DB `tolerance = 0.0`, *including the
   settable `Current_Limit`*. So GEECS's blocking set "converges" the instant
   the limit register echoes what was written — a zero-tolerance check on a
   value true by construction; the physical `Current.ChN` is never involved.
   A 4b device polling `Current.ChN` therefore needs a **real convergence
   tolerance** to compare against (what counts as "arrived" for these
   supplies — e.g. 0.05–0.1 A). Per the §4.2 discipline that is a *device
   fact* and belongs in the DB (`Current.ChN`'s tolerance), not the config.
   **Prerequisite for closing 4b for real: populate a sensible tolerance on
   `Current.ChN` in the DB** — otherwise an exact-float confirm-poll never
   settles. This is the same "how much PV metadata can the DB serve" thread as
   Deferred #1.
