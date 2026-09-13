# Pseudo scan variables on the native path: the design brief

**Status:** not started. Split out of phase 3 on 2026-09-13 (`03` §8, "The
pseudo arc"). This document exists so the session that builds it starts from
the survey below rather than repeating it.

**Why it is a merge-gate item.** Composite variables are scanned regularly on
HTU (Sam, 2026-09-13), and **they cannot be scanned at all on this branch**:
`qs_client/presets.py` refuses a `kind: pseudo` axis with "pseudo axes are not
scannable through the namespace yet (phase 3)", and `GeecsNamespace` builds no
noun for one. Master can do this; the branch cannot. That is a lost
capability, and the merge gate is "every piece in place".

---

## 1. The one decision this arc turns on

`CaPseudoMovable` (`geecs_bluesky/devices/ca/pseudo.py`, ~200 lines) is a
hand-rolled `StandardReadable` that fans one number out to several GEECS
targets through compiled `forward` formulas. It is **forward-only**: it can
compute where the targets should go, and cannot compute where the pseudo *is*.

**ophyd-async already has the mechanism, and it is bidirectional by
construction:** `Transform` + `DerivedSignalFactory` (`ophyd_async.core`,
0.19.3 — already installed, no new dependency). A `Transform` subclass
declares two methods:

| our name | ophyd-async | what it does |
|---|---|---|
| `forward` | `derived_to_raw` | derived value → each raw target's setting |
| *(never had one)* | `raw_to_derived` | the raw readbacks → the derived value |

The product is a plain `SignalRW`. It reads, it `locate()`s, it moves, all
through stock bluesky machinery.

**Adopting it is the recommendation.** Not because the current class is bad,
but because the half it is missing is the half ophyd-async requires — so
adopting the stock mechanism *forces* the inverse that this arc has been
deferring since #600, rather than deferring it again.

---

## 2. What that changes, concretely

### #855 as filed is the wrong fix — do not build it

#855 asks for `CaPseudoMovable.locate()`. Its hard case is stated in the
issue: *an absolute pseudo's readback is NaN before its first set, so a
`rel_scan` over one moves to `NaN + offset`.*

A derived signal has no such case. `raw_to_derived` computes the readback from
the component motors' **live readbacks**, so it is defined from the moment the
components are connected — before any set, after a restart, after someone
moved a component by hand. `locate()` is then free and correct.

Hand-rolling `locate()` onto `CaPseudoMovable` builds something the migration
deletes. Close #855 into this arc rather than doing it first.

### `mode: relative` is probably a workaround for the missing inverse

11 of the 13 corpus entries are `mode: relative`, and the class carries real
machinery for them: `_capture_baselines` (read every target at `stage`),
`restore_baselines_plan` (a custom end-of-scan restore), the lazy capture for
unstaged callers, and the `absolute`/`relative` branch in `_target_values`.

With a real readback, **"relative" is just `rel_scan` over the pseudo** —
stock bluesky computes the offsets from `locate()` and stock
`reset_positions_wrapper` restores. So the whole mode split, the baseline
capture and the custom restore plan are candidates for deletion, not
migration.

Verify before deleting: the current relative restore is *exact* (it puts the
captured baselines back, formula-independent, deliberately not assuming
`f(0) = 0` — see the docstring, owner request 2026-07-22). A stock restore
goes back through the inverse. For the affine corpus those agree; confirm that
before dropping the exact path, and keep it if they do not.

### The corpus is nearly all affine, and it names its own reference

13 pseudos, 26 target formulas.

| pseudo | mode | targets |
|---|---|---|
| `ALine_e_beam_angle_offset_x` | relative | `U_S3H:Current` = `composite_var * 1`<br>`U_S4H:Current` = `composite_var * -2` |
| `ALine_e_beam_angle_offset_y` | relative | `U_S3V:Current` = `composite_var * 1`<br>`U_S4V:Current` = `composite_var * -2` |
| `ALine_e_beam_position_offset_x` | relative | `U_S3H:Current` = `composite_var * 1`<br>`U_S4H:Current` = `composite_var * -1` |
| `ALine_e_beam_position_offset_y` | relative | `U_S3V:Current` = `composite_var * 1`<br>`U_S4V:Current` = `composite_var * -1` |
| `CompressorAndMode` | absolute | `U_CompAerotech:Position.Axis1` = `composite_var`<br>`U_ModeImagerESP:Position.Axis 1` = `(composite_var-41000) * 14/1000 - 20` |
| `ERROR_Laser_angle_offset_y` | relative | `U_ESP302_02:Position.Axis 3` = `composite_var * 1.0`<br>`U_ESP302_02:Position.Axis 1` = `composite_var * -1.5` |
| `HexY_S1H_relative` | relative | `U_Hexapod:ypos` = `composite_var`<br>`U_S1H:Current` = `(composite_var)*8` |
| `HexZ_S1V` | absolute | `U_Hexapod:zpos` = `composite_var`<br>`U_S1V:Current` = `-(composite_var + 0.2411) * 9` |
| `HexZ_S1V_relative` | relative | `U_Hexapod:zpos` = `composite_var`<br>`U_S1V:Current` = `-(composite_var)*9.0` |
| `JetZ_with_probe` | absolute | `U_ESP_JetXYZ:Position.Axis 3` = `composite_var`<br>`U_ProbeCamStage:Position` = `8.5 + (composite_var-10)*2.5` |
| `Laser_angle_offset_x` | relative | `U_ESP302_02:Position.Axis 2` = `composite_var * 1.5`<br>`U_ESP302_01:Position.Axis 3` = `composite_var * -1.5` |
| `R56_at_100MeV` | absolute | `U_ChicaneInner:Current` = `sqrt(100 ** 2 * composite_var / 560968.636)`<br>`U_ChicaneOuter:Current` = `-sqrt(100 ** 2 * composite_var / 560968.636)` |
| `grating_angle_w_DMsurf_TRA` | relative | `U_Grating2Rotation:Position.Axis 1` = `composite_var`<br>`U_TRAServer03:Position.Axis 5` = `composite_var*(7.66)` |

Two observations from this table:

1. **24 of 26 formulas are affine** (`a*x + b`). One `AffineTransform`, reading
   `a` and `b` from the catalog, covers them with an inverse that is one
   division. The remaining two are `±sqrt(100**2 * x / 560968.636)`
   (`R56_at_100MeV`), invertible in a line — and note its two targets are
   exact negatives of each other, so it is one function with a sign.
2. **11 of 13 have a target whose formula is the identity** — `composite_var`
   with no coefficient, or `* 1` / `* 1.0`. That target *is* the pseudo value.
   This matters for §3 below: the corpus mostly names its own reference
   component, so the inverse is usually not a modelling choice at all. The two
   that do not are `Laser_angle_offset_x` (`*1.5` / `*-1.5`) and
   `R56_at_100MeV`.

### YAML stays, but only for what it is good at

Sam, 2026-09-13: *"we should think about the bluesky preferred path for
defining pseudos... there is literally no reason to stick to [yamls]."*

The shape to aim at splits the two jobs the YAML currently does:

- **Keep in the catalog:** which pseudos exist, their targets, their
  coefficients, their units. Operators edit this without a deploy, which is
  the whole reason it is config.
- **Move to Python:** the maths, as a `Transform` subclass.
  `AffineTransform` reading the catalog's coefficients covers 24 of 26
  formulas, so almost nothing needs a bespoke class; `R56_at_100MeV` gets one.

This also retires `forward_expr.py` (the restricted-expression compiler) for
the affine majority, which is a net deletion — a small expression evaluator
exists today only because the maths had nowhere else to live.

Note a longer-term direction that does **not** change this: Sam has discussed
with the other controls person eventually migrating all controls YAMLs to the
database. That is a storage change, not a maths change, and this split
survives it.

---

## 3. The open question, which is Sam's and not guessable

**Reading a pseudo back is over-determined.** `ALine_e_beam_angle_offset_x`
drives `U_S3H` at ×1 and `U_S4H` at ×−2: two raw numbers, one derived number.
`raw_to_derived` must answer two things:

1. **Which raw value defines the derived one?** For the 11 entries with an
   identity target the obvious answer is "that one" — a reference component
   declared per pseudo. For `Laser_angle_offset_x` and `R56_at_100MeV` it has
   to be chosen.
2. **What happens when the components have drifted out of the formula's
   relation?** Someone moved `U_S4H` by hand; the two targets no longer agree
   on any single pseudo value. Options: take the reference component and
   ignore the rest; report the disagreement as a warning and proceed;
   least-squares over all components; or refuse to read. Each is defensible
   and they behave very differently during a scan.

A reasonable default to put to Sam — **declare a reference component per
pseudo (defaulting to the identity target where one exists), invert through
it, and warn when another component differs from what `forward` predicts by
more than a tolerance** — but this is a physics call, per pseudo, and should
be confirmed rather than assumed.

---

## 4. Also in scope, from `01_device_namespace.md`

These were deferred alongside the pseudo work and belong with it, not with
calibration:

- **The catalog's `kind: motor` opt-in.** Today `GeecsNamespace._movable`
  decides motor-vs-settable **purely from a positive DB tolerance** and
  ignores what the scan-variable catalog says. That is a correctness question
  of its own: a catalog entry declaring `kind: motor` on a target whose DB
  tolerance is 0 silently gets a plain setpoint with no convergence
  confirmation.
- **`confirm` entries as namespace nouns.**
- **The axis expansion.**

---

## 5. Suggested order

1. **Decide §3 with Sam** before writing code — it shapes the interface.
2. `AffineTransform` + the two bespoke ones, as pure units with the corpus
   values as fixtures. No hardware, no namespace.
3. Pseudos as namespace nouns; `presets.py` stops refusing them.
4. Delete what the inverse makes redundant — probably the `relative` mode,
   `_capture_baselines`, `restore_baselines_plan`, and most of
   `forward_expr.py`. The repo rule is that the modernising PR deletes the old
   path in the same change.
5. `kind: motor` opt-in.
6. **Hardware acceptance:** a pseudo scan on the native path. None has ever
   run — the 2b acceptance (2026-09-12) exercised only direct settables
   (`U_S1H`, `U_CompAerotech`, `U_ModeImagerESP`). Include a `rel_scan` over a
   pseudo, which is the case #855 was filed about.

**A warning from the 2b acceptance, worth carrying:** a restore script must
use the pre-scan values read at the top, never a hard-coded zero. `U_S4H`
(−0.099 A) was moved to 0 by a restore after a refused scan and had to be put
back by hand. Pseudo scans move magnets in pairs, so this costs double.
