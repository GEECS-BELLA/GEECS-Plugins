# Pseudo scan variables on the native path: the design brief

**Status:** building (arc issue #904). Split out of phase 3 on 2026-09-13
(`03` §8, "The pseudo arc"). The open question in §3 was **ruled on by Sam on
2026-09-14** (see §3); a session can start on §5 without re-deriving it. This
document exists so the session that builds it starts from the survey below
rather than repeating it. **Build log:** §5 step 2 landed 2026-09-15
(GeecsBluesky 0.89.0: `CaSettable.offset`, `CaPseudoPositioner` +
`build_pseudo`, `affine_coefficients`/`compile_inverse`; `CaPseudoMovable`
deleted). One deviation from §2's sketch, on purpose: `forward_expr.py` is
*not* retired — the catalog keeps its `forward` expressions (§6's YAML), so
the compiler stays and the software reads `a`/`b` off the expression's AST
instead of the catalog carrying coefficients. Step 3 is PR #913 (0.90.0:
`GeecsNamespace.add_pseudos`, the startup profile calls it, the preset
expansion stops refusing `kind: pseudo`; the web scanner picker flips with
it, #879). Step 5 (`kind: motor` opt-in) is the PR after it (0.91.0).

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

### `mode: relative` is NOT a workaround for the missing inverse — it stays

An earlier draft of this brief claimed that with a real readback, "relative"
is just `rel_scan` over the pseudo, and that the baseline capture and the
custom restore could be deleted. **That is wrong for the steering bumps, and
the bumps are 11 of the 13 corpus entries.** Checked 2026-09-14:

The x angle bump drives `U_S3H` at ×1 and `U_S4H` at ×−2. The *absolute*
currents on those magnets are set by beam alignment, which changes day to
day, and are essentially never in the formula's ratio. If the pseudo's
readback were inverted through `U_S3H` alone and a `rel_scan` took its first
step, `U_S4H` would be **snapped onto the formula's absolute prediction**
instead of being offset from where it was — a large unintended magnet move,
the same incident class as the `U_S4H` restore-to-zero in the 2b acceptance
(§5), doubled because pseudos move pairs.

So a relative pseudo is not a position at all. It is a **deviation from a
captured baseline**: the physical meaning is "bump about today's alignment".
Sam, 2026-09-14: *"we need to be able to manually adjust alignment and then
once things are aligned (which can change day to day), we need the relative
moves about that set point to be the pseudo variable."*

What survives, and what the migration changes:

- **Baseline capture at scan start stays** (today's `_capture_baselines` at
  `stage()`, lazily on first set for unstaged callers). Its timing is ours;
  ophyd-async has no "capture" concept.
- **The baselines become transform parameters.** ophyd-async's `Transform`
  declares parameters as fields fed live from signals or constants; a soft
  signal per component holding its captured baseline is the native hook. One
  transform class then covers both modes — relative wires the baseline
  signals in, absolute wires zeros — and the `absolute`/`relative` branch in
  `_target_values` goes.
- **The readback becomes real:** `raw_to_derived` = the class's inverse over
  (each component's readback − its offset). Before a scan every deviation
  is zero, so the readback is 0.0 by construction, and `locate()` is defined.
- **The exact restore survives for free.** Every relative formula in the
  corpus is pure linear (`a*x`, no constant term — see the table), so
  `set(0)` through the inverse puts the captured baselines back exactly.
  `restore_baselines_plan`'s formula-independent restore (owner request
  2026-07-22) therefore survives as `unstage()` putting each component
  back at its captured dial baseline (formula-independent, on success and
  abort) — **and `f(0) = 0` is pinned at build time** for any relative
  entry, so `set(0)` is the same restore (the recovery gesture after a
  failed or skipped one) and a future `a*x + b` relative formula is refused
  rather than restored wrong. *(As built, 2026-09-15.)*

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

**Geometry stays hidden in the coefficients — deferred, by design.** The
bump coefficients (×−1 position, ×−2 angle) are the result of lever-arm
arithmetic Sam did once, plus an implicit assumption that `U_S3H` and
`U_S4H` kick the same angle per ampere. Exposing the geometry (magnet and
target-plane positions along the beamline + kick per ampere per magnet,
coefficients derived) would make one entry yield both bumps and make the
arithmetic auditable — but it is a physics-modelling step needing a magnet
calibration, nobody is asking for it, and a geometry-derived transform is
just a generator for the same affine coefficients, so it can be added later
without touching this arc. **Do now, at zero cost:** give each bump entry a
`description` field stating the drift lengths and the equal-kick assumption
that produced its coefficients (Sam, 2026-09-14: "It's just a matter of
geometry").

Note a longer-term direction that does **not** change this: Sam has discussed
with the other controls person eventually migrating all controls YAMLs to the
database. That is a storage change, not a maths change, and this split
survives it.

---

## 3. The over-determined readback — RULED 2026-09-14

Reading a pseudo back is over-determined: two raw numbers, one derived number.
The rulings below came out of the 2026-09-14 discussion with Sam; the items
marked *default* were not put to him explicitly and can be revisited cheaply
during the build, the others are his.

**Vocabulary: use the frameworks' words, not invented ones** (Sam, 2026-09-14:
"I don't want to invent new terms"). There is no single native concept for the
steering bump, but it decomposes into two that are native:

- A **pseudo positioner** (ophyd `PseudoPositioner`, ophyd-async `Transform`):
  a bidirectional function of its components' positions, always with an
  inverse. `R56_at_100MeV` is the textbook case. "Absolute pseudo" is just
  this; do not use the word "absolute" in code or docs.
- A **user offset on each component** (the EPICS motor record's user/dial
  coordinates and `.OFF` field; ophyd `set_current_position`; ophyd-async
  `Motor.offset`): "define where I am now as zero". A steering bump is an
  ordinary pseudo positioner whose components are read in a user coordinate
  that was zeroed at alignment. GEECS devices have no offset field (the
  gateway serves raw readback + `:SP`), so the offset lives in software: a
  soft `offset` signal per component, the raw GEECS value being its dial.
  The catalog's `mode: relative` keeps its YAML spelling (no corpus rewrite)
  and means "zero the components' user offsets at scan start".

What no framework provides and stays ours: *when* zeroing happens (scan
start, as legacy), and the disagreement check (a pseudo positioner just runs
whatever inverse you wrote).

**Precedent, and a follow-on this arc does not take on.** Sam (2026-09-14):
stages with absolute encoders and power supplies in amperes both have a
global coordinate, so an alignment reference belongs on both, and "if that
doesn't exist in the EPICS or Bluesky world maybe we need it in ours — I just
want to make sure I'm not reinventing things." He is not: **spec** gave every
motor a user offset regardless of hardware, and **Sardana** (Tango) gives
every motor `Offset`/`Sign` at the pool level and builds pseudo motors on top.
EPICS confined the idea to the motor record; bluesky/ophyd never added a
generic layer. A soft `offset` signal on *every* namespace component (the
Sardana model from ophyd-async parts, with an operator-facing "set current
position as zero", persistence across restarts, and what the scanner shows)
is a follow-on arc. Sam's conceptual model (2026-09-14): *an alignment
offset, applying to mirror mounts, magnets, or any other component.* **So
this arc builds the offset as the primitive and consumes it:** the soft
`offset` signal lives on the component (every `CaSettable`/`CaMotor` carries
one, dial = raw GEECS value), and the pseudo transform reads its components'
offsets as parameters. This arc still captures them automatically at scan
start and adds nothing operator-facing, so its scope barely moves; the
follow-on becomes purely additive (set-as-aligned, persistence, display)
instead of a relocation.

**Two meanings of "relative", orthogonal — keep them apart** (Sam's check,
2026-09-14: "R56 can never be scanned relatively whereas a steering bump can
only be scanned relatively"):

- *Relative as a scan choice* — the range is about the current readback
  (`rel_scan`) or absolute (`scan`). A plan-level, transient property; applies
  to any movable with a readback, single PV, R56 or bump.
- *Relative as a definition* — the variable's value is a deviation from an
  aligned state and has no absolute meaning. The catalog's `mode: relative`;
  a property of the variable.

| | single PV | `R56_at_100MeV` | steering bump |
|---|---|---|---|
| value has absolute meaning | yes (A, mm) | yes | no — deviation from the aligned currents |
| `rel_scan` (about current) | fine | fine **once the inverse exists** (this arc) | fine but trivial: current = 0 by definition |
| `scan` (absolute range) | fine | fine | same scan as above, since 0 = aligned |

So R56 *can* be scanned relatively in the scan-choice sense (read the current
R56 from the chicane currents, scan around it, put it back); what it lacks
is the relative *definition*. The bump is the mirror image. In offset terms:
a bump is a pseudo positioner over components in their **user** frame, R56
one over components in their **dial** frame. A single PV today has only the
scan-choice kind; the offset follow-on gives it the definition kind too.

**The two cases behave differently, and that is the whole answer.**

| | `mode: relative` (components zeroed at scan start) | plain pseudo positioner |
|---|---|---|
| corpus | 11 entries, all `a*x` | 4 entries (`CompressorAndMode`, `HexZ_S1V`, `JetZ_with_probe`, `R56_at_100MeV`) |
| readback | (reference − its baseline) / a | inverse of the reference component's live readback |
| before a scan | all deviations are 0 → readback 0.0, **components cannot disagree** | other components off the formula is *normal* (someone moved the mode imager by hand); readback = the inverse over the live readbacks; **log a warning** at locate naming each component and its discrepancy |
| first step | every component moves by its own offset from baseline | every component snaps onto the formula (today's absolute-mode behaviour, unchanged) |
| component disagrees mid-scan | **fail the scan** (Sam: a hand move during a bump scan is not a use case; a silent partial move on paired steering magnets is the incident nobody wants) | fail the scan, same rule |
| end of scan | `set(0)` restores the baselines exactly (§2) | ends at last position, as today |

**No reference component — the inverse is a formula over all components**
(Sam, 2026-09-14, on R56: "there is no primary component. The R56 is the
primary component ... there's a concept that exists and then you map that
concept to physical things"). A pseudo positioner's inverse is a function of
*all* its components — that is what ophyd provides — so each transform class
defines its own inverse and **there is no `reference:` catalog field**. For
`R56_at_100MeV` the inverse is the formula solved for the value (from the
inner magnet, sign-checked against the outer). For the affine class the
inverse reads the identity target where one exists (11 of 13) and the first
target otherwise. Nobody has to think about which, because the
**disagreement check carries the weight**: run the inverse, run `forward` on
the result, compare with what the components actually read. When they agree
every choice gives the same value; when they disagree the scan fails or
warns (table above), so the choice only affects the number in the message.
Earlier draft of this ruling named a per-pseudo reference component; retired.

**Disagreement tolerance** *(default)*: the component's DB tolerance where one
is set (the same column the motor-vs-settable rule reads); where the DB says
0 — the magnets #780 is about — a small fixed fraction of the scan step.
Refine during the build against real readback noise on `U_S3H`/`U_S4H`.

**Rejected:** least-squares over all components. With a 1:−2 ratio it cannot
say which magnet moved, and it invents a "position" nobody commanded.

**Rejected:** dropping relative mode in favour of `rel_scan` alone (§2).

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

1. §3 is decided — read it, do not re-open it.
2. The per-component `offset` soft signal on `CaSettable`/`CaMotor` (§3 —
   the primitive), then `AffineTransform` + the two bespoke ones reading
   component offsets as parameters, as pure units with the corpus values as
   fixtures. No hardware, no namespace.
3. Pseudos as namespace nouns; `presets.py` stops refusing them.
4. Delete what the inverse makes redundant: the `absolute`/`relative` branch
   in `_target_values`, `restore_baselines_plan` (the restore moved into
   `unstage()`, with `f(0) = 0` pinned at build — §2). **Not** the baseline
   capture — it stays as transform parameters — and, as built, **not**
   `forward_expr.py`: the catalog keeps its `forward` expressions (§6), so
   the compiler stays and grows the affine walk + the inverse compiler.
   The repo rule is that the modernising PR deletes the old path in the
   same change. *(Done in step 2, 2026-09-15.)*
5. `kind: motor` opt-in.
6. **Hardware acceptance:** a pseudo scan on the native path. None has ever
   run — the 2b acceptance (2026-09-12) exercised only direct settables
   (`U_S1H`, `U_CompAerotech`, `U_ModeImagerESP`). Include a `rel_scan` over a
   pseudo, which is the case #855 was filed about.

**A warning from the 2b acceptance, worth carrying:** a restore script must
use the pre-scan values read at the top, never a hard-coded zero. `U_S4H`
(−0.099 A) was moved to 0 by a restore after a refused scan and had to be put
back by hand. Pseudo scans move magnets in pairs, so this costs double.

---

## 6. Sketch (pseudocode, 2026-09-15 — shape, not exact API)

Physicists write the relations; the software inverts linear ones itself and
takes a physicist-supplied `inverse` for the rest (the schema field exists
already, unconsumed until now).

```yaml
ALine_e_beam_angle_offset_x:
  kind: pseudo
  mode: relative            # components are zeroed at scan start
  description: >
    Angle bump at the undulator entrance. S3H→S4H and S4H→target drifts
    give the −2; assumes equal kick per ampere on S3H and S4H.
  targets:
    - {target: U_S3H:Current, forward: x * 1}
    - {target: U_S4H:Current, forward: x * -2}
  # no inverse: linear, the software inverts it

R56_at_100MeV:
  kind: pseudo
  mode: absolute            # a plain pseudo positioner, no zeroing
  targets:
    - {target: U_ChicaneInner:Current, forward: sqrt(100**2 * x / 560968.636)}
    - {target: U_ChicaneOuter:Current, forward: -sqrt(100**2 * x / 560968.636)}
  inverse: 560968.636 * U_ChicaneInner**2 / 100**2   # physicist-supplied
```

```python
class CaSettable(StandardReadable):           # every component carries an offset
    offset: SignalR[float]                    # soft, default 0.0; dial = raw GEECS value
    async def zero_here(self): self._set_offset(await self.readback.get_value())

class AffineTransform(Transform):             # every 'a*x + b' entry, inverse derived
    a: list[float]; b: list[float]; offsets: list[float]   # offsets fed live from the signals
    def derived_to_raw(self, x): return {n: off + a*x + b for ...}
    def raw_to_derived(self, **raw):
        i = identity_index_or_first
        return (raw[names[i]] - offsets[i] - b[i]) / a[i]

class ExpressionTransform(Transform):         # entries giving forward AND inverse (R56)
    def derived_to_raw(self, x): return {n: f(x) for n, f in zip(names, forward)}
    def raw_to_derived(self, **raw): return inverse(**raw)

def build_pseudo(entry, ns) -> SignalRW:      # one factory call → a plain movable signal
    factory = DerivedSignalFactory(transform_cls, set_derived=move_all,
                                   **{n: c.readback ...}, **{f"offset_{n}": c.offset ...})
    return factory.derived_signal_rw(float, entry.name)

async def check_agreement(pseudo, comps, tol):  # same for both cases, inverse-agnostic
    raw = {n: await c.readback.get_value() ...}
    predicted = derived_to_raw(raw_to_derived(**raw))
    bad = {n for n in raw if abs(raw[n] - predicted[n]) > tol[n]}
    if bad and entry.mode == "relative": raise ScanAborted(...)   # fail
    if bad: log.warning(...)                                        # plain pseudo: warn

# bump: stage zeroes the components (readback 0 by construction), scan, unstage restores
# — on success and abort alike (July ruling; the one path that protects the magnets);
# a failed/skipped restore makes the next stage refuse until `mv <pseudo> 0` (as built)
# R56: an ordinary movable — bp.scan and bp.rel_scan both work, rel_scan via the inverse
```

**`scan` vs `rel_scan` over a bump coincide** (Sam's check, 2026-09-15): the
bump's readback is 0 at stage, so rel_scan's "read current + range" and its
end-of-scan put-back produce exactly the moves and restore a plain scan +
`set(0)` does. The preflight neither refuses nor special-cases it. The
plan-level relative concept earns its keep on the *other* kind: rel_scan over
R56 or a single magnet. **Edge for the acceptance:** zeroing happens at every
stage, so an aborted bump scan whose restore did not run would be re-zeroed
*with the leftover bump baked in* — abort a bump scan mid-way on hardware and
confirm the magnets return before anything else is staged. Restore-on-abort
for *plain* pseudos (R56) is optional and is just rel_scan's own put-back.
