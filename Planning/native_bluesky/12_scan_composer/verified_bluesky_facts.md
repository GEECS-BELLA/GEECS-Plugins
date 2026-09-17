# Verified Bluesky 1.15.1 facts

Historical inspection from the owner’s 2026-09-15 design session.
The stock roster and scanner observations describe the pre-composer implementation.


Each of these was verified against the installed package during the design
conversation. **Do not re-derive them from documentation; they are load-bearing
for §3–§5.**

- **`bluesky.plans.scan_nd(cycler, …)` is the engine under every stock verb.**
  It does the staging, the run bracket, `declare_stream`, the `per_step` calls,
  and writes `motors`, `num_points`, `num_intervals` and the dimension hints.
  Each verb is a thin wrapper that builds a `cycler`, writes
  `plan_pattern`/`plan_args`, and delegates. `scan_nd` was excluded from our
  registry (`EXCLUDED_STOCK_PLANS`) only because a `Cycler` is not JSON — which
  is precisely the gap a serializable `Sweep` fills.
- **`bluesky.plan_patterns` runs server-side with no devices and no hardware**,
  given a stand-in that satisfies Movable **and** Readable (`is_movable`
  requires both):

  ```python
  class Ref:
      def __init__(self, name): self.name = name
      def set(self, value): raise NotImplementedError
      def read(self): return {}
      def describe(self): return {}
  ```

  Verified for `inner_product`, `outer_product` (including `snake_axes`),
  `inner_list_product`, `outer_list_product`, `spiral`, `spiral_fermat`,
  `spiral_square_pattern`. `inner_product` and `inner_list_product` accept bare
  strings as the motor key; `outer_product` sniffs for movables and needs the
  stub. **This is why no scan math is ever ported into JavaScript.**
- `scan` takes `(motor, start, stop)` triplets with a **single trailing
  `num`** — a correlated multi-axis range scan therefore shares one point count
  by construction, and each axis gets its own step. `grid_scan` takes per-axis
  `(motor, start, stop, num)` quadruplets.
- `list_scan` / `rel_list_scan` take `(motor, points)` flat pairs, correlated,
  and **every list must be the same length**: `inner_list_product` builds the
  trajectory by *adding* cyclers, so unequal lengths raise `ValueError` before
  the first move. `list_grid_scan` takes the same pairs as an outer product and
  accepts any lengths.
- `log_scan(detectors, motor, start, stop, num)` passes `start`/`stop` to
  `np.logspace`, so they are **decade exponents**, not values. Single motor
  only (typed signature, not varargs), and there is **no `log_grid_scan`**.
- `x2x_scan` is `relative_inner_product_scan(dets, num, m1, start, stop, m2,
  start/2, stop/2)` — relative, and motor 2 traverses **half** motor 1's range.
- `spiral`, `spiral_fermat`, `spiral_square`, `x2x_scan` and `log_scan` have
  **fully typed signatures**. Only the classic family is vararg. A generic
  signature-driven form therefore fails for the classic family and would work
  for these.
- `POST /api/submit` on the scanner accepts any `Preset` dict
  (`service/models.py:118`); the only server-side plan check is
  `expand_preset`'s "is it a scan verb". **The Start gate is client-side only**
  (`scanner.js` `S.formable`), so a hand-authored preset of any shape is
  submittable today by `curl`.
