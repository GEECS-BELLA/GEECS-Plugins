"""The GEECS plan layer.

The registered plans (:mod:`.registry`) are stock ``bp.count`` and
``scan_nd`` (behind :mod:`.sweep`) with the GEECS ``take_reading`` hooks
bound — strict (:mod:`.on_demand`, one shot-table row per fire, every
per-shot device in its own stream, the native :mod:`.optimize` loop
included; :mod:`.strict` holds the shared shot helpers) or
gated (:mod:`.gated`, the box free-running while the plugin-backed
cameras count) — plus the day-scoped
scan-number claim (:mod:`.claim_scan`), the ActionPlan → plan-stub
compiler (:mod:`geecs_bluesky.actions.compiler`) and the once-run
shot-offset :mod:`.calibration`.
"""

from geecs_bluesky.plans.claim_scan import claim_scan, claim_scan_number
from geecs_bluesky.plans.strict import fire_and_await_shot

__all__ = [
    "claim_scan",
    "claim_scan_number",
    "fire_and_await_shot",
]
