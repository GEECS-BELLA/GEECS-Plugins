"""The PV bridge (``Planning/epics_estate.md`` §2.3): every device's PVs, as family-shaped streams.

One generic :class:`~geecs_panels.bridge.feed.Feed` per device, parameterised
by its family file (GEECS-Schemas' ``FamilyFile``) and its instance data
(``PanelDevices``), with the derived rules each template needs in
:mod:`geecs_panels.bridge.rules`; :mod:`~geecs_panels.bridge.stream`
shares one feed per device among every listener and turns it into
``DeviceFrame``s (:mod:`geecs_panels.contracts`); :mod:`~geecs_panels.bridge.routes`
serves them under ``/api``. The per-kind pages (``/camera``, ``/motor`` …)
keep their own feeds until their templates are deleted; the rules both
judge by (freshness, the state words, the run lock, saturation) are one
copy, in the per-kind modules or here.
"""
