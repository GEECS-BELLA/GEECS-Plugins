"""PyQt6 windows over the worker's interlock streams (the ``gui`` extra).

- :mod:`.client` — ``geecs-interlock-client``: the status stream (the
  same text LabVIEW reads), a full window and a compact always-on-top strip.
- :mod:`.server` — ``geecs-interlock-server-gui``: the detail stream
  (conditions, live values, the status stream's clients, the RunEngine).

Both are read-only views: neither sends anything to the worker.  They use
Qt's default style; the only colours set are the state colours (palette
and item brushes, never a style sheet).  Import-light: Qt and
:mod:`geecs_bluesky.interlocks.wire` only, never bluesky.
"""
