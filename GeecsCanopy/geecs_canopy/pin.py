"""The Canopy build this package runs in: the fleet pin, bumped only in a reviewed PR.

The one record of it. ``deploy/render_conf.sh`` reads ``CANOPY_IMAGE`` for
the derived image's base; a startup self-test compares ``CANOPY_COMMIT``
with the installed ``canopy`` distribution's version
(``0.0.2.devN+g<commit>…``). Never ``latest``: Canopy has no releases and
its tags are short SHAs of ``main``.
"""

CANOPY_COMMIT = "0a5ee23"
CANOPY_IMAGE = "ghcr.io/slaclab/canopy@sha256:219da36a98713bf524cd623d1364ead9eef95016fc474e15ab27fcc62d4d8abd"
