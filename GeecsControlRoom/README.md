# GeecsControlRoom

The operator surfaces over [Canopy](https://github.com/slaclab/canopy) and
the GEECS device module (`GeecsCanopy`): the experiment's home, the
"Control Room", and the screens renderer — a control GUI is a screen
file drawn by one renderer (`Planning/operator_surfaces.md`).

A FastAPI service on the kit (`GeecsWebTheme`), plain JavaScript, no build
step. Live values come from Hare's WebSocket with a read key; every write
goes through the module's put route. `CLAUDE.md` holds the rules.

    GEECS_EXPERIMENT=… GEECS_CONFIGS_ROOT=… \
    GEECS_CONTROL_ROOM_CANOPY_URL=http://127.0.0.1:8000 \
    GEECS_CONTROL_ROOM_READ_KEY=… poetry run geecs-control-room
