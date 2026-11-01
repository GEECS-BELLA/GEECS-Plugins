# GeecsControlRoom

The operator surfaces over [Canopy](https://github.com/slaclab/canopy) and
the GEECS device module (`GeecsCanopy`): the experiment's home, the
"Control Room", and the screens renderer — a control GUI is a screen
file drawn by one renderer (`Planning/operator_surfaces.md`).

A FastAPI service on the kit (`GeecsWebTheme`), plain JavaScript, no build
step. Live values come from Hare's WebSocket with a read key; every write
goes through the module's put route. `CLAUDE.md` holds the rules.

    GEECS_EXPERIMENT=… GEECS_CONFIGS_ROOT=… \
    GEECS_CONTROL_ROOM_CANOPY_URL=http://127.0.0.1:8600 \
    GEECS_CONTROL_ROOM_READ_KEY=… poetry run geecs-control-room

## Deploy

On the services host, beside Canopy (`GeecsCanopy/DEPLOYMENT.md`), from its
own clone `<GEECS_CHECKOUT_ROOT>/control-room-checkout` (`deploy/bootstrap_host.sh`
clones it, installs the env and renders the unit). The `GEECS_CONTROL_ROOM_*`
keys are in `site.env` (`deploy/site.env.example`); on a host without a front
door, `GEECS_CONTROL_ROOM_BIND=0.0.0.0` and `GEECS_CONTROL_ROOM_CANOPY_BROWSER_URL=http://<host>:<Canopy's port>`
(the browser opens Hare's WebSocket there; a path like `/canopy` would be this
page's own origin). The secret is
the display's read key, the `read@*` entry of Canopy's `CANOPY_API_KEYS`, never one
that carries write:

    sudo install -d -m 0755 /etc/geecs/control-room
    sudo install -m 0600 -o <service account> /dev/null /etc/geecs/control-room/secrets.env
    # then, as root: GEECS_CONTROL_ROOM_READ_KEY=<the display key>
    sudo install -m 0644 <staging>/geecs-control-room.service /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now geecs-control-room

Check: `GET /healthz` → `{"status": "ok"}`. The home and the screens read the
configs share per request; Canopy's containers do not (restart `geecs-canopy`).
