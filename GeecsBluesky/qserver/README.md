# GEECS Queueserver Launch Assets

This directory contains the user-level launch mechanics for a local
bluesky-queueserver RE Manager. The startup profile content is intentionally
not here; it lands with the separate plan-preamble task.

## Launch

From this directory:

```bash
./launch_re_manager.sh
```

The launcher expects `start-re-manager` on `PATH`. It checks
`127.0.0.1:6379` and starts a local Redis server when that port is not
answering. The Redis binary can be overridden with `QS_REDIS_SERVER`; the
default is `redis-server`.

```bash
QS_REDIS_SERVER=/path/to/redis-server ./launch_re_manager.sh
```

The startup directory defaults to `./startup` and can be overridden with
`QS_STARTUP_DIR`.

```bash
QS_STARTUP_DIR=/path/to/startup ./launch_re_manager.sh
```

The launcher passes `user_group_permissions.yaml` explicitly. Keep that
argument: without a permissions file, RE Manager may accept startup but reject
queue submissions with `success: False`, and the command-line failure can be
silent.

## Verify

In another terminal:

```bash
qserver status
qserver environment open
```

The placeholder `startup/` profile contains no Python yet, so environment
opening verifies the manager mechanics only. Plan registration and GEECS
preamble setup arrive in the separate startup-profile task.
