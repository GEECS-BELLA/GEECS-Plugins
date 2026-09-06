# Redis Notes for the Queueserver Host

The dedicated Ubuntu 22.04 queueserver host should use the distro Redis
package and its systemd unit:

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable --now redis-server.service
```

Keep Redis bound to loopback only. The Ubuntu package default is suitable for
the RE Manager control-plane shape:

```conf
bind 127.0.0.1 ::1
protected-mode yes
```

The source-built Redis used during sandbox testing was only a no-sudo
workaround. It is not the deployment shape for the service host.

## Never carry a dump.rdb across Redis major versions

Migrating the interim host off its source-built Redis (2026-09-06) failed on
exactly this. The spike binary was Redis 8.10.1, which writes **RDB format
version 15**; the jammy package is 6.0.16, which reads up to version 9. Given
the newer dump it refuses to start:

```
# Can't handle RDB format version 15
# Fatal error loading the DB: Invalid argument. Exiting.
```

Two things make that hard to read. Redis logs it to
`/var/log/redis/redis-server.log`, **not** the journal — `systemctl status`
shows only `status=1/FAILURE` and a restart-throttle, so read that logfile
first for any Redis start failure. And with nothing then answering on 6379,
the launcher's fallback (below) starts an unsupervised Redis instead, which
looks like a working queueserver.

Start the packaged Redis on an **empty** dataset rather than downgrading the
dump. Of the RE Manager's keys only `qs_default_plan_history` holds anything
you cannot recreate: permissions are re-published from
`user_group_permissions.yaml` by the launcher's `--user-group-permissions` on
every start, `qs_default_running_plan` is empty unless a plan is mid-flight,
and the autostart/stop-pending keys are transient flags. Archive the old dump
rather than deleting it — reading it again needs a Redis of the version that
wrote it.

## The launcher starts its own Redis when none answers

`launch_re_manager.sh` runs `${QS_REDIS_SERVER:-redis-server} --bind 127.0.0.1
--port 6379 --daemonize yes` whenever nothing answers on 6379, and
`geecs-qserver.service` is ordered `After=redis-server.service` without
requiring it. So a host that never got the package does not fail loudly — it
silently acquires an **unsupervised** Redis that dies with whatever started
it, is absent after a reboot, and comes back empty. That is how the interim
host ended up running a hand-built 8.10.1 for two weeks.

`deploy/bootstrap_host.sh` now prints the package's root steps when
`redis-server.service` is not enabled, and `scripts/fleet_status.sh` reports a
`Redis` row whose finding is exactly this case: answering on 6379 but not
supervised by the unit. After any Redis work, confirm the unit owns the port:

```bash
systemctl show -p MainPID --value redis-server.service   # must equal
ss -ltnp 'sport = :6379'                                 # the pid here
```

Redis may warn at startup that `vm.overcommit_memory` is disabled. Apply the
host-level sysctl fix once:

```bash
echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-redis-overcommit.conf
sudo sysctl --system
```
