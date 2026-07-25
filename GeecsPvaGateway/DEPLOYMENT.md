# GeecsPvaGateway — Windows camera server deployment

One NSSM service per camera server, serving that host's cameras. The pilot box
(Win10 22H2, 192.168.6.100) is the canary; every step below is scripted in
`deploy/`.

## The two session-0 rules (hard-won; violating them hangs or breaks silently)

1. **Services cannot see per-user mapped drives** (`Z:` does not exist in
   session 0), and LocalSystem's `~` is not a user's home. The service
   therefore runs with **`USERPROFILE` overridden** to a service-owned profile
   dir, and the GEECS config chain resolves entirely inside it:
   - `{root}\profile\.config\geecs_python_api\config.ini`
   - `{root}\profile\user data\Configurations.INI` (local copy — DB
     credentials; never referenced via a drive letter)
2. **Windows never kills orphaned processes** — lifecycle belongs to NSSM,
   not to whoever launched something. Never run the service from an SSH
   session and walk away.

## One-time bootstrap (per box, elevated PowerShell, from a repo checkout)

```powershell
.\GeecsPvaGateway\deploy\bootstrap.ps1 -Experiment Undulator -Source .\GeecsPvaGateway
# optional pull-on-restart: add  -WheelShare \\fileserver\software\pva-wheels
```

This creates `C:\geecs\pva-gateway\{venv,profile,logs}`, installs the package,
copies `launch.bat`, opens the PVA firewall ports (TCP 5075 / UDP 5076),
fetches `nssm.exe`, and registers the `GeecsPvaGateway` service (auto-start,
restart on any exit, rotating logs). Then place the two config files in the
profile (see rule 1) and `nssm start GeecsPvaGateway`.

## Rollout (fleet upgrade without touching boxes)

```bash
cd GeecsPvaGateway && poetry build           # dist/geecs_pva_gateway-X.Y.Z-*.whl
# copy the wheel to the share, then point CURRENT at it:
#   \\fileserver\software\pva-wheels\geecs_pva_gateway-X.Y.Z-py3-none-any.whl
#   \\fileserver\software\pva-wheels\CURRENT   (one line: the wheel filename)
```

Then restart instances **via the `:restart` PV** — canary first:

```bash
pvput undulator:pvagateway:192_168_6_100:restart 1
```

The server exits with code 86, NSSM relaunches `launch.bat`, which re-pins to
the `CURRENT` wheel and re-resolves the DB config. Watch the instance's
`version` PV flip on the fleet screen (`deploy/fleet_status.bob` — one row per
host: version, heartbeat, and a confirm-dialog restart button); roll the rest
when the canary soaks clean. An unreachable share falls through to the
installed version — a restart never bricks an instance.

## Smoke test

```bat
C:\geecs\pva-gateway\venv\Scripts\geecs-pva-gateway.exe --experiment Undulator --list
```

prints the host's served PV names (DB-scoped: this box's cameras only). From
any machine with p4p (over VPN, pass the server IP so name search unicasts):

```python
from p4p.client.thread import Context
img = Context("pva").get("undulator:uc_amp2_ir_input:image", timeout=5)
print(img.shape, img.dtype)
```

First read after idle takes one gating round-trip (subscribe + next device
push, ~1–2 s at 1 Hz) — that is the unwatched-variables-are-free trade
(gating is per image variable; an unwatched camera holds zero connections).

## Instance PVs

| PV | Meaning |
|---|---|
| `{exp}:pvagateway:{host}:version` | Installed package version (fleet skew check) |
| `{exp}:pvagateway:{host}:heartbeat` | Counter, +1 per 5 s (liveness) |
| `{exp}:pvagateway:{host}:restart` | Write 1 → clean exit 86 → NSSM relaunch |

`{host}` is the served endpoint IP, normalized (`192.168.6.100` →
`192_168_6_100`).

## Fleet notes

- The service is a tenant beside the LabVIEW device apps; it idles unless
  someone is watching (per-variable gating), so default priority is fine.
- SSH is for bootstrap and debugging only; keep sshd `Manual`+stopped
  otherwise. Quick service restarts can hit TIME_WAIT on 5075 — NSSM's
  5 s throttle rides through it.
