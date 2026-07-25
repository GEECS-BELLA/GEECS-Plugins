# GeecsPvaGateway — Windows camera server deployment

One instance per camera server, serving that host's cameras. Everything below
was exercised on the pilot box (Win10 22H2, Python 3.11) on 2026-07-25.

## One-time bootstrap (per box)

1. **Python 3.11** (python.org installer is fine; the service gets its own
   venv, independent of anything LabVIEW uses).

2. **Install the package** (from a clone or a built wheel):

   ```bat
   py -3.11 -m venv C:\geecs\pva-gateway\venv
   C:\geecs\pva-gateway\venv\Scripts\pip install <wheel-or-source-path>
   ```

3. **GEECS DB credentials — the session-0 rule.** Services cannot see per-user
   mapped drives (`Z:` does not exist in session 0), so the config chain must
   resolve locally:
   - `%USERPROFILE%\.config\geecs_python_api\config.ini` with a `[Paths]
     geecs_data` pointing at a **local** directory that contains
     `user data\Configurations.INI` (copy it from the share once), or a UNC
     path readable by the service account.
   - Symptom of getting this wrong: the process hangs forever at import in a
     headless session (a tkinter file dialog nobody can see) or fails DB
     resolution.

4. **Firewall** (inbound, one-time):

   ```powershell
   New-NetFirewallRule -Name pva-server-tcp -DisplayName 'PVA Server TCP 5075' -Direction Inbound -Protocol TCP -Action Allow -LocalPort 5075
   New-NetFirewallRule -Name pva-server-udp -DisplayName 'PVA Search UDP 5076' -Direction Inbound -Protocol UDP -Action Allow -LocalPort 5076
   ```

5. **NSSM service** (lifecycle belongs to the service manager — Windows never
   kills orphaned processes, and a quick restart can hit TIME_WAIT on 5075, so
   let NSSM own restarts):

   ```bat
   nssm install GeecsPvaGateway C:\geecs\pva-gateway\venv\Scripts\geecs-pva-gateway.exe --experiment Undulator
   nssm set GeecsPvaGateway AppStdout C:\geecs\pva-gateway\logs\service.log
   nssm set GeecsPvaGateway AppStderr C:\geecs\pva-gateway\logs\service.log
   nssm set GeecsPvaGateway AppExit Default Restart
   nssm start GeecsPvaGateway
   ```

   The service is a tenant beside the LabVIEW device apps — leave priority
   default (the gateway idles unless someone is watching).

## Smoke test

On the box (or anywhere on the lab network):

```bat
C:\geecs\pva-gateway\venv\Scripts\geecs-pva-gateway.exe --experiment Undulator --list
```

prints the served PV names. From any machine with p4p (VPN: pass the server
IP so name search unicasts):

```python
from p4p.client.thread import Context
img = Context("pva").get("undulator:uc_amp2_ir_input:image", timeout=5)
print(img.shape, img.dtype)
```

First read after idle takes one gating round-trip (subscribe + next device
push, ~1–2 s at 1 Hz) — that is the unwatched-cameras-are-free trade.

## Fleet notes

- Rollout shape (endorsed, built at ladder rung C): versioned wheel on the
  data share + a pinned current-version file; services pip-install it at
  startup, so restart ⇒ current. Canary one box, watch its
  `{experiment}:pvagateway:{host}:version`/`heartbeat` PVs, then roll the rest.
- Per-instance identity PVs make version skew visible on one Phoebus table.
- SSH is for bootstrap/debugging only; keep sshd `Manual`+stopped otherwise.
