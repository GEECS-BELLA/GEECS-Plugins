@echo off
rem GeecsPvaGateway service launcher: pull-on-restart, then serve.
rem NSSM runs this with USERPROFILE, GEECS_PVA_ROOT, GEECS_PVA_EXPERIMENT and
rem (optionally) GEECS_PVA_SOURCE set (see bootstrap.ps1). GEECS_PVA_SOURCE is
rem the UNC path of the shared GEECS-Plugins clone (the lab's "Active
rem Version" clone): a restart — :restart PV (exit 86), crash, or reboot —
rem reinstalls the five intra-repo packages from it, so the clone's commit
rem IS the fleet pin. Rollout = git pull in the clone + restart PVs; rollback
rem = git checkout <rev> there + restarts. An unreachable share falls through
rem to the installed versions (a restart never bricks an instance).
rem NOTE: LocalSystem authenticates to shares as the MACHINE account — the
rem share must be readable by it or the check silently fails every restart
rem (visible only as version-PV skew; see DEPLOYMENT.md).
rem
rem --no-deps: monorepo path-dep metadata never resolves outside a checkout;
rem external (PyPI) deps are frozen at bootstrap by design — except the pins
rem in deploy/requirements-fleet.txt, installed offline from the share's
rem wheel cache first (see below).
rem --no-build-isolation: build with the venv's poetry-core (installed at
rem bootstrap), so a restart needs no internet.

if "%GEECS_PVA_ROOT%"=="" set GEECS_PVA_ROOT=C:\geecs\pva-gateway

rem The wheel cache beside the share clone (resolved outside the block
rem below: cmd expands %VAR% inside parentheses when the block is parsed).
set "GEECS_PVA_WHEELS="
if not "%GEECS_PVA_SOURCE%"=="" for %%I in ("%GEECS_PVA_SOURCE%\..") do set "GEECS_PVA_WHEELS=%%~fI\pva-wheels"

if not "%GEECS_PVA_SOURCE%"=="" (
    if exist "%GEECS_PVA_SOURCE%\GeecsPvaGateway\pyproject.toml" (
        rem External deps added after bootstrap (deploy/requirements-fleet.txt)
        rem install offline from the wheel cache beside the clone, staged by
        rem deploy/stage_wheels.sh.  --no-deps on both sides: the pin file IS
        rem the closure (transitive deps are pinned explicitly or were frozen
        rem at bootstrap).  Best effort: a missing cache or wheel leaves the
        rem installed versions, exactly like the reinstall below.
        if exist "%GEECS_PVA_WHEELS%" (
            echo pull-on-restart: fleet requirements from "%GEECS_PVA_WHEELS%"
            "%GEECS_PVA_ROOT%\venv\Scripts\python" -m pip install --quiet --no-index --no-deps --find-links "%GEECS_PVA_WHEELS%" -r "%GEECS_PVA_SOURCE%\GeecsPvaGateway\deploy\requirements-fleet.txt"
        )
        echo pull-on-restart: reinstalling from "%GEECS_PVA_SOURCE%"
        "%GEECS_PVA_ROOT%\venv\Scripts\python" -m pip install --quiet --upgrade --no-deps --no-build-isolation "%GEECS_PVA_SOURCE%\GEECS-Schemas" "%GEECS_PVA_SOURCE%\GEECS-Core" "%GEECS_PVA_SOURCE%\GEECS-Data-Utils" "%GEECS_PVA_SOURCE%\GeecsCAGateway" "%GEECS_PVA_SOURCE%\GeecsPvaGateway"
    )
)

"%GEECS_PVA_ROOT%\venv\Scripts\geecs-pva-gateway.exe" --experiment %GEECS_PVA_EXPERIMENT%
exit /b %ERRORLEVEL%
