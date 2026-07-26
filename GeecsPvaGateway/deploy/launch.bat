@echo off
rem GeecsPvaGateway service launcher: pull-on-restart, then serve.
rem NSSM runs this with USERPROFILE, GEECS_PVA_ROOT, GEECS_PVA_EXPERIMENT and
rem (optionally) GEECS_PVA_WHEEL_SHARE set (see bootstrap.ps1). A restart —
rem :restart PV (exit 86), crash, or reboot — re-pins to the wheels listed in
rem CURRENT on the share; an unreachable share falls through to the installed
rem versions. NOTE: LocalSystem authenticates to shares as the MACHINE
rem account — the share must be readable by it or the check silently fails
rem every restart (visible only as version-PV skew; see DEPLOYMENT.md).
rem
rem CURRENT lists one wheel filename per line (intra-repo deps first, e.g.
rem geecs_data_utils, geecs_ca_gateway, then geecs_pva_gateway). Wheels are
rem installed together with --no-deps: monorepo wheels carry unresolvable
rem build-machine path metadata, and external (PyPI) deps are frozen at
rem bootstrap time by design.
setlocal enabledelayedexpansion

if "%GEECS_PVA_ROOT%"=="" set GEECS_PVA_ROOT=C:\geecs\pva-gateway

if not "%GEECS_PVA_WHEEL_SHARE%"=="" (
    if exist "%GEECS_PVA_WHEEL_SHARE%\CURRENT" (
        set WHEELS=
        for /f "usebackq delims=" %%w in ("%GEECS_PVA_WHEEL_SHARE%\CURRENT") do (
            set WHEELS=!WHEELS! "%GEECS_PVA_WHEEL_SHARE%\%%w"
        )
        if not "!WHEELS!"=="" (
            echo pull-on-restart: pinning to !WHEELS!
            "%GEECS_PVA_ROOT%\venv\Scripts\python" -m pip install --quiet --upgrade --no-deps !WHEELS!
        )
    )
)

"%GEECS_PVA_ROOT%\venv\Scripts\geecs-pva-gateway.exe" --experiment %GEECS_PVA_EXPERIMENT%
exit /b %ERRORLEVEL%
