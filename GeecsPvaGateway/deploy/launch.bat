@echo off
rem GeecsPvaGateway service launcher: pull-on-restart, then serve.
rem NSSM runs this with USERPROFILE, GEECS_PVA_ROOT, GEECS_PVA_EXPERIMENT and
rem (optionally) GEECS_PVA_WHEEL_SHARE set (see bootstrap.ps1). A restart —
rem :restart PV (exit 86), crash, or reboot — re-pins to the fleet's current
rem wheel; an unreachable share falls through to the installed version.
setlocal enabledelayedexpansion

if "%GEECS_PVA_ROOT%"=="" set GEECS_PVA_ROOT=C:\geecs\pva-gateway

if not "%GEECS_PVA_WHEEL_SHARE%"=="" (
    if exist "%GEECS_PVA_WHEEL_SHARE%\CURRENT" (
        set /p WHEEL=<"%GEECS_PVA_WHEEL_SHARE%\CURRENT"
        echo pull-on-restart: pinning to !WHEEL!
        "%GEECS_PVA_ROOT%\venv\Scripts\python" -m pip install --quiet "%GEECS_PVA_WHEEL_SHARE%\!WHEEL!"
    )
)

"%GEECS_PVA_ROOT%\venv\Scripts\geecs-pva-gateway.exe" --experiment %GEECS_PVA_EXPERIMENT%
exit /b %ERRORLEVEL%
