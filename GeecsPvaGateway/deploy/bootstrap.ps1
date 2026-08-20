# One-time GeecsPvaGateway bootstrap for a Windows camera server.
# Prereqs: internet (pip, nssm.cc, and — when `py -3.11` is missing —
# python.org for a silent all-users Python 3.11 install).
# Run from an elevated PowerShell inside the repo checkout:
#   powershell -ExecutionPolicy Bypass -File .\GeecsPvaGateway\deploy\bootstrap.ps1 `
#       -Experiment Undulator -Source .\GeecsPvaGateway
# -Source accepts the package source dir (path deps resolve inside a
# checkout; monorepo wheels do not work — unresolvable path metadata).
# -SourceShare (optional) enables pull-on-restart from the lab's shared
# GEECS-Plugins clone (UNC path, machine-account readable):
#   -SourceShare "\\fileserver\software\...\Active Version\GEECS-Plugins"
param(
    [Parameter(Mandatory = $true)][string]$Experiment,
    [Parameter(Mandatory = $true)][string]$Source,
    [string]$Root = "C:\geecs\pva-gateway",
    [string]$SourceShare = "",
    # Path to a readable Configurations.INI (UNC share path from a console
    # session — mapped drives are invisible here — or a local copy when
    # driving over SSH). Copied into the service profile so the box is
    # start-ready after bootstrap.
    [string]$ConfigSource = ""
)
$ErrorActionPreference = "Stop"

function Assert-Native([string]$What) {
    # PS 5.1's EAP=Stop ignores native exit codes; check them explicitly.
    if ($LASTEXITCODE) { throw "$What failed (exit $LASTEXITCODE)" }
}

# Validate inputs before any destructive step (the service is removed below).
if ($ConfigSource -and -not (Test-Path $ConfigSource -PathType Leaf)) {
    throw "-ConfigSource is not a readable file: $ConfigSource (mapped drives " +
    "are not visible over SSH or in an elevated session - use a UNC path or " +
    "a local copy)"
}

# Stop/remove any existing service FIRST, so pip never upgrades files a
# running service holds open. Guarded lookup: bare `nssm stop` on a fresh box
# writes stderr, which PS 5.1 + EAP=Stop turns into a terminating error.
$nssm = "$Root\nssm.exe"
if (Get-Service GeecsPvaGateway -ErrorAction SilentlyContinue) {
    & $nssm stop GeecsPvaGateway | Out-Null
    & $nssm remove GeecsPvaGateway confirm | Out-Null
}

# Layout. The service runs as LocalSystem with USERPROFILE overridden to
# $Root\profile, so the GEECS config chain resolves without mapped drives or
# service-account passwords (session-0 rule — see DEPLOYMENT.md).
foreach ($dir in @($Root, "$Root\logs",
        "$Root\profile\.config\geecs_python_api", "$Root\profile\user data")) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# config.ini is fully determined by $Root — generate it. The operator supplies
# only Configurations.INI (DB credentials) into profile\user data\.
$configIni = "$Root\profile\.config\geecs_python_api\config.ini"
if (-not (Test-Path $configIni)) {
    Set-Content -Path $configIni -Value @(
        "[Paths]",
        "geecs_data = $Root\profile\user data",
        "",
        "[Experiment]",
        "expt = $Experiment"
    )
}
if ($ConfigSource) {
    Copy-Item $ConfigSource "$Root\profile\user data\Configurations.INI" -Force
}

# Python 3.11 — installed silently if absent, so a bare box needs no manual
# prep. 3.11.9 is the last 3.11 release with a binary installer (later 3.11.x
# are source-only security releases). All-users install (the service runs as
# LocalSystem); the py launcher lands in C:\Windows, on PATH immediately.
$py311 = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    # cmd swallows the probe's stderr: PS 5.1 + EAP=Stop turns redirected
    # native stderr into a terminating error (same quirk as nssm above).
    $null = & cmd /c "py -3.11 -c pass 2>nul"
    $py311 = -not $LASTEXITCODE
}
if (-not $py311) {
    Write-Host "Python 3.11 not found - installing from python.org ..."
    $pyInstaller = "$env:TEMP\python-3.11.9-amd64.exe"
    # -f: fail on HTTP errors — otherwise a 404/proxy page downloads "fine"
    # and the failure surfaces later as a corrupt installer with no hint why.
    curl.exe -L -s -f -o $pyInstaller `
        https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    Assert-Native "python installer download"
    $proc = Start-Process $pyInstaller -Wait -PassThru -ArgumentList `
        "/quiet InstallAllUsers=1 InstallLauncherAllUsers=1 PrependPath=0 Include_test=0"
    # 3010 = success, reboot required (fine: nothing below needs the reboot)
    if ($proc.ExitCode -notin 0, 3010) {
        throw "python silent install failed (exit $($proc.ExitCode))"
    }
    & py -3.11 -c "pass"
    Assert-Native "python 3.11 verification after install"
}

# Python env + package
if (-not (Test-Path "$Root\venv")) {
    py -3.11 -m venv "$Root\venv"
    Assert-Native "venv creation (is Python 3.11 installed? 'py -3.11')"
}
& "$Root\venv\Scripts\python" -m pip install --quiet --upgrade pip
Assert-Native "pip self-upgrade"
# poetry-core lets pull-on-restart build from the source clone with
# --no-build-isolation (no internet needed at restart time).
& "$Root\venv\Scripts\python" -m pip install --quiet poetry-core
Assert-Native "poetry-core install"
# --no-cache-dir: pip's wheel cache can serve a stale same-URL build of a
# monorepo path dep (bit the canary: cached morning build shadowed a newer
# same-day version).
& "$Root\venv\Scripts\python" -m pip install --quiet --no-cache-dir $Source
Assert-Native "package install from $Source"
& "$Root\venv\Scripts\geecs-pva-gateway.exe" --version
Assert-Native "installed-entrypoint check"

# Launcher (pull-on-restart logic; note: copied once here, NOT part of the
# pull-on-restart loop — a launch.bat change needs re-running this bootstrap)
Copy-Item "$PSScriptRoot\launch.bat" "$Root\launch.bat" -Force

# Firewall (idempotent): PVA server TCP + search UDP
foreach ($rule in @(
        @{Name = "pva-server-tcp"; Display = "PVA Server TCP 5075"; Proto = "TCP"; Port = 5075 },
        @{Name = "pva-server-udp"; Display = "PVA Search UDP 5076"; Proto = "UDP"; Port = 5076 })) {
    if (-not (Get-NetFirewallRule -Name $rule.Name -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -Name $rule.Name -DisplayName $rule.Display `
            -Direction Inbound -Protocol $rule.Proto -Action Allow `
            -LocalPort $rule.Port | Out-Null
    }
}

# NSSM (single exe; fetched once)
if (-not (Test-Path $nssm)) {
    curl.exe -L -s -f -o "$env:TEMP\nssm.zip" https://nssm.cc/release/nssm-2.24.zip
    Assert-Native "nssm download"
    Expand-Archive "$env:TEMP\nssm.zip" -DestinationPath "$env:TEMP\nssm" -Force
    Copy-Item "$env:TEMP\nssm\nssm-2.24\win64\nssm.exe" $nssm
}

# Service: launch.bat under LocalSystem, restart on any exit (the :restart PV
# exits 86; NSSM relaunches -> DB re-resolve + reinstall from the source clone).
& $nssm install GeecsPvaGateway "$Root\launch.bat"
Assert-Native "nssm install"
& $nssm set GeecsPvaGateway AppDirectory $Root
& $nssm set GeecsPvaGateway AppEnvironmentExtra `
    "USERPROFILE=$Root\profile" `
    "GEECS_PVA_ROOT=$Root" `
    "GEECS_PVA_EXPERIMENT=$Experiment" `
    "GEECS_PVA_SOURCE=$SourceShare"
& $nssm set GeecsPvaGateway AppStdout "$Root\logs\service.log"
& $nssm set GeecsPvaGateway AppStderr "$Root\logs\service.log"
& $nssm set GeecsPvaGateway AppRotateFiles 1
& $nssm set GeecsPvaGateway AppRotateOnline 1
& $nssm set GeecsPvaGateway AppRotateBytes 10485760
& $nssm set GeecsPvaGateway AppExit Default Restart
& $nssm set GeecsPvaGateway AppThrottle 5000
& $nssm set GeecsPvaGateway Start SERVICE_AUTO_START

Write-Host ""
if (Test-Path "$Root\profile\user data\Configurations.INI" -PathType Leaf) {
    Write-Host "Bootstrap done (config.ini generated, Configurations.INI in place)."
    Write-Host "Start with:  $nssm start GeecsPvaGateway"
} else {
    Write-Host "Bootstrap done. Before starting, place the DB credentials file at:"
    Write-Host "  $Root\profile\user data\Configurations.INI"
    Write-Host "(or re-run with -ConfigSource <path>).  Then:  $nssm start GeecsPvaGateway"
}
