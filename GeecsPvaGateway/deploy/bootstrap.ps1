# One-time GeecsPvaGateway bootstrap for a Windows camera server.
# Prereqs: Python 3.11 (`py -3.11` must work), internet for pip + nssm.cc.
# Run from an elevated PowerShell inside the repo checkout:
#   powershell -ExecutionPolicy Bypass -File .\GeecsPvaGateway\deploy\bootstrap.ps1 `
#       -Experiment Undulator -Source .\GeecsPvaGateway
# -Source accepts the package source dir (recommended: resolves the monorepo
# path deps) or a wheel. -WheelShare (optional) enables pull-on-restart, e.g.
#   -WheelShare \\fileserver\software\pva-wheels
param(
    [Parameter(Mandatory = $true)][string]$Experiment,
    [Parameter(Mandatory = $true)][string]$Source,
    [string]$Root = "C:\geecs\pva-gateway",
    [string]$WheelShare = "",
    # Path to a readable Configurations.INI (share path from a console session
    # with the drive mapped, or a local copy when driving over SSH). Copied
    # into the service profile so the box is start-ready after bootstrap.
    [string]$ConfigSource = ""
)
$ErrorActionPreference = "Stop"

function Assert-Native([string]$What) {
    # PS 5.1's EAP=Stop ignores native exit codes; check them explicitly.
    if ($LASTEXITCODE) { throw "$What failed (exit $LASTEXITCODE)" }
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
    if (-not (Test-Path $ConfigSource)) {
        throw "-ConfigSource not readable: $ConfigSource (mapped drives are not visible over SSH — use a local copy there)"
    }
    Copy-Item $ConfigSource "$Root\profile\user data\Configurations.INI" -Force
}

# Python env + package
if (-not (Test-Path "$Root\venv")) {
    py -3.11 -m venv "$Root\venv"
    Assert-Native "venv creation (is Python 3.11 installed? 'py -3.11')"
}
& "$Root\venv\Scripts\python" -m pip install --quiet --upgrade pip
Assert-Native "pip self-upgrade"
# --no-cache-dir: pip's wheel cache can serve a stale same-URL build of a
# monorepo path dep (bit the canary: cached morning build shadowed a newer
# same-day version).
& "$Root\venv\Scripts\python" -m pip install --quiet --no-cache-dir $Source
Assert-Native "package install from $Source"
& "$Root\venv\Scripts\geecs-pva-gateway.exe" --version
Assert-Native "installed-entrypoint check"

# Launcher (pull-on-restart logic; note: copied once here, NOT part of the
# wheel rollout loop — a launch.bat change needs re-running this bootstrap)
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
    curl.exe -L -s -o "$env:TEMP\nssm.zip" https://nssm.cc/release/nssm-2.24.zip
    Assert-Native "nssm download"
    Expand-Archive "$env:TEMP\nssm.zip" -DestinationPath "$env:TEMP\nssm" -Force
    Copy-Item "$env:TEMP\nssm\nssm-2.24\win64\nssm.exe" $nssm
}

# Service: launch.bat under LocalSystem, restart on any exit (the :restart PV
# exits 86; NSSM relaunches -> DB re-resolve + wheel re-pin).
& $nssm install GeecsPvaGateway "$Root\launch.bat"
Assert-Native "nssm install"
& $nssm set GeecsPvaGateway AppDirectory $Root
& $nssm set GeecsPvaGateway AppEnvironmentExtra `
    "USERPROFILE=$Root\profile" `
    "GEECS_PVA_ROOT=$Root" `
    "GEECS_PVA_EXPERIMENT=$Experiment" `
    "GEECS_PVA_WHEEL_SHARE=$WheelShare"
& $nssm set GeecsPvaGateway AppStdout "$Root\logs\service.log"
& $nssm set GeecsPvaGateway AppStderr "$Root\logs\service.log"
& $nssm set GeecsPvaGateway AppRotateFiles 1
& $nssm set GeecsPvaGateway AppRotateOnline 1
& $nssm set GeecsPvaGateway AppRotateBytes 10485760
& $nssm set GeecsPvaGateway AppExit Default Restart
& $nssm set GeecsPvaGateway AppThrottle 5000
& $nssm set GeecsPvaGateway Start SERVICE_AUTO_START

Write-Host ""
if (Test-Path "$Root\profile\user data\Configurations.INI") {
    Write-Host "Bootstrap done (config.ini generated, Configurations.INI in place)."
    Write-Host "Start with:  $nssm start GeecsPvaGateway"
} else {
    Write-Host "Bootstrap done. Before starting, place the DB credentials file at:"
    Write-Host "  $Root\profile\user data\Configurations.INI"
    Write-Host "(or re-run with -ConfigSource <path>).  Then:  $nssm start GeecsPvaGateway"
}
