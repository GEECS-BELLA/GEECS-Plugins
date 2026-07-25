# One-time GeecsPvaGateway bootstrap for a Windows camera server.
# Run from an elevated PowerShell inside the repo checkout:
#   .\deploy\bootstrap.ps1 -Experiment Undulator -Source ..\GeecsPvaGateway
# -Source accepts a wheel path or the package source dir. -WheelShare
# (optional) enables pull-on-restart from a shared wheel drop, e.g.
#   -WheelShare \\fileserver\software\pva-wheels
param(
    [Parameter(Mandatory = $true)][string]$Experiment,
    [Parameter(Mandatory = $true)][string]$Source,
    [string]$Root = "C:\geecs\pva-gateway",
    [string]$WheelShare = ""
)
$ErrorActionPreference = "Stop"

# Layout. The service runs as LocalSystem with USERPROFILE overridden to
# $Root\profile, so the GEECS config chain resolves without mapped drives or
# service-account passwords (session-0 rule — see DEPLOYMENT.md).
foreach ($dir in @($Root, "$Root\logs",
        "$Root\profile\.config\geecs_python_api", "$Root\profile\user data")) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# Python env + package
if (-not (Test-Path "$Root\venv")) { py -3.11 -m venv "$Root\venv" }
& "$Root\venv\Scripts\python" -m pip install --quiet --upgrade pip
& "$Root\venv\Scripts\python" -m pip install --quiet $Source
& "$Root\venv\Scripts\geecs-pva-gateway.exe" --version

# Launcher (pull-on-restart lives in launch.bat, versioned with the repo)
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
$nssm = "$Root\nssm.exe"
if (-not (Test-Path $nssm)) {
    curl.exe -L -s -o "$env:TEMP\nssm.zip" https://nssm.cc/release/nssm-2.24.zip
    Expand-Archive "$env:TEMP\nssm.zip" -DestinationPath "$env:TEMP\nssm" -Force
    Copy-Item "$env:TEMP\nssm\nssm-2.24\win64\nssm.exe" $nssm
}

# Service: launch.bat under LocalSystem, restart on any exit (the :restart PV
# exits 86; NSSM relaunches -> DB re-resolve + wheel re-pin).
& $nssm stop GeecsPvaGateway 2>$null | Out-Null
& $nssm remove GeecsPvaGateway confirm 2>$null | Out-Null
& $nssm install GeecsPvaGateway "$Root\launch.bat"
& $nssm set GeecsPvaGateway AppDirectory $Root
& $nssm set GeecsPvaGateway AppEnvironmentExtra `
    "USERPROFILE=$Root\profile" `
    "GEECS_PVA_ROOT=$Root" `
    "GEECS_PVA_EXPERIMENT=$Experiment" `
    "GEECS_PVA_WHEEL_SHARE=$WheelShare"
& $nssm set GeecsPvaGateway AppStdout "$Root\logs\service.log"
& $nssm set GeecsPvaGateway AppStderr "$Root\logs\service.log"
& $nssm set GeecsPvaGateway AppRotateFiles 1
& $nssm set GeecsPvaGateway AppRotateBytes 10485760
& $nssm set GeecsPvaGateway AppExit Default Restart
& $nssm set GeecsPvaGateway AppThrottle 5000
& $nssm set GeecsPvaGateway Start SERVICE_AUTO_START

Write-Host ""
Write-Host "Bootstrap done. Before starting, place GEECS config in the service profile:"
Write-Host "  $Root\profile\.config\geecs_python_api\config.ini"
Write-Host "  $Root\profile\user data\Configurations.INI"
Write-Host "Then:  $nssm start GeecsPvaGateway"
