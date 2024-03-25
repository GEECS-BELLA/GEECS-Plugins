@echo off
setlocal

rem Define global directory and config file path
set "global_dir=%userprofile%\.config\geecs_python_api"
set "config_file=%global_dir%\config.ini"

rem Check if the global directory exists, if not, create it
if not exist "%global_dir%" (
    echo Creating config directory in "%global_dir%"...
    mkdir "%global_dir%"
)

rem Check if the config file exists, if not, create it with default content
if not exist "%config_file%" (
    echo Generating new config file with default values...
    echo [Paths] > "%config_file%"
    echo geecs_data = C:\GEECS\user data\ >> "%config_file%"
    echo conda_xopt = C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS >> "%config_file%"
    echo. >> "%config_file%"
    echo [Experiment] >> "%config_file%"
    echo expt = Undulator >> "%config_file%"
    echo examples = True >> "%config_file%"
    echo legacy = False >> "%config_file%"
    echo Config file generated.
) else (
    echo Config file already exists.
)

rem Check if a shortcut exists in the current directory pointing to the config file directory
set "shortcut_name=config.lnk"
set "shortcut_target=%config_file%"

if not exist "%shortcut_name%" (
    echo Creating shortcut...
    powershell -command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('.\%shortcut_name%'); $s.TargetPath = '%shortcut_target%'; $s.Save()"
    echo Shortcut created successfully.
) else (
    echo Shortcut already exists.
)

endlocal
