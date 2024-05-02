@echo off
setlocal

SET PYTHON_SCRIPT=generate_symlink_bat.py

CALL python "%python_script%" %ANACONDA_Xopt_GEECS%
if exist "autosetup_Xopt-GEECS.bat" (
    CALL autosetup_Xopt-GEECS.bat
    ECHO.
    ECHO Run Badger on a conda terminal using 'badger -g' after activating the Xopt-GEECS environment
)

endlocal
