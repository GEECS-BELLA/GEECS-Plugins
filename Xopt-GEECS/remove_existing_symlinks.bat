@echo off
setlocal

rem Check if directory argument is provided
if "%-1"=="" (
    echo Usage: %0 directory_path
    exit /b 1
)

set "directory=%~1"
echo Deleting old symbolic links in "%directory%"

rem Iterate over all files in the directory
for /f "tokens=*" %%F in ('dir /b /a:-d "%directory%"') do (
    rem Check if the file is a symbolic link
    if exist "%directory%\%%F" (
        rem Check if the file is a symbolic link
        fsutil reparsepoint query "%directory%\%%F" >nul 2>nul
        if not errorlevel 1 (
            rem Delete the symbolic link
            del "%directory%\%%F"
            echo Deleted symbolic link: "%directory%\%%F"
        )
    )
)

endlocal
