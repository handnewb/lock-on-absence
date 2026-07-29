@echo off
:: Lock on Absence — silent launcher for Windows
:: Double-click to start monitoring without a terminal window.

cd /d "%~dp0"

:: Find Python — try common locations
set PYTHON=
for %%p in (python python3) do (
    where %%p >nul 2>&1
    if not errorlevel 1 (
        set PYTHON=%%p
        goto :found
    )
)

:: Try Microsoft Store path
for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%d\pythonw.exe" (
        set PYTHON="%%d\pythonw.exe"
        goto :found
    )
)

echo ERROR: Python not found. Install from https://python.org
pause
exit /b 1

:found
:: Check for face model
if not exist "%~dp0face_model.yml" (
    echo No face model found — running in ANY-FACE mode.
    echo Run 'python enroll.py' first for owner-only recognition.
    echo.
)

echo Starting Lock on Absence...
start "" /B %PYTHON% "%~dp0lock-on-absence.py" %*
echo Running in background. Check Task Manager for pythonw.exe.
exit /b 0
