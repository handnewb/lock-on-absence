@echo off
:: Lock on Absence — one-click installer for Windows
:: Installs dependencies, tests webcam, and sets up auto-start.

cd /d "%~dp0"
setlocal enabledelayedexpansion

echo ================================================
echo   Lock on Absence — Windows Installer
echo ================================================
echo.

:: 1. Find Python
echo [1/4] Checking Python...
set PYTHON=
for %%p in (python python3) do (
    where %%p >nul 2>&1
    if not errorlevel 1 (
        set PYTHON=%%p
        goto :pyok
    )
)
for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%d\python.exe" (
        set PYTHON="%%d\python.exe"
        goto :pyok
    )
)
echo ERROR: Python not found.
echo Install from https://python.org (check "Add Python to PATH")
pause
exit /b 1

:pyok
%PYTHON% --version
echo.

:: 2. Install dependencies
echo [2/4] Installing dependencies...
%PYTHON% -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: some packages failed. Trying with --user...
    %PYTHON% -m pip install -r requirements.txt --user --quiet
)
echo OK
echo.

:: 3. Test webcam
echo [3/4] Testing webcam...
%PYTHON% -c "import cv2; cap=cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Webcam FAILED'); cap.release()"
echo.

:: 4. Auto-start with Windows
echo [4/4] Setting up auto-start...
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS=%STARTUP%\LockOnAbsence.vbs"

(
echo Set WshShell = CreateObject^("WScript.Shell"^)
echo WshShell.Run """%~dp0run_hidden.bat""", 0, False
) > "%VBS%"

echo.
echo ================================================
echo   Done!
echo.
echo   To enroll your face (recommended):
echo     python enroll.py
echo.
echo   To start now:
echo     run_hidden.bat
echo.
echo   Auto-start on next login: ENABLED
echo.
echo   To UNINSTALL: delete %VBS%
echo ================================================
pause
