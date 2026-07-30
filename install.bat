@echo off
:: Lock on Absence — one-click installer for Windows (v4.1)
:: Creates a venv, installs deps, tests webcam, sets up auto-start.

cd /d "%~dp0"
setlocal enabledelayedexpansion

echo ================================================
echo   Lock on Absence — Windows Installer v4.1
echo ================================================
echo.

:: 1. Find Python 3.8+
echo [1/5] Checking Python 3.8+...
set PYTHON=
for %%p in (python python3) do (
    where %%p >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2 delims=." %%a in ('%%p --version 2^>^&1 ^| findstr /r "3\.[0-9]"') do (
            if %%a geq 8 set PYTHON=%%p
        )
        if defined PYTHON goto :pyok
    )
)
:: Fallback: LocalAppData Python
for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%d\python.exe" (
        for /f "tokens=2 delims=." %%a in ('"%%d\python.exe" --version 2^>^&1 ^| findstr /r "3\.[0-9]"') do (
            if %%a geq 8 set PYTHON="%%d\python.exe"
        )
        if defined PYTHON goto :pyok
    )
)
echo ERROR: Python 3.8+ not found.
echo Install from https://python.org (check "Add Python to PATH")
pause
exit /b 1

:pyok
%PYTHON% --version
echo.

:: 2. Create virtual environment
echo [2/5] Creating virtual environment...
if not exist venv\ (
    %PYTHON% -m venv venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )
)
echo OK
echo.

:: 3. Install dependencies
echo [3/5] Installing dependencies (in venv)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: pip install failed. Trying alternative...
    pip install -r requirements.txt --quiet
)
python -c "import cv2, numpy; print('OpenCV + numpy OK')" 2>nul
if errorlevel 1 (
    echo WARNING: Dependency verification failed.
    echo Run: python -m pip install -r requirements.txt
)
echo OK
echo.

:: 4. Test webcam
echo [4/5] Testing webcam...
python -c "import cv2; cap=cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() and cap.read()[0] else 'WARNING: Webcam not detected — check camera driver or index'); cap.release()"
echo.

:: 5. Auto-start with Windows
echo [5/5] Setting up auto-start...
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
echo     python enroll.py --samples 50
echo.
echo   To start now:
echo     run_hidden.bat
echo.
echo   To start with custom model:
echo     python lock-on-absence.py --model face_model_Alice.json
echo.
echo   Auto-start on next login: ENABLED
echo.
echo   To UNINSTALL: delete %VBS%
echo ================================================
pause
