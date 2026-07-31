@echo off
REM Thin wrapper: the real installer is install.ps1.
REM
REM The old install.bat wrote a hidden VBS into the Startup folder to launch the
REM agent with no visible window. That pattern -- webcam access plus a hidden
REM launcher plus persistence -- is what EDR products flag as stalkerware, and it
REM made the process impossible to stop cleanly. It is gone.
echo Running install.ps1...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
pause
