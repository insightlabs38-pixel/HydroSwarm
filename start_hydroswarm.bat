@echo off
rem Compatibility wrapper. Prefer the explicit PowerShell launcher, which
rem uses the project-local .venv interpreter and never an ambient system
rem Python:
rem   .\start_hydroswarm_windows.ps1
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_hydroswarm_windows.ps1" %*
exit /b %ERRORLEVEL%
