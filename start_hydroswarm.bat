@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%PROJECT_ROOT%src;%PYTHONPATH%"
python -m hydroswarm.cli self-test || exit /b 1
python -m hydroswarm.cli start --host 127.0.0.1 --port 8765
exit /b %ERRORLEVEL%
