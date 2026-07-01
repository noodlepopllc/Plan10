#! /bin/bash
@echo off
setlocal

:loop
python bin/bot.py %* -F --max-steps 1
set EXITCODE=%errorlevel%

if %EXITCODE% equ 0 (
    echo Completed successfully.
    exit /b 0
)

if %EXITCODE% equ 1 (
    echo OOM detected. Restarting in 5 seconds...
    timeout /t 5 /nobreak >nul
    goto loop
)

echo Failed with error code %EXITCODE%. Not retrying.
exit /b %EXITCODE%