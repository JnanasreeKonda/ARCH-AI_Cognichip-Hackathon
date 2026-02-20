@echo off
REM Batch script to set API keys from .env file
REM Usage: set_api_keys.bat

echo Setting API keys from .env file...

if exist .env (
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" (
            if not "%%a"=="#" (
                set "%%a=%%b"
                echo   Set %%a
            )
        )
    )
    echo.
    echo API keys loaded! You can now run: python main.py
) else (
    echo .env file not found!
    echo Please create .env file with your API keys.
)
