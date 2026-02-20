@echo off
REM Batch script to activate the virtual environment
REM Usage: activate_venv.bat

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
python --version
pip --version
echo.
echo To deactivate, run: deactivate
