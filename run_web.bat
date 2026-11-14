@echo off
echo ===================================================
echo  Starting AI Fall Detection Web Server
echo ===================================================
"%~dp0venv\Scripts\python.exe" "%~dp0web_server.py" %*
