@echo off
REM ────────────────────────────────────────────────────────
REM Creates (if needed) and populates a global venv at:
REM    %USERPROFILE%\.venvs\global_thesis_env
REM ────────────────────────────────────────────────────────

SET VENV_DIR=%USERPROFILE%\.venvs\global_thesis_env

if not exist "%VENV_DIR%" (
  echo ➜ Creating global_thesis_env at %VENV_DIR%
  py -3.10 -m venv "%VENV_DIR%"
) else (
  echo ➜ global_thesis_env already exists at %VENV_DIR%
)

echo ➜ Activating and upgrading pip…
call "%VENV_DIR%\Scripts\activate"
python -m pip install --upgrade pip

echo ➜ Installing core packages…
pip install mediapipe opencv-python PyQt5 ffmpeg-python

echo Downloading & installing MSVC runtime (silent)...
powershell -Command "Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile 'vc_redist.x64.exe'; Start-Process -FilePath 'vc_redist.x64.exe' -ArgumentList '/install','/quiet','/norestart' -Wait; Remove-Item 'vc_redist.x64.exe'"


echo.
echo ✅  global_thesis_env is ready!
echo To use it in *any* project, just run:
echo    call "%%USERPROFILE%%\.venvs\global_thesis_env\Scripts\activate"
pause
