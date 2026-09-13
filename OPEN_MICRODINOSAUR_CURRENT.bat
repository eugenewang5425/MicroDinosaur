@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if defined BLENDER_EXE goto validate_blender
if exist "%~dp0tools\blender-4.5.9-windows-x64\blender.exe" set "BLENDER_EXE=%~dp0tools\blender-4.5.9-windows-x64\blender.exe"
if defined BLENDER_EXE goto validate_blender
for /f "delims=" %%I in ('where blender.exe 2^>nul') do if not defined BLENDER_EXE set "BLENDER_EXE=%%I"
:validate_blender
if not defined BLENDER_EXE goto missing_blender
if not exist "%BLENDER_EXE%" goto missing_blender
if not exist "%~dp0current\MicroDinosaur_v1.blender" goto missing_model
for %%I in ("%~dp0current\MicroDinosaur_v1.blender") do if %%~zI LSS 1024 goto missing_model
start "MicroDinosaur v1 mechanical review" "%BLENDER_EXE%" "%~dp0current\MicroDinosaur_v1.blender" --python "%~dp0blender_controls.py"
exit /b 0
:missing_blender
echo Blender not found. Install Blender 4.5 LTS and set BLENDER_EXE to the full path of blender.exe, or add Blender to PATH.
pause
exit /b 1
:missing_model
echo Current model is missing or is a Git LFS pointer. Run: git lfs pull
pause
exit /b 1
