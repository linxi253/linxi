@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_annotation.ps1" -Modality haadf_stem
