@echo off
setlocal
cd /d "%~dp0"
if not exist "imp\Scripts\python.exe" (
  echo Python du projet introuvable. Suivre les commandes du README pour installer l'environnement.
  pause
  exit /b 1
)
"imp\Scripts\python.exe" -m src.cli --offline retrieve --image-id 490 --top-k 5 --show
set "taskExit=%errorlevel%"
if not "%taskExit%"=="0" pause
exit /b %taskExit%
