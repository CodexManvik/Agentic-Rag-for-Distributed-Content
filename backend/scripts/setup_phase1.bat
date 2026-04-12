@echo off
REM Phase 1 Setup Script - Creates all required directories for the refactoring plan
REM Run this script from the repository root: backend\scripts\setup_phase1.bat

echo Creating inference module directories...
mkdir backend\app\inference 2>nul
mkdir backend\app\system 2>nul
mkdir backend\app\session 2>nul
mkdir backend\app\supervisor 2>nul
mkdir backend\app\supervisor\prompts 2>nul
mkdir backend\app\agent_builder 2>nul
mkdir backend\app\agent_builder\prompts 2>nul
mkdir backend\app\models 2>nul
mkdir backend\app\tools\file_access 2>nul
mkdir backend\app\tools\web_search 2>nul
mkdir backend\app\agents\builtin 2>nul
mkdir backend\scripts 2>nul
mkdir backend\tests\unit 2>nul
mkdir backend\tests\integration 2>nul
mkdir backend\tests\benchmarks 2>nul
mkdir docs 2>nul

echo.
echo All directories created successfully!
echo.
echo Next step: Run 'pip install -r requirements.txt' after updating dependencies
pause
