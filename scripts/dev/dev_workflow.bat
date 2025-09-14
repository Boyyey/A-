@echo off 
echo A# Development Workflow 
echo ======================= 
echo. 
echo 1. Build compiler 
echo 2. Create new project 
echo 3. Edit project 
echo 4. Compile project 
echo 5. Run project 
echo 6. Run tests 
echo. 
set /p choice="Enter choice (1-6): " 
if %choice%==1 .\build\scripts\build_minimal.bat 
if %choice%==2 .\tools\cmd\cmd_new.bat %2 
if %choice%==3 .\tools\cmd\cmd_edit.bat %2 
if %choice%==4 .\tools\cmd\cmd_compile.bat %2 
if %choice%==5 .\%2.exe 
if %choice%==6 .\bin\test_lexer.exe 
