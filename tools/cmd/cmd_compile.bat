@echo off
echo Compiling A# program: %1
.\bin\ashc.exe "%1" -o "%~n1.exe"
if %errorlevel% equ 0 (
    echo Compilation successful!
    echo Running %~n1.exe...
    .\"%~n1.exe"
) else (
    echo Compilation failed!
)