@echo off
echo ========================================
echo A# CMD Development Environment Setup
echo ========================================

REM Create CMD-friendly directories
if not exist "cmd_projects" mkdir cmd_projects
if not exist "cmd_projects\examples" mkdir cmd_projects\examples
if not exist "cmd_projects\templates" mkdir cmd_projects\templates

REM Create a simple text editor helper
echo @echo off > cmd_edit.bat
echo echo A# Simple Text Editor >> cmd_edit.bat
echo echo ====================== >> cmd_edit.bat
echo echo. >> cmd_edit.bat
echo echo Opening %1 in notepad... >> cmd_edit.bat
echo notepad %1 >> cmd_edit.bat

REM Create a quick compile script
echo @echo off > cmd_compile.bat
echo echo Compiling A# program: %1 >> cmd_compile.bat
echo .\bin\ashc.exe %1 -o %~n1.exe >> cmd_compile.bat
echo if %errorlevel% equ 0 ( >> cmd_compile.bat
echo     echo Compilation successful! >> cmd_compile.bat
echo     echo Running %~n1.exe... >> cmd_compile.bat
echo     .\%~n1.exe >> cmd_compile.bat
echo ^) else ( >> cmd_compile.bat
echo     echo Compilation failed! >> cmd_compile.bat
echo ^) >> cmd_compile.bat

REM Create a project template
echo mod my_project { > cmd_projects\templates\project_template.ash
echo     fn main() -^> i32 @ Pure { >> cmd_projects\templates\project_template.ash
echo         println("Hello from A# CMD!"); >> cmd_projects\templates\project_template.ash
echo         return 0; >> cmd_projects\templates\project_template.ash
echo     } >> cmd_projects\templates\project_template.ash
echo } >> cmd_projects\templates\project_template.ash

REM Create AI template
echo mod ai_project { > cmd_projects\templates\ai_template.ash
echo     use std::ml::*; >> cmd_projects\templates\ai_template.ash
echo. >> cmd_projects\templates\ai_template.ash
echo     fn main() -^> () @ Resource { >> cmd_projects\templates\ai_template.ash
echo         println("A# AI Project in CMD"); >> cmd_projects\templates\ai_template.ash
echo. >> cmd_projects\templates\ai_template.ash
echo         // Create neural network >> cmd_projects\templates\ai_template.ash
echo         let model = create_mlp([784, 128, 10], 3, ACTIVATION_RELU, 0.001); >> cmd_projects\templates\ai_template.ash
echo         println("Neural network created!"); >> cmd_projects\templates\ai_template.ash
echo     } >> cmd_projects\templates\ai_template.ash
echo } >> cmd_projects\templates\ai_template.ash

REM Create a help script
echo @echo off > cmd_help.bat
echo echo A# CMD Development Help >> cmd_help.bat
echo echo ========================= >> cmd_help.bat
echo echo. >> cmd_help.bat
echo echo Available commands: >> cmd_help.bat
echo echo   cmd_edit filename.ash    - Edit A# file in notepad >> cmd_help.bat
echo echo   cmd_compile filename.ash - Compile and run A# program >> cmd_help.bat
echo echo   cmd_new project_name     - Create new A# project >> cmd_help.bat
echo echo   cmd_ai project_name      - Create new AI project >> cmd_help.bat
echo echo   cmd_help                 - Show this help >> cmd_help.bat
echo echo. >> cmd_help.bat
echo echo Examples: >> cmd_help.bat
echo echo   cmd_new hello_world      - Creates hello_world.ash >> cmd_help.bat
echo echo   cmd_edit hello_world.ash - Opens in notepad >> cmd_help.bat
echo echo   cmd_compile hello_world.ash - Compiles and runs >> cmd_help.bat

REM Create new project script
echo @echo off > cmd_new.bat
echo if "%%1"=="" ( >> cmd_new.bat
echo     echo Usage: cmd_new project_name >> cmd_new.bat
echo     exit /b 1 >> cmd_new.bat
echo ^) >> cmd_new.bat
echo echo Creating new A# project: %%1 >> cmd_new.bat
echo copy cmd_projects\templates\project_template.ash cmd_projects\%%1.ash >> cmd_new.bat
echo echo Project created: cmd_projects\%%1.ash >> cmd_new.bat
echo echo Edit with: cmd_edit cmd_projects\%%1.ash >> cmd_new.bat
echo echo Compile with: cmd_compile cmd_projects\%%1.ash >> cmd_new.bat

REM Create AI project script
echo @echo off > cmd_ai.bat
echo if "%%1"=="" ( >> cmd_ai.bat
echo     echo Usage: cmd_ai project_name >> cmd_ai.bat
echo     exit /b 1 >> cmd_ai.bat
echo ^) >> cmd_ai.bat
echo echo Creating new A# AI project: %%1 >> cmd_ai.bat
echo copy cmd_projects\templates\ai_template.ash cmd_projects\%%1.ash >> cmd_ai.bat
echo echo AI project created: cmd_projects\%%1.ash >> cmd_ai.bat
echo echo Edit with: cmd_edit cmd_projects\%%1.ash >> cmd_ai.bat
echo echo Compile with: cmd_compile cmd_projects\%%1.ash >> cmd_ai.bat

REM Create a quick start guide
echo A# CMD Development Quick Start > cmd_projects\README.txt
echo ================================ >> cmd_projects\README.txt
echo. >> cmd_projects\README.txt
echo 1. Create new project: >> cmd_projects\README.txt
echo    cmd_new my_project >> cmd_projects\README.txt
echo. >> cmd_projects\README.txt
echo 2. Edit your code: >> cmd_projects\README.txt
echo    cmd_edit cmd_projects\my_project.ash >> cmd_projects\README.txt
echo. >> cmd_projects\README.txt
echo 3. Compile and run: >> cmd_projects\README.txt
echo    cmd_compile cmd_projects\my_project.ash >> cmd_projects\README.txt
echo. >> cmd_projects\README.txt
echo 4. For AI projects: >> cmd_projects\README.txt
echo    cmd_ai my_ai_project >> cmd_projects\README.txt
echo. >> cmd_projects\README.txt
echo Available commands: >> cmd_projects\README.txt
echo - cmd_help          Show help >> cmd_projects\README.txt
echo - cmd_new name      Create basic project >> cmd_projects\README.txt
echo - cmd_ai name       Create AI project >> cmd_projects\README.txt
echo - cmd_edit file     Edit A# file >> cmd_projects\README.txt
echo - cmd_compile file  Compile and run >> cmd_projects\README.txt

echo.
echo ========================================
echo A# CMD Environment Setup Complete!
echo ========================================
echo.
echo Available commands:
echo   cmd_help          - Show help
echo   cmd_new name      - Create new project
echo   cmd_ai name       - Create AI project
echo   cmd_edit file     - Edit A# file
echo   cmd_compile file  - Compile and run
echo.
echo Quick start:
echo   1. cmd_new hello_world
echo   2. cmd_edit cmd_projects\hello_world.ash
echo   3. cmd_compile cmd_projects\hello_world.ash
echo.
echo Happy coding in CMD! 🚀
