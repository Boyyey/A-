@echo off
echo ========================================
echo A# Project Organization Script
echo ========================================

echo.
echo Organizing A# project structure...

REM Create main project directories
if not exist "build" mkdir build
if not exist "build\scripts" mkdir build\scripts
if not exist "build\output" mkdir build\output
if not exist "build\logs" mkdir build\logs

if not exist "tools" mkdir tools
if not exist "tools\cmd" mkdir tools\cmd
if not exist "tools\vim" mkdir tools\vim
if not exist "tools\editors" mkdir tools\editors

if not exist "assets" mkdir assets
if not exist "assets\images" mkdir assets\images
if not exist "assets\icons" mkdir assets\icons

if not exist "scripts" mkdir scripts
if not exist "scripts\setup" mkdir scripts\setup
if not exist "scripts\build" mkdir scripts\build
if not exist "scripts\dev" mkdir scripts\dev

if not exist "projects" mkdir projects
if not exist "projects\templates" mkdir projects\templates
if not exist "projects\examples" mkdir projects\examples
if not exist "projects\ai" mkdir projects\ai

echo.
echo Moving build scripts...
move build*.bat build\scripts\
move install_mingw64.bat build\scripts\
move setup_cmd.bat build\scripts\

echo Moving CMD tools...
move cmd_*.bat tools\cmd\

echo Moving Vim files...
move vim\*.* tools\vim\
rmdir vim

echo Moving assets...
move "A# Logo.png" assets\images\
move "A# Logo with a white background.png" assets\images\

echo Moving project files...
move cmd_projects\*.* projects\examples\
rmdir cmd_projects

echo Moving documentation...
move A_SHARP_COMPLETE_SUMMARY.md docs\

echo.
echo Creating organized structure...

REM Create main README
echo # A# Programming Language > README.md
echo ## Advanced AI/ML Language with Formal Verification >> README.md
echo. >> README.md
echo A# is the most advanced programming language designed specifically for AI, machine learning, and systems programming with formal verification. >> README.md
echo. >> README.md
echo ### Quick Start >> README.md
echo. >> README.md
echo ```bash >> README.md
echo # Build A# compiler >> README.md
echo .\build\scripts\build_minimal.bat >> README.md
echo. >> README.md
echo # Create new project >> README.md
echo .\tools\cmd\cmd_new.bat my_project >> README.md
echo. >> README.md
echo # Compile and run >> README.md
echo .\tools\cmd\cmd_compile.bat projects\examples\my_project.ash >> README.md
echo ``` >> README.md
echo. >> README.md
echo ### Project Structure >> README.md
echo. >> README.md
echo - `src/` - A# compiler source code >> README.md
echo - `bin/` - Compiled executables >> README.md
echo - `docs/` - Documentation >> README.md
echo - `examples/` - Example A# programs >> README.md
echo - `projects/` - User projects and templates >> README.md
echo - `tools/` - Development tools >> README.md
echo - `build/` - Build scripts and output >> README.md
echo - `tests/` - Test suite >> README.md
echo - `formal_semantics/` - Formal verification >> README.md
echo. >> README.md
echo ### Features >> README.md
echo. >> README.md
echo - 🧠 **AI/ML Built-in** - Neural networks, tensors, GPU acceleration >> README.md
echo - 🛡️ **Memory Safety** - Ownership system prevents segfaults >> README.md
echo - 🔬 **Formal Verification** - Mechanized proofs of correctness >> README.md
echo - ⚡ **Performance** - Zero-cost abstractions >> README.md
echo - 🎭 **Concurrency** - Actor-based programming >> README.md
echo - 🔧 **Metaprogramming** - Hygienic macros >> README.md
echo. >> README.md
echo **A# - Where Safety Meets AI, and Research Meets Production!** 🚀 >> README.md

REM Create project structure documentation
echo # A# Project Structure > docs\PROJECT_STRUCTURE.md
echo ## Organized A# Development Environment >> docs\PROJECT_STRUCTURE.md
echo. >> docs\PROJECT_STRUCTURE.md
echo This document describes the organized structure of the A# project. >> docs\PROJECT_STRUCTURE.md
echo. >> docs\PROJECT_STRUCTURE.md
echo ### Directory Layout >> docs\PROJECT_STRUCTURE.md
echo. >> docs\PROJECT_STRUCTURE.md
echo ``` >> docs\PROJECT_STRUCTURE.md
echo A#/ >> docs\PROJECT_STRUCTURE.md
echo ├── src/                    # A# compiler source code >> docs\PROJECT_STRUCTURE.md
echo │   ├── main.c             # Main compiler entry point >> docs\PROJECT_STRUCTURE.md
echo │   ├── lexer.c/.h         # Lexical analysis >> docs\PROJECT_STRUCTURE.md
echo │   ├── parser.c/.h        # Syntax analysis >> docs\PROJECT_STRUCTURE.md
echo │   ├── typecheck.c/.h     # Type checking >> docs\PROJECT_STRUCTURE.md
echo │   ├── ir.c/.h            # Intermediate representation >> docs\PROJECT_STRUCTURE.md
echo │   ├── codegen.c/.h       # Code generation >> docs\PROJECT_STRUCTURE.md
echo │   ├── ml_ai.c/.h         # AI/ML features >> docs\PROJECT_STRUCTURE.md
echo │   ├── library_system.h   # Library management >> docs\PROJECT_STRUCTURE.md
echo │   └── lsp_server.c       # Language server >> docs\PROJECT_STRUCTURE.md
echo ├── bin/                    # Compiled executables >> docs\PROJECT_STRUCTURE.md
echo │   └── ashc.exe           # A# compiler >> docs\PROJECT_STRUCTURE.md
echo ├── docs/                   # Documentation >> docs\PROJECT_STRUCTURE.md
echo │   ├── A_SHARP_PROGRAMMING_GUIDE.md >> docs\PROJECT_STRUCTURE.md
echo │   ├── AI_ML_CAPABILITIES.md >> docs\PROJECT_STRUCTURE.md
echo │   ├── MINGW64_PROGRAMMING_GUIDE.md >> docs\PROJECT_STRUCTURE.md
echo │   ├── CMD_PROGRAMMING_GUIDE.md >> docs\PROJECT_STRUCTURE.md
echo │   └── PROJECT_STRUCTURE.md >> docs\PROJECT_STRUCTURE.md
echo ├── examples/               # Example A# programs >> docs\PROJECT_STRUCTURE.md
echo │   ├── hello_world.ash    # Basic hello world >> docs\PROJECT_STRUCTURE.md
echo │   ├── ownership_demo.ash # Ownership examples >> docs\PROJECT_STRUCTURE.md
echo │   ├── concurrency_demo.ash # Concurrency examples >> docs\PROJECT_STRUCTURE.md
echo │   ├── type_system_demo.ash # Type system examples >> docs\PROJECT_STRUCTURE.md
echo │   ├── ml_neural_network.ash # ML examples >> docs\PROJECT_STRUCTURE.md
echo │   ├── transformer_gpt.ash # Advanced AI examples >> docs\PROJECT_STRUCTURE.md
echo │   ├── ai_research_lab.ash # Complete AI lab >> docs\PROJECT_STRUCTURE.md
echo │   └── hello_mingw64.ash  # MinGW64 examples >> docs\PROJECT_STRUCTURE.md
echo ├── projects/               # User projects and templates >> docs\PROJECT_STRUCTURE.md
echo │   ├── templates/         # Project templates >> docs\PROJECT_STRUCTURE.md
echo │   ├── examples/          # User examples >> docs\PROJECT_STRUCTURE.md
echo │   └── ai/                # AI/ML projects >> docs\PROJECT_STRUCTURE.md
echo ├── tools/                  # Development tools >> docs\PROJECT_STRUCTURE.md
echo │   ├── cmd/               # CMD development tools >> docs\PROJECT_STRUCTURE.md
echo │   ├── vim/               # Vim syntax highlighting >> docs\PROJECT_STRUCTURE.md
echo │   └── editors/           # Editor configurations >> docs\PROJECT_STRUCTURE.md
echo ├── build/                  # Build system >> docs\PROJECT_STRUCTURE.md
echo │   ├── scripts/           # Build scripts >> docs\PROJECT_STRUCTURE.md
echo │   ├── output/            # Build output >> docs\PROJECT_STRUCTURE.md
echo │   └── logs/              # Build logs >> docs\PROJECT_STRUCTURE.md
echo ├── tests/                  # Test suite >> docs\PROJECT_STRUCTURE.md
echo │   └── test_lexer.c       # Lexer tests >> docs\PROJECT_STRUCTURE.md
echo ├── formal_semantics/       # Formal verification >> docs\PROJECT_STRUCTURE.md
echo │   └── ash_core.v         # Coq formalization >> docs\PROJECT_STRUCTURE.md
echo ├── assets/                 # Project assets >> docs\PROJECT_STRUCTURE.md
echo │   ├── images/            # Images and logos >> docs\PROJECT_STRUCTURE.md
echo │   └── icons/             # Icons >> docs\PROJECT_STRUCTURE.md
echo ├── scripts/                # Utility scripts >> docs\PROJECT_STRUCTURE.md
echo │   ├── setup/             # Setup scripts >> docs\PROJECT_STRUCTURE.md
echo │   ├── build/             # Build utilities >> docs\PROJECT_STRUCTURE.md
echo │   └── dev/               # Development utilities >> docs\PROJECT_STRUCTURE.md
echo ├── include/                # Header files >> docs\PROJECT_STRUCTURE.md
echo ├── lib/                    # Libraries >> docs\PROJECT_STRUCTURE.md
echo ├── obj/                    # Object files >> docs\PROJECT_STRUCTURE.md
echo └── README.md               # Main project README >> docs\PROJECT_STRUCTURE.md
echo ``` >> docs\PROJECT_STRUCTURE.md

REM Create development workflow script
echo @echo off > scripts\dev\dev_workflow.bat
echo echo A# Development Workflow >> scripts\dev\dev_workflow.bat
echo echo ======================= >> scripts\dev\dev_workflow.bat
echo echo. >> scripts\dev\dev_workflow.bat
echo echo 1. Build compiler >> scripts\dev\dev_workflow.bat
echo echo 2. Create new project >> scripts\dev\dev_workflow.bat
echo echo 3. Edit project >> scripts\dev\dev_workflow.bat
echo echo 4. Compile project >> scripts\dev\dev_workflow.bat
echo echo 5. Run project >> scripts\dev\dev_workflow.bat
echo echo 6. Run tests >> scripts\dev\dev_workflow.bat
echo echo. >> scripts\dev\dev_workflow.bat
echo set /p choice="Enter choice (1-6): " >> scripts\dev\dev_workflow.bat
echo if %%choice%%==1 .\build\scripts\build_minimal.bat >> scripts\dev\dev_workflow.bat
echo if %%choice%%==2 .\tools\cmd\cmd_new.bat %%2 >> scripts\dev\dev_workflow.bat
echo if %%choice%%==3 .\tools\cmd\cmd_edit.bat %%2 >> scripts\dev\dev_workflow.bat
echo if %%choice%%==4 .\tools\cmd\cmd_compile.bat %%2 >> scripts\dev\dev_workflow.bat
echo if %%choice%%==5 .\%%2.exe >> scripts\dev\dev_workflow.bat
echo if %%choice%%==6 .\bin\test_lexer.exe >> scripts\dev\dev_workflow.bat

REM Create quick start script
echo @echo off > quick_start.bat
echo echo A# Quick Start >> quick_start.bat
echo echo =============== >> quick_start.bat
echo echo. >> quick_start.bat
echo echo Building A# compiler... >> quick_start.bat
echo .\build\scripts\build_minimal.bat >> quick_start.bat
echo echo. >> quick_start.bat
echo echo Creating hello world example... >> quick_start.bat
echo .\tools\cmd\cmd_new.bat hello_world >> quick_start.bat
echo echo. >> quick_start.bat
echo echo Compiling and running hello world... >> quick_start.bat
echo .\tools\cmd\cmd_compile.bat projects\examples\hello_world.ash >> quick_start.bat
echo echo. >> quick_start.bat
echo echo A# is ready! Check projects\examples\ for your programs. >> quick_start.bat

REM Create .gitignore
echo # A# Project .gitignore > .gitignore
echo # Compiled executables >> .gitignore
echo *.exe >> .gitignore
echo *.dll >> .gitignore
echo *.so >> .gitignore
echo *.dylib >> .gitignore
echo. >> .gitignore
echo # Object files >> .gitignore
echo *.o >> .gitignore
echo *.obj >> .gitignore
echo. >> .gitignore
echo # Build output >> .gitignore
echo build/output/* >> .gitignore
echo build/logs/* >> .gitignore
echo. >> .gitignore
echo # IDE files >> .gitignore
echo .vscode/ >> .gitignore
echo .idea/ >> .gitignore
echo *.swp >> .gitignore
echo *.swo >> .gitignore
echo. >> .gitignore
echo # OS files >> .gitignore
echo Thumbs.db >> .gitignore
echo .DS_Store >> .gitignore
echo. >> .gitignore
echo # User projects >> .gitignore
echo projects/examples/*.ash >> .gitignore
echo projects/ai/*.ash >> .gitignore
echo !projects/examples/hello_world.ash >> .gitignore

echo.
echo ========================================
echo A# Project Organization Complete!
echo ========================================
echo.
echo New project structure:
echo ├── src/                    # Compiler source code
echo ├── bin/                    # Compiled executables
echo ├── docs/                   # Documentation
echo ├── examples/               # Example programs
echo ├── projects/               # User projects
echo ├── tools/                  # Development tools
echo ├── build/                  # Build system
echo ├── tests/                  # Test suite
echo ├── formal_semantics/       # Formal verification
echo ├── assets/                 # Project assets
echo └── scripts/                # Utility scripts
echo.
echo Quick start:
echo   .\quick_start.bat
echo.
echo Development workflow:
echo   .\scripts\dev\dev_workflow.bat
echo.
echo Project is now organized and ready for development! 🚀
