@echo off
REM A# Compiler Complete Build Script
REM Builds the full A# compiler with ML/AI features and tooling

echo ========================================
echo A# Compiler Complete Build Script
echo ========================================
echo.

REM Check for required tools
echo Checking build environment...
where gcc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: GCC not found. Please install GCC or add it to PATH.
    exit /b 1
)

where make >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Make not found. Using batch build instead.
    set USE_MAKE=0
) else (
    set USE_MAKE=1
)

REM Create directories
echo Creating build directories...
if not exist obj mkdir obj
if not exist bin mkdir bin
if not exist lib mkdir lib
if not exist include mkdir include
if not exist vim mkdir vim
if not exist examples mkdir examples
if not exist tests mkdir tests
if not exist docs mkdir docs

REM Copy headers to include directory
echo Copying headers...
copy src\*.h include\ >nul 2>nul

REM Compile source files
echo Compiling A# compiler...
gcc -Wall -Wextra -std=c99 -O2 -g -c src\main.c -o obj\main.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src\lexer.c -o obj\lexer.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src\parser.c -o obj\parser.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src\typecheck.c -o obj\typecheck.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src\ir.c -o obj\ir.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src\codegen.c -o obj\codegen.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src\ml_ai.c -o obj\ml_ai.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src\lsp_server.c -o obj\lsp_server.o

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation failed!
    exit /b 1
)

REM Link executable
echo Linking A# compiler...
gcc obj\main.o obj\lexer.o obj\parser.o obj\typecheck.o obj\ir.o obj\codegen.o obj\ml_ai.o obj\lsp_server.o -o bin\ashc.exe -lm -ljson-c

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Linking failed!
    exit /b 1
)

REM Create language server
echo Creating language server...
gcc obj\lexer.o obj\parser.o obj\typecheck.o obj\ir.o obj\codegen.o obj\ml_ai.o obj\lsp_server.o -o bin\ash-lsp.exe -lm -ljson-c

REM Create test runner
echo Creating test runner...
gcc -Wall -Wextra -std=c99 -O2 -g tests\test_lexer.c src\lexer.c -o bin\test_lexer.exe -lm

REM Create Vim integration
echo Setting up Vim integration...
copy vim\ash.vim %USERPROFILE%\.vim\syntax\ash.vim >nul 2>nul
if not exist %USERPROFILE%\.vim\syntax mkdir %USERPROFILE%\.vim\syntax

REM Create configuration files
echo Creating configuration files...
echo # A# Compiler Configuration > ash.conf
echo compiler_path=bin\ashc.exe >> ash.conf
echo lsp_path=bin\ash-lsp.exe >> ash.conf
echo vim_syntax=%USERPROFILE%\.vim\syntax\ash.vim >> ash.conf

REM Test compilation
echo Testing A# compiler...
bin\ashc.exe -v examples\hello_world.ash -o test_output.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ Hello World compilation successful
) else (
    echo ✗ Hello World compilation failed
)

bin\ashc.exe -v examples\ownership_demo.ash -o test_ownership.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ Ownership demo compilation successful
) else (
    echo ✗ Ownership demo compilation failed
)

bin\ashc.exe -v examples\ml_neural_network.ash -o test_ml.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ ML/AI demo compilation successful
) else (
    echo ✗ ML/AI demo compilation failed
)

REM Run tests
echo Running test suite...
bin\test_lexer.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ Lexer tests passed
) else (
    echo ✗ Lexer tests failed
)

REM Create installation script
echo Creating installation script...
echo @echo off > install.bat
echo echo Installing A# Compiler... >> install.bat
echo copy bin\ashc.exe %USERPROFILE%\bin\ashc.exe >> install.bat
echo copy bin\ash-lsp.exe %USERPROFILE%\bin\ash-lsp.exe >> install.bat
echo copy ash.conf %USERPROFILE%\.ash.conf >> install.bat
echo echo A# Compiler installed successfully! >> install.bat
echo echo Usage: ashc [options] ^<input_file^> >> install.bat

REM Create uninstallation script
echo Creating uninstallation script...
echo @echo off > uninstall.bat
echo echo Uninstalling A# Compiler... >> uninstall.bat
echo del %USERPROFILE%\bin\ashc.exe >> uninstall.bat
echo del %USERPROFILE%\bin\ash-lsp.exe >> uninstall.bat
echo del %USERPROFILE%\.ash.conf >> uninstall.bat
echo echo A# Compiler uninstalled successfully! >> uninstall.bat

REM Create package
echo Creating distribution package...
if exist ash-compiler.zip del ash-compiler.zip
powershell Compress-Archive -Path bin,include,examples,tests,docs,vim,*.md,*.conf,*.bat -DestinationPath ash-compiler.zip

echo.
echo ========================================
echo A# Compiler Build Complete!
echo ========================================
echo.
echo Built components:
echo   ✓ A# Compiler (bin\ashc.exe)
echo   ✓ Language Server (bin\ash-lsp.exe)
echo   ✓ Test Suite (bin\test_lexer.exe)
echo   ✓ Vim Integration (vim\ash.vim)
echo   ✓ Example Programs (examples\)
echo   ✓ Documentation (docs\)
echo.
echo Usage:
echo   bin\ashc.exe [options] ^<input_file^>
echo.
echo Options:
echo   -o ^<file^>     Output file
echo   -v              Verbose output
echo   -d              Debug mode
echo   --verify        Enable verification
echo   --ml            Enable ML/AI features
echo   -h, --help      Show help
echo.
echo Examples:
echo   bin\ashc.exe -v examples\hello_world.ash
echo   bin\ashc.exe --ml examples\ml_neural_network.ash
echo   bin\ashc.exe -o program.exe my_program.ash
echo.
echo Installation:
echo   run install.bat
echo.
echo Package created: ash-compiler.zip
echo.
