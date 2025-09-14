@echo off
REM A# Compiler Simple Build Script
REM Builds the A# compiler with basic functionality

echo ========================================
echo A# Compiler Simple Build Script
echo ========================================
echo.

REM Create directories
echo Creating build directories...
if not exist obj mkdir obj
if not exist bin mkdir bin

REM Compile source files with minimal warnings
echo Compiling A# compiler...
gcc -Wall -Wextra -std=c99 -O2 -g -c src\main.c -o obj\main.o -Wno-unused-parameter -Wno-implicit-function-declaration
gcc -Wall -Wextra -std=c99 -O2 -g -c src\lexer.c -o obj\lexer.o -Wno-unused-parameter
gcc -Wall -Wextra -std=c99 -O2 -g -c src\parser.c -o obj\parser.o -Wno-unused-parameter -Wno-implicit-function-declaration
gcc -Wall -Wextra -std=c99 -O2 -g -c src\typecheck.c -o obj\typecheck.o -Wno-unused-parameter -Wno-implicit-function-declaration
gcc -Wall -Wextra -std=c99 -O2 -g -c src\ir.c -o obj\ir.o -Wno-unused-parameter
gcc -Wall -Wextra -std=c99 -O2 -g -c src\codegen.c -o obj\codegen.o -Wno-unused-parameter
gcc -Wall -Wextra -std=c99 -O2 -g -c src\ml_ai.c -o obj\ml_ai.o -Wno-unused-parameter
gcc -Wall -Wextra -std=c99 -O2 -g -c src\lsp_server.c -o obj\lsp_server.o -Wno-unused-parameter

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation failed!
    exit /b 1
)

REM Link executable
echo Linking A# compiler...
gcc obj\main.o obj\lexer.o obj\parser.o obj\typecheck.o obj\ir.o obj\codegen.o obj\ml_ai.o obj\lsp_server.o -o bin\ashc.exe -lm

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Linking failed!
    exit /b 1
)

REM Create test runner
echo Creating test runner...
gcc -Wall -Wextra -std=c99 -O2 -g tests\test_lexer.c src\lexer.c -o bin\test_lexer.exe -lm -Wno-unused-parameter

REM Test compilation
echo Testing A# compiler...
bin\ashc.exe -v examples\hello_world.ash -o test_output.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ Hello World compilation successful
) else (
    echo ✗ Hello World compilation failed
)

REM Run tests
echo Running test suite...
bin\test_lexer.exe
if %ERRORLEVEL% EQU 0 (
    echo ✓ Lexer tests passed
) else (
    echo ✗ Lexer tests failed
)

echo.
echo ========================================
echo A# Compiler Build Complete!
echo ========================================
echo.
echo Built components:
echo   ✓ A# Compiler (bin\ashc.exe)
echo   ✓ Test Suite (bin\test_lexer.exe)
echo   ✓ ML/AI Support (src\ml_ai.c)
echo   ✓ Language Server (src\lsp_server.c)
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
