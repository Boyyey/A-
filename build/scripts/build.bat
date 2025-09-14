@echo off
REM A# Compiler Build Script for Windows

echo Building A# Compiler...

REM Create directories
if not exist obj mkdir obj
if not exist bin mkdir bin

REM Compile source files
echo Compiling source files...
gcc -Wall -Wextra -std=c99 -O2 -g -c src/main.c -o obj/main.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src/lexer.c -o obj/lexer.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src/parser.c -o obj/parser.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src/typecheck.c -o obj/typecheck.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src/ir.c -o obj/ir.o
gcc -Wall -Wextra -std=c99 -O2 -g -c src/codegen.c -o obj/codegen.o

REM Link executable
echo Linking executable...
gcc obj/main.o obj/lexer.o obj/parser.o obj/typecheck.o obj/ir.o obj/codegen.o -o bin/ashc.exe -lm

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Compiler: bin/ashc.exe
) else (
    echo Build failed!
    exit /b 1
)

echo.
echo A# Compiler v0.1.0 - Research Language with Formal Verification
echo Usage: bin\ashc.exe [options] <input_file>
echo.
echo Options:
echo   -o <file>     Output file (default: a.out)
echo   -v            Verbose output
echo   -d            Debug mode
echo   --verify      Enable formal verification
echo   -h, --help    Show help
echo.
echo Example:
echo   bin\ashc.exe -o program examples\hello_world.ash
