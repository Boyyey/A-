@echo off 
echo A# Quick Start 
echo =============== 
echo. 
echo Building A# compiler... 
.\build\scripts\build_minimal.bat 
echo. 
echo Creating hello world example... 
.\tools\cmd\cmd_new.bat hello_world 
echo. 
echo Compiling and running hello world... 
.\tools\cmd\cmd_compile.bat projects\examples\hello_world.ash 
echo. 
echo A# is ready! Check projects\examples\ for your programs. 
