@echo off 
if "%1"=="" ( 
    echo Usage: cmd_ai project_name 
    exit /b 1 
) 
echo Creating new A# AI project: %1 
copy cmd_projects\templates\ai_template.ash cmd_projects\%1.ash 
echo AI project created: cmd_projects\%1.ash 
echo Edit with: cmd_edit cmd_projects\%1.ash 
echo Compile with: cmd_compile cmd_projects\%1.ash 
