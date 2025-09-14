@echo off 
if "%1"=="" ( 
    echo Usage: cmd_new project_name 
    exit /b 1 
) 
echo Creating new A# project: %1 
copy cmd_projects\templates\project_template.ash cmd_projects\%1.ash 
echo Project created: cmd_projects\%1.ash 
echo Edit with: cmd_edit cmd_projects\%1.ash 
echo Compile with: cmd_compile cmd_projects\%1.ash 
