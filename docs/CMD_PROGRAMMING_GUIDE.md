# A# Programming in CMD
## Complete Command Prompt Development Guide

Yes! You can absolutely program A# directly in the Windows Command Prompt (CMD). This guide shows you how to set up a complete CMD-based development environment for A#.

## Table of Contents

1. [Quick Start](#quick-start)
2. [CMD Commands](#cmd-commands)
3. [Creating Projects](#creating-projects)
4. [Editing Code](#editing-code)
5. [Compiling and Running](#compiling-and-running)
6. [AI/ML Programming](#aiml-programming)
7. [Advanced CMD Usage](#advanced-cmd-usage)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Open Command Prompt
```cmd
# Press Win + R, type "cmd", press Enter
# Or search "Command Prompt" in Start menu
```

### 2. Navigate to A# Directory
```cmd
cd C:\Users\MEHR\OneDrive\Documents\Desktop\A#
```

### 3. Create Your First Project
```cmd
cmd_new hello_world
```

### 4. Edit Your Code
```cmd
cmd_edit cmd_projects\hello_world.ash
```

### 5. Compile and Run
```cmd
cmd_compile cmd_projects\hello_world.ash
```

## CMD Commands

### **Available Commands**

| Command | Description | Example |
|---------|-------------|---------|
| `cmd_help` | Show help information | `cmd_help` |
| `cmd_new name` | Create new A# project | `cmd_new my_app` |
| `cmd_ai name` | Create AI/ML project | `cmd_ai neural_net` |
| `cmd_edit file` | Edit A# file in notepad | `cmd_edit my_app.ash` |
| `cmd_compile file` | Compile and run A# program | `cmd_compile my_app.ash` |

### **Command Details**

#### **cmd_help**
```cmd
cmd_help
```
Shows all available commands and usage examples.

#### **cmd_new**
```cmd
cmd_new project_name
```
Creates a new basic A# project with template code.

#### **cmd_ai**
```cmd
cmd_ai project_name
```
Creates a new A# AI/ML project with neural network template.

#### **cmd_edit**
```cmd
cmd_edit filename.ash
```
Opens the specified A# file in Notepad for editing.

#### **cmd_compile**
```cmd
cmd_compile filename.ash
```
Compiles the A# file and runs the resulting executable.

## Creating Projects

### 1. Basic Project
```cmd
# Create a new basic A# project
cmd_new my_first_app

# This creates: cmd_projects\my_first_app.ash
```

### 2. AI/ML Project
```cmd
# Create a new AI project
cmd_ai my_neural_network

# This creates: cmd_projects\my_neural_network.ash
```

### 3. Project Structure
```
cmd_projects/
├── my_first_app.ash
├── my_neural_network.ash
├── templates/
│   ├── project_template.ash
│   └── ai_template.ash
└── README.txt
```

## Editing Code

### 1. Using Notepad (Built-in)
```cmd
# Edit existing file
cmd_edit cmd_projects\my_app.ash

# Edit any A# file
cmd_edit examples\hello_world.ash
```

### 2. Using Other Editors
```cmd
# Using VS Code (if installed)
code cmd_projects\my_app.ash

# Using Vim (if installed)
vim cmd_projects\my_app.ash

# Using any text editor
notepad++ cmd_projects\my_app.ash
```

### 3. Quick Edit Commands
```cmd
# Create and edit new file
echo mod test { > test.ash
echo     fn main() -^> i32 @ Pure { >> test.ash
echo         println("Hello from CMD!"); >> test.ash
echo         return 0; >> test.ash
echo     } >> test.ash
echo } >> test.ash

# Edit the file
cmd_edit test.ash
```

## Compiling and Running

### 1. Basic Compilation
```cmd
# Compile and run
cmd_compile cmd_projects\my_app.ash

# This will:
# 1. Compile my_app.ash to my_app.exe
# 2. Run my_app.exe
# 3. Show output
```

### 2. Manual Compilation
```cmd
# Compile only
.\bin\ashc.exe cmd_projects\my_app.ash -o my_app.exe

# Run manually
.\my_app.exe
```

### 3. Compilation Options
```cmd
# Verbose compilation
.\bin\ashc.exe -v cmd_projects\my_app.ash -o my_app.exe

# Debug compilation
.\bin\ashc.exe -g cmd_projects\my_app.ash -o my_app_debug.exe

# Compile with ML support
.\bin\ashc.exe --ml cmd_projects\ai_app.ash -o ai_app.exe
```

## AI/ML Programming

### 1. Create AI Project
```cmd
cmd_ai my_ai_project
```

### 2. Edit AI Code
```cmd
cmd_edit cmd_projects\my_ai_project.ash
```

### 3. AI Project Template
The AI template includes:
```a#
mod ai_project {
    use std::ml::*;
    
    fn main() -> () @ Resource {
        println("A# AI Project in CMD");
        
        // Create neural network
        let model = create_mlp([784, 128, 10], 3, ACTIVATION_RELU, 0.001);
        println("Neural network created!");
    }
}
```

### 4. Compile AI Project
```cmd
cmd_compile cmd_projects\my_ai_project.ash
```

## Advanced CMD Usage

### 1. Batch Scripts for A#
```cmd
# Create a build script
echo @echo off > build_all.bat
echo echo Building all A# projects... >> build_all.bat
echo .\bin\ashc.exe cmd_projects\*.ash -o bin\ >> build_all.bat
echo echo Build complete! >> build_all.bat

# Run the build script
build_all.bat
```

### 2. Project Management
```cmd
# List all projects
dir cmd_projects\*.ash

# Create project directory
mkdir cmd_projects\advanced_ai
cmd_ai advanced_ai\main

# Copy project
copy cmd_projects\template.ash cmd_projects\new_project.ash
```

### 3. Debugging in CMD
```cmd
# Compile with debug info
.\bin\ashc.exe -g cmd_projects\my_app.ash -o my_app_debug.exe

# Run with error checking
.\my_app_debug.exe
if %errorlevel% neq 0 (
    echo Program failed with error code %errorlevel%
)
```

### 4. File Management
```cmd
# Create backup
copy cmd_projects\my_app.ash cmd_projects\my_app_backup.ash

# Compare files
fc cmd_projects\my_app.ash cmd_projects\my_app_backup.ash

# Delete compiled files
del *.exe
```

## Complete CMD Workflow

### 1. Daily Development
```cmd
# Start CMD
cmd

# Navigate to A# directory
cd C:\Users\MEHR\OneDrive\Documents\Desktop\A#

# Create new project
cmd_new daily_project

# Edit code
cmd_edit cmd_projects\daily_project.ash

# Compile and test
cmd_compile cmd_projects\daily_project.ash

# Make changes and repeat
cmd_edit cmd_projects\daily_project.ash
cmd_compile cmd_projects\daily_project.ash
```

### 2. AI Development Workflow
```cmd
# Create AI project
cmd_ai neural_network

# Edit AI code
cmd_edit cmd_projects\neural_network.ash

# Compile with ML support
.\bin\ashc.exe --ml cmd_projects\neural_network.ash -o neural_network.exe

# Run AI program
.\neural_network.exe
```

### 3. Project Organization
```cmd
# Create project structure
mkdir cmd_projects\my_ai_suite
mkdir cmd_projects\my_ai_suite\models
mkdir cmd_projects\my_ai_suite\data
mkdir cmd_projects\my_ai_suite\utils

# Create main project
cmd_ai my_ai_suite\main

# Create utility modules
cmd_new my_ai_suite\utils\math_utils
cmd_new my_ai_suite\utils\data_loader
```

## Troubleshooting

### 1. Common Issues

**Issue**: `'cmd_new' is not recognized`
```cmd
# Solution: Make sure you're in the A# directory
cd C:\Users\MEHR\OneDrive\Documents\Desktop\A#
```

**Issue**: `'ashc' is not recognized`
```cmd
# Solution: Use full path
.\bin\ashc.exe your_file.ash -o output.exe
```

**Issue**: Compilation errors
```cmd
# Solution: Check syntax and use verbose mode
.\bin\ashc.exe -v your_file.ash -o output.exe
```

### 2. CMD Tips

**Enable Command History**:
```cmd
# Press F7 to see command history
# Use ↑↓ arrows to navigate history
```

**Tab Completion**:
```cmd
# Type first few letters and press Tab
cmd_edit cmd_proj<Tab>
# Expands to: cmd_edit cmd_projects\
```

**Multiple Commands**:
```cmd
# Run multiple commands
cmd_new app1 && cmd_new app2 && cmd_compile cmd_projects\app1.ash
```

### 3. Performance Tips

**Quick Compilation**:
```cmd
# Create a quick compile alias
doskey qc=.\bin\ashc.exe $1 -o $2.exe && $2.exe
# Usage: qc my_file my_output
```

**Batch Operations**:
```cmd
# Compile all projects
for %f in (cmd_projects\*.ash) do .\bin\ashc.exe "%f" -o "%~nf.exe"
```

## Examples

### 1. Hello World in CMD
```cmd
# Create project
cmd_new hello

# Edit (opens in notepad)
cmd_edit cmd_projects\hello.ash

# Compile and run
cmd_compile cmd_projects\hello.ash
```

### 2. AI Neural Network in CMD
```cmd
# Create AI project
cmd_ai neural_net

# Edit AI code
cmd_edit cmd_projects\neural_net.ash

# Compile with ML support
.\bin\ashc.exe --ml cmd_projects\neural_net.ash -o neural_net.exe

# Run AI program
.\neural_net.exe
```

### 3. Advanced CMD Script
```cmd
# Create development script
echo @echo off > dev.bat
echo echo A# Development Environment >> dev.bat
echo echo ========================== >> dev.bat
echo echo 1. Create project >> dev.bat
echo echo 2. Edit project >> dev.bat
echo echo 3. Compile project >> dev.bat
echo echo 4. Run project >> dev.bat
echo set /p choice="Enter choice (1-4): " >> dev.bat
echo if %%choice%%==1 cmd_new %%2 >> dev.bat
echo if %%choice%%==2 cmd_edit cmd_projects\%%2.ash >> dev.bat
echo if %%choice%%==3 cmd_compile cmd_projects\%%2.ash >> dev.bat
echo if %%choice%%==4 .\%%2.exe >> dev.bat

# Run development script
dev.bat
```

## Conclusion

CMD provides a powerful, lightweight environment for A# development:

- **Simple Setup** - No IDE required
- **Fast Compilation** - Direct command-line compilation
- **Full Control** - Complete control over build process
- **AI/ML Ready** - Built-in support for AI programming
- **Portable** - Works on any Windows system

**Start programming A# in CMD today!** 🚀

---

**A# - Where Safety Meets AI, and CMD Meets Power!** 🖥️✨
