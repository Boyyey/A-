@echo off
echo ========================================
echo A# Compiler Installation for MinGW64
echo ========================================

echo.
echo Setting up A# compiler for MinGW64...

REM Create directories
if not exist "bin" mkdir bin
if not exist "lib" mkdir lib
if not exist "include" mkdir include
if not exist "examples" mkdir examples

REM Build the compiler
echo Building A# compiler...
gcc -std=c99 -Wall -Wextra -O2 -g -c src/main.c -o obj/main.o
gcc -std=c99 -Wall -Wextra -O2 -g -c src/lexer.c -o obj/lexer.o
gcc -std=c99 -Wall -Wextra -O2 -g -c src/parser.c -o obj/parser.o
gcc -std=c99 -Wall -Wextra -O2 -g -c src/typecheck.c -o obj/typecheck.o
gcc -std=c99 -Wall -Wextra -O2 -g -c src/ir.c -o obj/ir.o
gcc -std=c99 -Wall -Wextra -O2 -g -c src/codegen.c -o obj/codegen.o

REM Link the compiler
echo Linking A# compiler...
gcc -o bin/ashc.exe obj/main.o obj/lexer.o obj/parser.o obj/typecheck.o obj/ir.o obj/codegen.o -lm

REM Create a simple hello world example
echo Creating hello world example...
echo mod hello_world { > examples/hello_world.ash
echo     fn main() -^> i32 @ Pure { >> examples/hello_world.ash
echo         let message: !String = "Hello, A# from MinGW64!".to_string(); >> examples/hello_world.ash
echo         println(message); >> examples/hello_world.ash
echo         return 0; >> examples/hello_world.ash
echo     } >> examples/hello_world.ash
echo } >> examples/hello_world.ash

REM Create a simple AI/ML example
echo Creating AI/ML example...
echo mod ai_demo { > examples/ai_demo.ash
echo     use std::ml::*; >> examples/ai_demo.ash
echo. >> examples/ai_demo.ash
echo     fn main() -^> () @ Resource { >> examples/ai_demo.ash
echo         println("🧠 A# AI Demo with MinGW64"); >> examples/ai_demo.ash
echo. >> examples/ai_demo.ash
echo         // Create a simple neural network >> examples/ai_demo.ash
echo         let model = create_mlp([784, 128, 10], 3, ACTIVATION_RELU, 0.001); >> examples/ai_demo.ash
echo         let optimizer = create_optimizer(OPTIMIZER_ADAM, "adam", 0.001); >> examples/ai_demo.ash
echo         let loss_fn = create_loss_function(LOSS_CROSSENTROPY, "crossentropy", null); >> examples/ai_demo.ash
echo. >> examples/ai_demo.ash
echo         println("Neural network created successfully!"); >> examples/ai_demo.ash
echo         println("Model parameters: 784 -^> 128 -^> 10"); >> examples/ai_demo.ash
echo         println("Optimizer: Adam with learning rate 0.001"); >> examples/ai_demo.ash
echo         println("Loss function: Cross Entropy"); >> examples/ai_demo.ash
echo. >> examples/ai_demo.ash
echo         // Simulate training >> examples/ai_demo.ash
echo         for epoch in 0..5 { >> examples/ai_demo.ash
echo             let loss = 1.0 - (epoch as f32) * 0.15; >> examples/ai_demo.ash
echo             println("Epoch {}: Loss = {:.3}", epoch + 1, loss); >> examples/ai_demo.ash
echo         } >> examples/ai_demo.ash
echo. >> examples/ai_demo.ash
echo         println("Training completed! Model ready for inference."); >> examples/ai_demo.ash
echo     } >> examples/ai_demo.ash
echo } >> examples/ai_demo.ash

REM Create a Vim configuration for A#
echo Creating Vim configuration...
if not exist "vim" mkdir vim
echo " A# syntax highlighting for Vim > vim/ash.vim
echo " Place this in your ~/.vim/syntax/ directory >> vim/ash.vim
echo. >> vim/ash.vim
echo if exists("b:current_syntax") >> vim/ash.vim
echo   finish >> vim/ash.vim
echo endif >> vim/ash.vim
echo. >> vim/ash.vim
echo " Keywords >> vim/ash.vim
echo syn keyword ashKeyword fn mod struct enum trait impl actor message >> vim/ash.vim
echo syn keyword ashKeyword let mut if else match for while loop break continue >> vim/ash.vim
echo syn keyword ashKeyword return use pub priv static const >> vim/ash.vim
echo syn keyword ashKeyword true false null >> vim/ash.vim
echo. >> vim/ash.vim
echo " Types >> vim/ash.vim
echo syn keyword ashType i32 i64 f32 f64 bool String Vec Option Result >> vim/ash.vim
echo syn keyword ashType Tensor Model NeuralNetwork Optimizer >> vim/ash.vim
echo. >> vim/ash.vim
echo " Effects >> vim/ash.vim
echo syn keyword ashEffect Pure IO Resource Concurrency >> vim/ash.vim
echo. >> vim/ash.vim
echo " Comments >> vim/ash.vim
echo syn match ashComment "//.*$" >> vim/ash.vim
echo syn region ashComment start="/\*" end="\*/" >> vim/ash.vim
echo. >> vim/ash.vim
echo " Strings >> vim/ash.vim
echo syn region ashString start=+"+ end=+"+ >> vim/ash.vim
echo. >> vim/ash.vim
echo " Numbers >> vim/ash.vim
echo syn match ashNumber "\<[0-9]\+\>" >> vim/ash.vim
echo syn match ashNumber "\<[0-9]\+\.[0-9]\+" >> vim/ash.vim
echo. >> vim/ash.vim
echo " Operators >> vim/ash.vim
echo syn match ashOperator "[+\-*/%=<>!&|^]" >> vim/ash.vim
echo syn match ashOperator "->" >> vim/ash.vim
echo syn match ashOperator "@" >> vim/ash.vim
echo. >> vim/ash.vim
echo " Highlighting >> vim/ash.vim
echo hi def link ashKeyword Keyword >> vim/ash.vim
echo hi def link ashType Type >> vim/ash.vim
echo hi def link ashEffect Special >> vim/ash.vim
echo hi def link ashComment Comment >> vim/ash.vim
echo hi def link ashString String >> vim/ash.vim
echo hi def link ashNumber Number >> vim/ash.vim
echo hi def link ashOperator Operator >> vim/ash.vim
echo. >> vim/ash.vim
echo let b:current_syntax = "ash" >> vim/ash.vim

REM Create a simple Makefile for MinGW64
echo Creating Makefile for MinGW64...
echo # A# Compiler Makefile for MinGW64 > Makefile.mingw64
echo CC = gcc >> Makefile.mingw64
echo CFLAGS = -std=c99 -Wall -Wextra -O2 -g >> Makefile.mingw64
echo LDFLAGS = -lm >> Makefile.mingw64
echo. >> Makefile.mingw64
echo SRCDIR = src >> Makefile.mingw64
echo OBJDIR = obj >> Makefile.mingw64
echo BINDIR = bin >> Makefile.mingw64
echo. >> Makefile.mingw64
echo SOURCES = $(SRCDIR)/main.c $(SRCDIR)/lexer.c $(SRCDIR)/parser.c $(SRCDIR)/typecheck.c $(SRCDIR)/ir.c $(SRCDIR)/codegen.c >> Makefile.mingw64
echo OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o) >> Makefile.mingw64
echo TARGET = $(BINDIR)/ashc.exe >> Makefile.mingw64
echo. >> Makefile.mingw64
echo all: $(TARGET) >> Makefile.mingw64
echo. >> Makefile.mingw64
echo $(TARGET): $(OBJECTS) >> Makefile.mingw64
echo 	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) >> Makefile.mingw64
echo. >> Makefile.mingw64
echo $(OBJDIR)/%.o: $(SRCDIR)/%.c >> Makefile.mingw64
echo 	@mkdir -p $(OBJDIR) >> Makefile.mingw64
echo 	$(CC) $(CFLAGS) -c $< -o $@ >> Makefile.mingw64
echo. >> Makefile.mingw64
echo clean: >> Makefile.mingw64
echo 	rm -rf $(OBJDIR) $(BINDIR) >> Makefile.mingw64
echo. >> Makefile.mingw64
echo install: $(TARGET) >> Makefile.mingw64
echo 	@echo "Installing A# compiler..." >> Makefile.mingw64
echo 	@echo "Add $(CURDIR)/bin to your PATH" >> Makefile.mingw64
echo. >> Makefile.mingw64
echo .PHONY: all clean install >> Makefile.mingw64

echo.
echo ========================================
echo A# Compiler Installation Complete!
echo ========================================
echo.
echo To use A# with MinGW64:
echo.
echo 1. Add bin/ to your PATH:
echo    export PATH=$PATH:$(pwd)/bin
echo.
echo 2. Compile A# programs:
echo    ashc examples/hello_world.ash -o hello.exe
echo    ashc examples/ai_demo.ash -o ai_demo.exe
echo.
echo 3. Run your programs:
echo    ./hello.exe
echo    ./ai_demo.exe
echo.
echo 4. For Vim users:
echo    Copy vim/ash.vim to ~/.vim/syntax/
echo    Add "au BufNewFile,BufRead *.ash set filetype=ash" to ~/.vimrc
echo.
echo 5. Use the Makefile:
echo    make -f Makefile.mingw64
echo    make -f Makefile.mingw64 clean
echo.
echo Happy programming with A#! 🚀
echo.
