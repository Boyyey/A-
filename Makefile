# A# Compiler Makefile
# Research Language with Formal Verification

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
LDFLAGS = -lm

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
DOCS_DIR = docs
TESTS_DIR = tests
EXAMPLES_DIR = examples

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET = $(BIN_DIR)/ashc

# Default target
all: $(TARGET)

# Create directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build the compiler
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Install the compiler
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/ashc
	chmod +x /usr/local/bin/ashc

# Uninstall the compiler
uninstall:
	rm -f /usr/local/bin/ashc

# Run tests
test: $(TARGET)
	@echo "Running tests..."
	@if [ -d "$(TESTS_DIR)" ]; then \
		for test in $(TESTS_DIR)/*.ash; do \
			echo "Testing $$test..."; \
			$(TARGET) -o /tmp/test_$$(basename $$test .ash) $$test; \
		done; \
	fi

# Run examples
examples: $(TARGET)
	@echo "Running examples..."
	@if [ -d "$(EXAMPLES_DIR)" ]; then \
		for example in $(EXAMPLES_DIR)/*.ash; do \
			echo "Compiling $$example..."; \
			$(TARGET) -o /tmp/example_$$(basename $$example .ash) $$example; \
		done; \
	fi

# Debug build
debug: CFLAGS += -DDEBUG -g3 -O0
debug: $(TARGET)

# Release build
release: CFLAGS += -DNDEBUG -O3 -flto
release: $(TARGET)

# Static analysis
analyze:
	@echo "Running static analysis..."
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --enable=all --std=c99 $(SRC_DIR)/; \
	else \
		echo "cppcheck not found, skipping static analysis"; \
	fi

# Format code
format:
	@echo "Formatting code..."
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format -i $(SRC_DIR)/*.c $(SRC_DIR)/*.h; \
	else \
		echo "clang-format not found, skipping formatting"; \
	fi

# Generate documentation
docs:
	@echo "Generating documentation..."
	@if command -v doxygen >/dev/null 2>&1; then \
		doxygen Doxyfile; \
	else \
		echo "doxygen not found, skipping documentation generation"; \
	fi

# Package for distribution
package: clean release
	@echo "Creating package..."
	tar -czf ash-compiler-$(shell date +%Y%m%d).tar.gz \
		$(SRC_DIR)/ $(DOCS_DIR)/ Makefile README.md LICENSE

# Help
help:
	@echo "A# Compiler Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the compiler (default)"
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install the compiler to /usr/local/bin"
	@echo "  uninstall  - Remove the compiler from /usr/local/bin"
	@echo "  test       - Run test suite"
	@echo "  examples   - Compile example programs"
	@echo "  debug      - Build with debug symbols"
	@echo "  release    - Build optimized release version"
	@echo "  analyze    - Run static analysis"
	@echo "  format     - Format source code"
	@echo "  docs       - Generate documentation"
	@echo "  package    - Create distribution package"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CC         - C compiler (default: gcc)"
	@echo "  CFLAGS     - Compiler flags"
	@echo "  LDFLAGS    - Linker flags"

# Phony targets
.PHONY: all clean install uninstall test examples debug release analyze format docs package help

# Dependencies
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.c $(SRC_DIR)/lexer.h $(SRC_DIR)/parser.h $(SRC_DIR)/typecheck.h $(SRC_DIR)/ir.h $(SRC_DIR)/codegen.h
$(OBJ_DIR)/lexer.o: $(SRC_DIR)/lexer.c $(SRC_DIR)/lexer.h
$(OBJ_DIR)/parser.o: $(SRC_DIR)/parser.c $(SRC_DIR)/parser.h $(SRC_DIR)/lexer.h
$(OBJ_DIR)/typecheck.o: $(SRC_DIR)/typecheck.c $(SRC_DIR)/typecheck.h $(SRC_DIR)/parser.h
$(OBJ_DIR)/ir.o: $(SRC_DIR)/ir.c $(SRC_DIR)/ir.h $(SRC_DIR)/parser.h $(SRC_DIR)/typecheck.h
$(OBJ_DIR)/codegen.o: $(SRC_DIR)/codegen.c $(SRC_DIR)/codegen.h $(SRC_DIR)/ir.h
