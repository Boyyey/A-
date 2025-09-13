#ifndef CODEGEN_H
#define CODEGEN_H

#include "ir.h"
#include <stdbool.h>

// Code generation functions
bool generate_code(const IRModule* module, const char* output_file);
bool generate_llvm_ir(const IRModule* module, const char* output_file);
bool generate_native_code(const IRModule* module, const char* output_file);

// LLVM IR generation
bool generate_llvm_module(const IRModule* module, const char* output_file);
bool generate_llvm_function(const IRNode* function, const char* output_file);
bool generate_llvm_basic_block(const IRNode* basic_block, const char* output_file);
bool generate_llvm_instruction(const IRNode* instruction, const char* output_file);

// Native code generation
bool generate_native_module(const IRModule* module, const char* output_file);
bool generate_native_function(const IRNode* function, const char* output_file);
bool generate_native_instruction(const IRNode* instruction, const char* output_file);

// Utility functions
const char* get_llvm_type_name(const Type* type);
const char* get_llvm_instruction_name(InstructionType type);
const char* get_native_instruction_name(InstructionType type);

#endif // CODEGEN_H
