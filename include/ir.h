#ifndef IR_H
#define IR_H

#include "parser.h"
#include "typecheck.h"
#include <stdbool.h>

typedef enum {
    IR_MODULE,
    IR_FUNCTION,
    IR_BLOCK,
    IR_BASIC_BLOCK,
    IR_INSTRUCTION,
    IR_PARAMETER,
    IR_LOCAL_VAR,
    IR_GLOBAL_VAR,
    IR_CONSTANT,
    IR_TYPE_DEF,
    IR_STRUCT_DEF,
    IR_ENUM_DEF,
    IR_TRAIT_DEF
} IRNodeType;

typedef enum {
    INST_ADD,
    INST_SUB,
    INST_MUL,
    INST_DIV,
    INST_MOD,
    INST_EQUAL,
    INST_NOT_EQUAL,
    INST_LESS,
    INST_LESS_EQUAL,
    INST_GREATER,
    INST_GREATER_EQUAL,
    INST_AND,
    INST_OR,
    INST_NOT,
    INST_LOAD,
    INST_STORE,
    INST_ALLOCA,
    INST_CALL,
    INST_RETURN,
    INST_BRANCH,
    INST_COND_BRANCH,
    INST_PHI,
    INST_MOVE,
    INST_BORROW,
    INST_DEREF,
    INST_FIELD_ACCESS,
    INST_INDEX_ACCESS,
    INST_CONSTANT,
    INST_ALLOC,
    INST_DEALLOC,
    INST_REGION_ENTER,
    INST_REGION_EXIT
} InstructionType;

typedef struct IRNode IRNode;
typedef struct IRModule IRModule;
typedef struct IRFunction IRFunction;
typedef struct IRBlock IRBlock;
typedef struct IRBasicBlock IRBasicBlock;
typedef struct IRInstruction IRInstruction;
typedef struct IRParameter IRParameter;
typedef struct IRLocalVar IRLocalVar;
typedef struct IRGlobalVar IRGlobalVar;
typedef struct IRConstant IRConstant;
typedef struct IRTypeDef IRTypeDef;

struct IRNode {
    IRNodeType type;
    uint32_t line;
    uint32_t column;
    
    union {
        struct {
            IRNode** functions;
            uint32_t function_count;
            IRNode** globals;
            uint32_t global_count;
            IRNode** type_defs;
            uint32_t type_def_count;
        } module;
        
        struct {
            char* name;
            Type* signature;
            IRNode** parameters;
            uint32_t parameter_count;
            IRNode** basic_blocks;
            uint32_t basic_block_count;
            IRNode* entry_block;
        } function;
        
        struct {
            IRNode** instructions;
            uint32_t instruction_count;
        } basic_block;
        
        struct {
            InstructionType inst_type;
            IRNode** operands;
            uint32_t operand_count;
            Type* result_type;
            char* result_name;
        } instruction;
        
        struct {
            char* name;
            Type* type;
            bool is_mutable;
        } parameter;
        
        struct {
            char* name;
            Type* type;
            bool is_mutable;
            bool is_owned;
        } local_var;
        
        struct {
            char* name;
            Type* type;
            bool is_mutable;
            IRNode* initializer;
        } global_var;
        
        struct {
            Type* type;
            union {
                int64_t int_value;
                double float_value;
                bool bool_value;
                char* string_value;
            } value;
        } constant;
        
        struct {
            char* name;
            Type* type;
        } type_def;
    } data;
};

struct IRModule {
    IRNode* root;
    uint32_t node_count;
};

// IR creation functions
IRModule* create_ir_module(void);
IRNode* create_ir_function(const char* name, Type* signature);
IRNode* create_ir_basic_block(const char* name);
IRNode* create_ir_instruction(InstructionType type, Type* result_type);
IRNode* create_ir_parameter(const char* name, Type* type, bool is_mutable);
IRNode* create_ir_local_var(const char* name, Type* type, bool is_mutable, bool is_owned);
IRNode* create_ir_global_var(const char* name, Type* type, bool is_mutable);
IRNode* create_ir_constant(Type* type, const void* value);
IRNode* create_ir_type_def(const char* name, Type* type);

// IR manipulation functions
void add_ir_function(IRModule* module, IRNode* function);
void add_ir_global(IRModule* module, IRNode* global);
void add_ir_type_def(IRModule* module, IRNode* type_def);
void add_ir_parameter(IRNode* function, IRNode* parameter);
void add_ir_basic_block(IRNode* function, IRNode* basic_block);
void add_ir_instruction(IRNode* basic_block, IRNode* instruction);
void add_ir_operand(IRNode* instruction, IRNode* operand);

// IR generation functions
IRModule* generate_ir(const AST* ast, TypeContext* ctx);
IRNode* generate_ir_function(const ASTNode* function, TypeContext* ctx);
IRNode* generate_ir_expression(const ASTNode* expr, TypeContext* ctx);
IRNode* generate_ir_statement(const ASTNode* stmt, TypeContext* ctx);
IRNode* generate_ir_pattern(const ASTNode* pattern, TypeContext* ctx);

// IR optimization functions
bool optimize_ir(IRModule* module);
bool optimize_function(IRNode* function);
bool optimize_basic_block(IRNode* basic_block);
bool optimize_instruction(IRNode* instruction);

// IR analysis functions
bool analyze_ir(IRModule* module);
bool analyze_function(IRNode* function);
bool analyze_basic_block(IRNode* basic_block);
bool analyze_instruction(IRNode* instruction);

// IR utility functions
void free_ir_module(IRModule* module);
void free_ir_node(IRNode* node);
void print_ir(const IRModule* module);
void print_ir_node(const IRNode* node);
const char* instruction_type_to_string(InstructionType type);

#endif // IR_H
