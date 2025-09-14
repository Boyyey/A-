#ifndef PARSER_H
#define PARSER_H

#include "lexer.h"
#include <stdbool.h>

typedef enum {
    NODE_PROGRAM,
    NODE_MODULE,
    NODE_FUNCTION,
    NODE_TYPE_DECL,
    NODE_ENUM_DECL,
    NODE_STRUCT_DECL,
    NODE_TRAIT_DECL,
    NODE_IMPL_BLOCK,
    NODE_ACTOR_DECL,
    NODE_MESSAGE_DECL,
    NODE_BLOCK,
    NODE_LET_STMT,
    NODE_EXPR_STMT,
    NODE_RETURN_STMT,
    NODE_IF_EXPR,
    NODE_MATCH_EXPR,
    NODE_LOOP_EXPR,
    NODE_BREAK_EXPR,
    NODE_CONTINUE_EXPR,
    NODE_BINARY_EXPR,
    NODE_UNARY_EXPR,
    NODE_CALL_EXPR,
    NODE_FIELD_ACCESS,
    NODE_INDEX_ACCESS,
    NODE_LITERAL,
    NODE_IDENTIFIER,
    NODE_TYPE_EXPR,
    NODE_FUNCTION_TYPE,
    NODE_REF_TYPE,
    NODE_MUT_REF_TYPE,
    NODE_OWNED_TYPE,
    NODE_ARRAY_TYPE,
    NODE_TUPLE_TYPE,
    NODE_LIFETIME,
    NODE_PATTERN,
    NODE_WILDCARD_PATTERN,
    NODE_LITERAL_PATTERN,
    NODE_IDENTIFIER_PATTERN,
    NODE_TUPLE_PATTERN,
    NODE_STRUCT_PATTERN,
    NODE_ENUM_PATTERN,
    NODE_REF_PATTERN,
    NODE_MUT_REF_PATTERN,
    NODE_OWNED_PATTERN,
    NODE_EFFECTS,
    NODE_REGION_ANNOTATION,
    // Ownership and borrowing nodes
    NODE_REF,
    NODE_MUT_REF,
    NODE_DEREF,
    NODE_MOVE,
    NODE_BORROW,
    NODE_ASSIGN
} ASTNodeType;

typedef struct ASTNode ASTNode;

struct ASTNode {
    ASTNodeType type;
    uint32_t line;
    uint32_t column;
    
    union {
        // Program structure
        struct {
            ASTNode** modules;
            uint32_t module_count;
        } program;
        
        struct {
            char* name;
            ASTNode** items;
            uint32_t item_count;
        } module;
        
        struct {
            char* name;
            ASTNode** type_params;
            uint32_t type_param_count;
            ASTNode** params;
            uint32_t param_count;
            ASTNode* return_type;
            ASTNode* effects;
            ASTNode* body;
        } function;
        
        struct {
            char* name;
            ASTNode* type_expr;
        } parameter;
        
        struct {
            char* name;
            ASTNode** type_params;
            uint32_t type_param_count;
            ASTNode* type_expr;
        } type_decl;
        
        struct {
            char* name;
            ASTNode** type_params;
            uint32_t type_param_count;
            ASTNode** variants;
            uint32_t variant_count;
        } enum_decl;
        
        struct {
            char* name;
            ASTNode** type_params;
            uint32_t type_param_count;
            ASTNode** fields;
            uint32_t field_count;
        } struct_decl;
        
        struct {
            char* name;
            ASTNode** type_params;
            uint32_t type_param_count;
            ASTNode** methods;
            uint32_t method_count;
        } trait_decl;
        
        struct {
            ASTNode* trait;
            ASTNode* type;
            ASTNode** methods;
            uint32_t method_count;
        } impl_block;
        
        struct {
            char* name;
            ASTNode** messages;
            uint32_t message_count;
            ASTNode** fields;
            uint32_t field_count;
        } actor_decl;
        
        struct {
            char* name;
            ASTNode** params;
            uint32_t param_count;
            ASTNode* return_type;
        } message_decl;
        
        // Statements
        struct {
            ASTNode** statements;
            uint32_t statement_count;
            ASTNode* expression;
        } block;
        
        struct {
            ASTNode* pattern;
            ASTNode* type_expr;
            ASTNode* expression;
            bool is_mutable;
        } let_stmt;
        
        struct {
            ASTNode* expression;
        } expr_stmt;
        
        struct {
            ASTNode* expression;
        } return_stmt;
        
        // Expressions
        struct {
            ASTNode* condition;
            ASTNode* then_branch;
            ASTNode* else_branch;
        } if_expr;
        
        struct {
            ASTNode* expression;
            ASTNode** arms;
            uint32_t arm_count;
        } match_expr;
        
        struct {
            ASTNode* body;
        } loop_expr;
        
        struct {
            ASTNode* expression;
        } break_expr;
        
        struct {
            // No additional fields
        } continue_expr;
        
        struct {
            ASTNode* left;
            ASTNode* right;
            TokenType operator;
        } binary_expr;
        
        struct {
            ASTNode* operand;
            TokenType operator;
        } unary_expr;
        
        struct {
            ASTNode* callee;
            ASTNode** arguments;
            uint32_t argument_count;
        } call_expr;
        
        struct {
            ASTNode* object;
            char* field_name;
        } field_access;
        
        struct {
            ASTNode* object;
            ASTNode* index;
        } index_access;
        
        // Literals and identifiers
        struct {
            TokenType literal_type;
            union {
                int64_t int_value;
                double float_value;
                bool bool_value;
                char* string_value;
            } value;
        } literal;
        
        struct {
            char* name;
        } identifier;
        
        // Types
        struct {
            char* name;
            ASTNode** type_args;
            uint32_t type_arg_count;
        } type_expr;
        
        struct {
            ASTNode* param_type;
            ASTNode* return_type;
            ASTNode* effects;
        } function_type;
        
        struct {
            ASTNode* lifetime;
            ASTNode* type;
        } ref_type;
        
        struct {
            ASTNode* lifetime;
            ASTNode* type;
        } mut_ref_type;
        
        struct {
            ASTNode* type;
        } owned_type;
        
        struct {
            ASTNode* element_type;
            uint32_t size;
        } array_type;
        
        struct {
            ASTNode** types;
            uint32_t type_count;
        } tuple_type;
        
        struct {
            char* name;
        } lifetime;
        
        // Patterns
        struct {
            ASTNode* pattern;
        } wildcard_pattern;
        
        struct {
            ASTNode* literal;
        } literal_pattern;
        
        struct {
            char* name;
            ASTNode* type_expr;
        } identifier_pattern;
        
        struct {
            ASTNode** patterns;
            uint32_t pattern_count;
        } tuple_pattern;
        
        struct {
            char* name;
            ASTNode** field_patterns;
            uint32_t field_pattern_count;
        } struct_pattern;
        
        struct {
            char* name;
            ASTNode** field_patterns;
            uint32_t field_pattern_count;
        } enum_pattern;
        
        struct {
            ASTNode* pattern;
        } ref_pattern;
        
        struct {
            ASTNode* pattern;
        } mut_ref_pattern;
        
        struct {
            ASTNode* pattern;
        } owned_pattern;
        
        // Effects and regions
        struct {
            TokenType effect_type;
        } effects;
        
        struct {
            ASTNode* lifetime;
        } region_annotation;
    } data;
};

typedef struct {
    ASTNode* root;
    uint32_t node_count;
} AST;

// Parser functions
AST* parse_tokens(TokenStream* stream);
void free_ast(AST* ast);
void print_ast(const AST* ast);

// AST node creation
ASTNode* create_ast_node(ASTNodeType type, uint32_t line, uint32_t column);
void free_ast_node(ASTNode* node);

// Parsing functions
ASTNode* parse_program(TokenStream* stream);
ASTNode* parse_module(TokenStream* stream);
ASTNode* parse_function(TokenStream* stream);
ASTNode* parse_type_decl(TokenStream* stream);
ASTNode* parse_enum_decl(TokenStream* stream);
ASTNode* parse_struct_decl(TokenStream* stream);
ASTNode* parse_trait_decl(TokenStream* stream);
ASTNode* parse_impl_block(TokenStream* stream);
ASTNode* parse_actor_decl(TokenStream* stream);
ASTNode* parse_message_decl(TokenStream* stream);
ASTNode* parse_block(TokenStream* stream);
ASTNode* parse_statement(TokenStream* stream);
ASTNode* parse_expression(TokenStream* stream);
ASTNode* parse_type_expression(TokenStream* stream);
ASTNode* parse_pattern(TokenStream* stream);
ASTNode* parse_effects(TokenStream* stream);

// Utility functions
bool expect_token(TokenStream* stream, TokenType expected);
bool match_token(TokenStream* stream, TokenType type);
Token* consume_token(TokenStream* stream, TokenType type);
// These functions are declared in lexer.h

#endif // PARSER_H
