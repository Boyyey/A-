#ifndef TYPECHECK_H
#define TYPECHECK_H

#include "parser.h"
#include <stdbool.h>

typedef enum {
    TYPE_PRIMITIVE,
    TYPE_FUNCTION,
    TYPE_REF,
    TYPE_MUT_REF,
    TYPE_OWNED,
    TYPE_ARRAY,
    TYPE_TUPLE,
    TYPE_ENUM,
    TYPE_STRUCT,
    TYPE_TRAIT,
    TYPE_TYPE_VAR,
    TYPE_FORALL,
    TYPE_EXISTS
} TypeKind;

typedef enum {
    EFFECT_IO,
    EFFECT_CONCURRENCY,
    EFFECT_RESOURCE,
    EFFECT_PURE
} EffectType;

typedef struct Type Type;
typedef struct TypeContext TypeContext;
typedef struct TypeVar TypeVar;
typedef struct EffectSet EffectSet;

struct Type {
    TypeKind kind;
    uint32_t line;
    uint32_t column;
    
    union {
        struct {
            TokenType primitive_type;
        } primitive;
        
        struct {
            Type** param_types;
            uint32_t param_count;
            Type* return_type;
            EffectSet* effects;
        } function;
        
        struct {
            char* lifetime;
            Type* type;
        } ref;
        
        struct {
            char* lifetime;
            Type* type;
        } mut_ref;
        
        struct {
            Type* type;
        } owned;
        
        struct {
            Type* element_type;
            uint32_t size;
        } array;
        
        struct {
            Type** types;
            uint32_t type_count;
        } tuple;
        
        struct {
            char* name;
            Type** type_args;
            uint32_t type_arg_count;
        } enum_type;
        
        struct {
            char* name;
            Type** type_args;
            uint32_t type_arg_count;
        } struct_type;
        
        struct {
            char* name;
            Type** type_args;
            uint32_t type_arg_count;
        } trait_type;
        
        struct {
            TypeVar* var;
        } type_var;
        
        struct {
            char** type_params;
            uint32_t type_param_count;
            Type* type;
        } forall;
        
        struct {
            char** type_params;
            uint32_t type_param_count;
            Type* type;
        } exists;
    } data;
};

struct TypeVar {
    char* name;
    Type* constraint;
    bool is_unified;
    Type* unified_type;
};

struct EffectSet {
    EffectType* effects;
    uint32_t effect_count;
    bool is_pure;
};

struct TypeContext {
    TypeVar** type_vars;
    uint32_t type_var_count;
    uint32_t type_var_capacity;
    
    Type** type_definitions;
    uint32_t type_def_count;
    uint32_t type_def_capacity;
    
    EffectSet* current_effects;
    char** current_lifetimes;
    uint32_t lifetime_count;
    uint32_t lifetime_capacity;
};

// Type context functions
TypeContext* create_type_context(void);
void free_type_context(TypeContext* ctx);
TypeVar* create_type_var(const char* name, Type* constraint);
void add_type_var(TypeContext* ctx, TypeVar* var);
Type* lookup_type_var(TypeContext* ctx, const char* name);
void add_type_definition(TypeContext* ctx, const char* name, Type* type);
Type* lookup_type_definition(TypeContext* ctx, const char* name);

// Type creation functions
Type* create_primitive_type(TokenType primitive_type);
Type* create_function_type(Type** param_types, uint32_t param_count, 
                          Type* return_type, EffectSet* effects);
Type* create_ref_type(const char* lifetime, Type* type);
Type* create_mut_ref_type(const char* lifetime, Type* type);
Type* create_owned_type(Type* type);
Type* create_array_type(Type* element_type, uint32_t size);
Type* create_tuple_type(Type** types, uint32_t type_count);
Type* create_enum_type(const char* name, Type** type_args, uint32_t type_arg_count);
Type* create_struct_type(const char* name, Type** type_args, uint32_t type_arg_count);
Type* create_trait_type(const char* name, Type** type_args, uint32_t type_arg_count);
Type* create_type_var_type(TypeVar* var);
Type* create_forall_type(char** type_params, uint32_t type_param_count, Type* type);
Type* create_exists_type(char** type_params, uint32_t type_param_count, Type* type);

// Type operations
void free_type(Type* type);
Type* copy_type(const Type* type);
bool type_equals(const Type* a, const Type* b);
bool type_unify(Type* a, Type* b);
Type* type_substitute(Type* type, const char* var_name, Type* replacement);
Type* type_instantiate(Type* type, Type** type_args, uint32_t type_arg_count);

// Effect functions
EffectSet* create_effect_set(void);
void free_effect_set(EffectSet* effects);
void add_effect(EffectSet* effects, EffectType effect);
bool has_effect(const EffectSet* effects, EffectType effect);
bool effect_subset(const EffectSet* a, const EffectSet* b);
EffectSet* effect_union(const EffectSet* a, const EffectSet* b);
EffectSet* effect_intersection(const EffectSet* a, const EffectSet* b);

// Type checking functions
bool typecheck_ast(const AST* ast, TypeContext* ctx);
bool typecheck_module(const ASTNode* module, TypeContext* ctx);
bool typecheck_function(const ASTNode* function, TypeContext* ctx);
bool typecheck_expression(const ASTNode* expr, TypeContext* ctx, Type* expected_type);
bool typecheck_pattern(const ASTNode* pattern, TypeContext* ctx, Type* expected_type);
bool typecheck_type_expression(const ASTNode* type_expr, TypeContext* ctx, Type** result);

// Expression type checking functions
Type* typecheck_literal(const ASTNode* literal, TypeContext* ctx);
Type* typecheck_identifier(const ASTNode* identifier, TypeContext* ctx);
Type* typecheck_binary_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_unary_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_call_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_field_access(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_index_access(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_if_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_match_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_loop_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_block_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_ref_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_mut_ref_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_deref_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_move_expression(const ASTNode* expr, TypeContext* ctx);
Type* typecheck_borrow_expression(const ASTNode* expr, TypeContext* ctx);

// Statement type checking functions
bool typecheck_statement(const ASTNode* stmt, TypeContext* ctx);
bool typecheck_let_statement(const ASTNode* stmt, TypeContext* ctx);
bool typecheck_expr_statement(const ASTNode* stmt, TypeContext* ctx);
bool typecheck_return_statement(const ASTNode* stmt, TypeContext* ctx);

// Declaration type checking functions
bool typecheck_type_declaration(const ASTNode* decl, TypeContext* ctx);
bool typecheck_enum_declaration(const ASTNode* decl, TypeContext* ctx);
bool typecheck_struct_declaration(const ASTNode* decl, TypeContext* ctx);
bool typecheck_trait_declaration(const ASTNode* decl, TypeContext* ctx);
bool typecheck_actor_declaration(const ASTNode* decl, TypeContext* ctx);

// Ownership checking functions
bool check_ownership_rules(const ASTNode* expr, TypeContext* ctx);
bool check_borrowing_rules(const ASTNode* expr, TypeContext* ctx);
bool check_lifetime_rules(const ASTNode* expr, TypeContext* ctx);
bool check_move_semantics(const ASTNode* expr, TypeContext* ctx);

// Region inference
bool infer_regions(const ASTNode* expr, TypeContext* ctx);
bool infer_lifetime(const ASTNode* expr, TypeContext* ctx, char** lifetime);

// Utility functions
const char* type_kind_to_string(TypeKind kind);
const char* effect_type_to_string(EffectType effect);
void print_type(const Type* type);
void print_effect_set(const EffectSet* effects);

#endif // TYPECHECK_H
