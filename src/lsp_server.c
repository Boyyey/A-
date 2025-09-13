#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"
#include "parser.h"
#include "typecheck.h"

// Language Server Protocol implementation for A#
typedef struct {
    int id;
    char* method;
    json_object* params;
} LSPRequest;

typedef struct {
    int id;
    json_object* result;
    json_object* error;
} LSPResponse;

typedef struct {
    int line;
    int character;
} Position;

typedef struct {
    Position start;
    Position end;
} Range;

typedef struct {
    Range range;
    int severity;
    char* message;
    char* source;
} Diagnostic;

typedef struct {
    char* label;
    char* kind;
    char* detail;
    char* documentation;
} CompletionItem;

// LSP Server functions
bool lsp_initialize(json_object* params);
bool lsp_initialized(void);
bool lsp_shutdown(void);
bool lsp_exit(void);

// Language features
json_object* lsp_text_document_did_open(json_object* params);
json_object* lsp_text_document_did_change(json_object* params);
json_object* lsp_text_document_did_close(json_object* params);
json_object* lsp_text_document_did_save(json_object* params);

// Diagnostics
json_object* lsp_text_document_publish_diagnostics(json_object* params);
Diagnostic* lsp_create_diagnostic(Range range, int severity, const char* message, const char* source);
void lsp_free_diagnostic(Diagnostic* diag);

// Completion
json_object* lsp_text_document_completion(json_object* params);
CompletionItem* lsp_create_completion_item(const char* label, const char* kind, const char* detail);
void lsp_free_completion_item(CompletionItem* item);

// Hover
json_object* lsp_text_document_hover(json_object* params);

// Definition
json_object* lsp_text_document_definition(json_object* params);

// References
json_object* lsp_text_document_references(json_object* params);

// Formatting
json_object* lsp_text_document_formatting(json_object* params);
json_object* lsp_text_document_range_formatting(json_object* params);

// Symbol information
json_object* lsp_text_document_document_symbol(json_object* params);
json_object* lsp_workspace_symbol(json_object* params);

// Code actions
json_object* lsp_text_document_code_action(json_object* params);

// Utility functions
Position lsp_parse_position(json_object* position);
Range lsp_parse_range(json_object* range);
json_object* lsp_create_position(int line, int character);
json_object* lsp_create_range(Position start, Position end);
json_object* lsp_create_diagnostic_json(Diagnostic* diag);
json_object* lsp_create_completion_item_json(CompletionItem* item);

// Main LSP server loop
int lsp_main(int argc, char* argv[]);

// A# specific functions
bool ash_parse_file(const char* filename, AST** ast);
bool ash_typecheck_file(const char* filename, TypeContext** ctx);
Diagnostic* ash_get_diagnostics(const char* filename, AST* ast, TypeContext* ctx);
CompletionItem* ash_get_completions(const char* filename, Position pos, AST* ast, TypeContext* ctx);
char* ash_get_hover_info(const char* filename, Position pos, AST* ast, TypeContext* ctx);
Position ash_get_definition(const char* filename, Position pos, AST* ast, TypeContext* ctx);
Position* ash_get_references(const char* filename, Position pos, AST* ast, TypeContext* ctx, int* count);
char* ash_format_document(const char* content);
char* ash_format_range(const char* content, Range range);
