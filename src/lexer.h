#ifndef LEXER_H
#define LEXER_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    // Literals
    TOKEN_INTEGER,
    TOKEN_FLOAT,
    TOKEN_STRING,
    TOKEN_BOOLEAN,
    TOKEN_IDENTIFIER,
    
    // Keywords
    TOKEN_FN,
    TOKEN_LET,
    TOKEN_MUT,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_MATCH,
    TOKEN_LOOP,
    TOKEN_BREAK,
    TOKEN_CONTINUE,
    TOKEN_RETURN,
    TOKEN_TYPE,
    TOKEN_ENUM,
    TOKEN_STRUCT,
    TOKEN_TRAIT,
    TOKEN_IMPL,
    TOKEN_MOD,
    TOKEN_USE,
    TOKEN_PUB,
    TOKEN_ACTOR,
    TOKEN_MESSAGE,
    TOKEN_SPAWN,
    TOKEN_CHANNEL,
    TOKEN_ASYNC,
    TOKEN_AWAIT,
    
    // Operators
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_MODULO,
    TOKEN_ASSIGN,
    TOKEN_EQUAL,
    TOKEN_NOT_EQUAL,
    TOKEN_LESS,
    TOKEN_LESS_EQUAL,
    TOKEN_GREATER,
    TOKEN_GREATER_EQUAL,
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_NOT,
    TOKEN_AMPERSAND,
    TOKEN_PIPE,
    TOKEN_CARET,
    TOKEN_TILDE,
    TOKEN_SHIFT_LEFT,
    TOKEN_SHIFT_RIGHT,
    
    // Ownership and borrowing
    TOKEN_MOVE,
    TOKEN_BORROW,
    TOKEN_DEREF,
    TOKEN_REF,
    TOKEN_MUT_REF,
    
    // Punctuation
    TOKEN_LEFT_PAREN,
    TOKEN_RIGHT_PAREN,
    TOKEN_LEFT_BRACKET,
    TOKEN_RIGHT_BRACKET,
    TOKEN_LEFT_BRACE,
    TOKEN_RIGHT_BRACE,
    TOKEN_COMMA,
    TOKEN_SEMICOLON,
    TOKEN_COLON,
    TOKEN_DOUBLE_COLON,
    TOKEN_DOT,
    TOKEN_ARROW,
    TOKEN_FAT_ARROW,
    TOKEN_QUESTION,
    TOKEN_EXCLAMATION,
    TOKEN_AT,
    TOKEN_HASH,
    TOKEN_DOLLAR,
    
    // Regions and lifetimes
    TOKEN_LIFETIME,
    TOKEN_STATIC_LIFETIME,
    
    // Effects
    TOKEN_IO_EFFECT,
    TOKEN_CONCURRENCY_EFFECT,
    TOKEN_RESOURCE_EFFECT,
    TOKEN_PURE_EFFECT,
    
    // Special
    TOKEN_EOF,
    TOKEN_ERROR
} TokenType;

typedef struct {
    TokenType type;
    char* lexeme;
    uint32_t line;
    uint32_t column;
    union {
        int64_t int_value;
        double float_value;
        bool bool_value;
        char* string_value;
        char* identifier;
        char* lifetime;
    } value;
} Token;

typedef struct {
    Token* tokens;
    uint32_t count;
    uint32_t capacity;
    uint32_t current;
} TokenStream;

// Lexer functions
TokenStream* lex_file(const char* filename);
TokenStream* lex_string(const char* source);
void free_token_stream(TokenStream* stream);
void print_tokens(const TokenStream* stream);

// Token stream functions
Token* peek_token(const TokenStream* stream);
Token* next_token(TokenStream* stream);
bool has_tokens(const TokenStream* stream);
void reset_token_stream(TokenStream* stream);

// Utility functions
const char* token_type_to_string(TokenType type);
bool is_keyword(const char* lexeme);
TokenType keyword_to_token_type(const char* lexeme);

#endif // LEXER_H
