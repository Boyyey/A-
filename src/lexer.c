#include "lexer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#define INITIAL_CAPACITY 1024
#define GROWTH_FACTOR 2

static TokenStream* create_token_stream(void) {
    TokenStream* stream = malloc(sizeof(TokenStream));
    if (!stream) return NULL;
    
    stream->tokens = malloc(sizeof(Token) * INITIAL_CAPACITY);
    if (!stream->tokens) {
        free(stream);
        return NULL;
    }
    
    stream->count = 0;
    stream->capacity = INITIAL_CAPACITY;
    stream->current = 0;
    return stream;
}

static bool grow_token_stream(TokenStream* stream) {
    uint32_t new_capacity = stream->capacity * GROWTH_FACTOR;
    Token* new_tokens = realloc(stream->tokens, sizeof(Token) * new_capacity);
    if (!new_tokens) return false;
    
    stream->tokens = new_tokens;
    stream->capacity = new_capacity;
    return true;
}

static void add_token(TokenStream* stream, TokenType type, const char* lexeme, 
                     uint32_t line, uint32_t column) {
    if (stream->count >= stream->capacity) {
        if (!grow_token_stream(stream)) {
            fprintf(stderr, "Error: Out of memory during tokenization\n");
            return;
        }
    }
    
    Token* token = &stream->tokens[stream->count++];
    token->type = type;
    token->line = line;
    token->column = column;
    
    if (lexeme) {
        token->lexeme = malloc(strlen(lexeme) + 1);
        if (token->lexeme) {
            strcpy(token->lexeme, lexeme);
        }
    } else {
        token->lexeme = NULL;
    }
    
    // Initialize value union
    memset(&token->value, 0, sizeof(token->value));
}

static bool is_alpha(char c) {
    return isalpha(c) || c == '_';
}

static bool is_alnum(char c) {
    return isalnum(c) || c == '_';
}

static bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\r';
}

static bool is_newline(char c) {
    return c == '\n';
}

static bool is_digit(char c) {
    return isdigit(c);
}

static bool is_hex_digit(char c) {
    return isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static bool is_octal_digit(char c) {
    return c >= '0' && c <= '7';
}

static bool is_binary_digit(char c) {
    return c == '0' || c == '1';
}

TokenType keyword_to_token_type(const char* lexeme) {
    static const struct {
        const char* keyword;
        TokenType type;
    } keywords[] = {
        {"fn", TOKEN_FN},
        {"let", TOKEN_LET},
        {"mut", TOKEN_MUT},
        {"if", TOKEN_IF},
        {"else", TOKEN_ELSE},
        {"match", TOKEN_MATCH},
        {"loop", TOKEN_LOOP},
        {"break", TOKEN_BREAK},
        {"continue", TOKEN_CONTINUE},
        {"return", TOKEN_RETURN},
        {"type", TOKEN_TYPE},
        {"enum", TOKEN_ENUM},
        {"struct", TOKEN_STRUCT},
        {"trait", TOKEN_TRAIT},
        {"impl", TOKEN_IMPL},
        {"mod", TOKEN_MOD},
        {"use", TOKEN_USE},
        {"pub", TOKEN_PUB},
        {"actor", TOKEN_ACTOR},
        {"message", TOKEN_MESSAGE},
        {"spawn", TOKEN_SPAWN},
        {"channel", TOKEN_CHANNEL},
        {"async", TOKEN_ASYNC},
        {"await", TOKEN_AWAIT},
        {"true", TOKEN_BOOLEAN},
        {"false", TOKEN_BOOLEAN},
        {"move", TOKEN_MOVE},
        {"borrow", TOKEN_BORROW},
        {"ref", TOKEN_REF},
        {"mut", TOKEN_MUT},
        {"static", TOKEN_STATIC_LIFETIME},
        {"IO", TOKEN_IO_EFFECT},
        {"Concurrency", TOKEN_CONCURRENCY_EFFECT},
        {"Resource", TOKEN_RESOURCE_EFFECT},
        {"Pure", TOKEN_PURE_EFFECT},
        {NULL, TOKEN_ERROR}
    };
    
    for (int i = 0; keywords[i].keyword; i++) {
        if (strcmp(lexeme, keywords[i].keyword) == 0) {
            return keywords[i].type;
        }
    }
    
    return TOKEN_IDENTIFIER;
}

static void scan_identifier(TokenStream* stream, const char* source, 
                           uint32_t* pos, uint32_t line, uint32_t column) {
    uint32_t start = *pos;
    
    while (source[*pos] && is_alnum(source[*pos])) {
        (*pos)++;
    }
    
    uint32_t length = *pos - start;
    char* lexeme = malloc(length + 1);
    if (!lexeme) return;
    
    strncpy(lexeme, source + start, length);
    lexeme[length] = '\0';
    
    TokenType type = keyword_to_token_type(lexeme);
    add_token(stream, type, lexeme, line, column);
    
    free(lexeme);
}

static void scan_number(TokenStream* stream, const char* source, 
                       uint32_t* pos, uint32_t line, uint32_t column) {
    uint32_t start = *pos;
    bool is_float = false;
    
    // Handle different number bases
    if (source[*pos] == '0' && *pos + 1 < strlen(source)) {
        char next = source[*pos + 1];
        if (next == 'x' || next == 'X') {
            // Hexadecimal
            *pos += 2;
            while (source[*pos] && is_hex_digit(source[*pos])) {
                (*pos)++;
            }
        } else if (next == 'o' || next == 'O') {
            // Octal
            *pos += 2;
            while (source[*pos] && is_octal_digit(source[*pos])) {
                (*pos)++;
            }
        } else if (next == 'b' || next == 'B') {
            // Binary
            *pos += 2;
            while (source[*pos] && is_binary_digit(source[*pos])) {
                (*pos)++;
            }
        }
    }
    
    // Decimal numbers
    while (source[*pos] && is_digit(source[*pos])) {
        (*pos)++;
    }
    
    // Float detection
    if (source[*pos] == '.') {
        is_float = true;
        (*pos)++;
        while (source[*pos] && is_digit(source[*pos])) {
            (*pos)++;
        }
    }
    
    // Scientific notation
    if (source[*pos] == 'e' || source[*pos] == 'E') {
        is_float = true;
        (*pos)++;
        if (source[*pos] == '+' || source[*pos] == '-') {
            (*pos)++;
        }
        while (source[*pos] && is_digit(source[*pos])) {
            (*pos)++;
        }
    }
    
    uint32_t length = *pos - start;
    char* lexeme = malloc(length + 1);
    if (!lexeme) return;
    
    strncpy(lexeme, source + start, length);
    lexeme[length] = '\0';
    
    if (is_float) {
        add_token(stream, TOKEN_FLOAT, lexeme, line, column);
    } else {
        add_token(stream, TOKEN_INTEGER, lexeme, line, column);
    }
    
    free(lexeme);
}

static void scan_string(TokenStream* stream, const char* source, 
                       uint32_t* pos, uint32_t line, uint32_t column) {
    (*pos)++; // Skip opening quote
    
    uint32_t start = *pos;
    while (source[*pos] && source[*pos] != '"') {
        if (source[*pos] == '\\' && source[*pos + 1]) {
            (*pos) += 2; // Skip escape sequence
        } else {
            (*pos)++;
        }
    }
    
    if (source[*pos] != '"') {
        add_token(stream, TOKEN_ERROR, "Unterminated string", line, column);
        return;
    }
    
    uint32_t length = *pos - start;
    char* lexeme = malloc(length + 1);
    if (!lexeme) return;
    
    strncpy(lexeme, source + start, length);
    lexeme[length] = '\0';
    
    add_token(stream, TOKEN_STRING, lexeme, line, column);
    (*pos)++; // Skip closing quote
    
    free(lexeme);
}

static void scan_lifetime(TokenStream* stream, const char* source, 
                         uint32_t* pos, uint32_t line, uint32_t column) {
    (*pos)++; // Skip opening quote
    
    uint32_t start = *pos;
    while (source[*pos] && is_alpha(source[*pos])) {
        (*pos)++;
    }
    
    uint32_t length = *pos - start;
    char* lexeme = malloc(length + 1);
    if (!lexeme) return;
    
    strncpy(lexeme, source + start, length);
    lexeme[length] = '\0';
    
    add_token(stream, TOKEN_LIFETIME, lexeme, line, column);
    
    free(lexeme);
}

TokenStream* lex_string(const char* source) {
    TokenStream* stream = create_token_stream();
    if (!stream) return NULL;
    
    uint32_t pos = 0;
    uint32_t line = 1;
    uint32_t column = 1;
    
    while (source[pos]) {
        char c = source[pos];
        
        if (is_whitespace(c)) {
            column++;
            pos++;
        } else if (is_newline(c)) {
            line++;
            column = 1;
            pos++;
        } else if (is_alpha(c)) {
            scan_identifier(stream, source, &pos, line, column);
            column += pos - (pos - strlen(stream->tokens[stream->count - 1].lexeme));
        } else if (is_digit(c)) {
            scan_number(stream, source, &pos, line, column);
            column += pos - (pos - strlen(stream->tokens[stream->count - 1].lexeme));
        } else if (c == '"') {
            scan_string(stream, source, &pos, line, column);
            column += pos - (pos - strlen(stream->tokens[stream->count - 1].lexeme));
        } else if (c == '\'') {
            scan_lifetime(stream, source, &pos, line, column);
            column += pos - (pos - strlen(stream->tokens[stream->count - 1].lexeme));
        } else {
            // Handle operators and punctuation
            switch (c) {
                case '+': add_token(stream, TOKEN_PLUS, "+", line, column); break;
                case '-': 
                    if (source[pos + 1] == '>') {
                        add_token(stream, TOKEN_ARROW, "->", line, column);
                        pos++;
                    } else {
                        add_token(stream, TOKEN_MINUS, "-", line, column);
                    }
                    break;
                case '*': add_token(stream, TOKEN_MULTIPLY, "*", line, column); break;
                case '/': add_token(stream, TOKEN_DIVIDE, "/", line, column); break;
                case '%': add_token(stream, TOKEN_MODULO, "%", line, column); break;
                case '=':
                    if (source[pos + 1] == '=') {
                        add_token(stream, TOKEN_EQUAL, "==", line, column);
                        pos++;
                    } else if (source[pos + 1] == '>') {
                        add_token(stream, TOKEN_FAT_ARROW, "=>", line, column);
                        pos++;
                    } else {
                        add_token(stream, TOKEN_ASSIGN, "=", line, column);
                    }
                    break;
                case '!':
                    if (source[pos + 1] == '=') {
                        add_token(stream, TOKEN_NOT_EQUAL, "!=", line, column);
                        pos++;
                    } else {
                        add_token(stream, TOKEN_NOT, "!", line, column);
                    }
                    break;
                case '<':
                    if (source[pos + 1] == '=') {
                        add_token(stream, TOKEN_LESS_EQUAL, "<=", line, column);
                        pos++;
                    } else if (source[pos + 1] == '<') {
                        add_token(stream, TOKEN_SHIFT_LEFT, "<<", line, column);
                        pos++;
                    } else {
                        add_token(stream, TOKEN_LESS, "<", line, column);
                    }
                    break;
                case '>':
                    if (source[pos + 1] == '=') {
                        add_token(stream, TOKEN_GREATER_EQUAL, ">=", line, column);
                        pos++;
                    } else if (source[pos + 1] == '>') {
                        add_token(stream, TOKEN_SHIFT_RIGHT, ">>", line, column);
                        pos++;
                    } else {
                        add_token(stream, TOKEN_GREATER, ">", line, column);
                    }
                    break;
                case '&':
                    if (source[pos + 1] == '&') {
                        add_token(stream, TOKEN_AND, "&&", line, column);
                        pos++;
                    } else {
                        add_token(stream, TOKEN_AMPERSAND, "&", line, column);
                    }
                    break;
                case '|':
                    if (source[pos + 1] == '|') {
                        add_token(stream, TOKEN_OR, "||", line, column);
                        pos++;
                    } else {
                        add_token(stream, TOKEN_PIPE, "|", line, column);
                    }
                    break;
                case '^': add_token(stream, TOKEN_CARET, "^", line, column); break;
                case '~': add_token(stream, TOKEN_TILDE, "~", line, column); break;
                case '(': add_token(stream, TOKEN_LEFT_PAREN, "(", line, column); break;
                case ')': add_token(stream, TOKEN_RIGHT_PAREN, ")", line, column); break;
                case '[': add_token(stream, TOKEN_LEFT_BRACKET, "[", line, column); break;
                case ']': add_token(stream, TOKEN_RIGHT_BRACKET, "]", line, column); break;
                case '{': add_token(stream, TOKEN_LEFT_BRACE, "{", line, column); break;
                case '}': add_token(stream, TOKEN_RIGHT_BRACE, "}", line, column); break;
                case ',': add_token(stream, TOKEN_COMMA, ",", line, column); break;
                case ';': add_token(stream, TOKEN_SEMICOLON, ";", line, column); break;
                case ':':
                    if (source[pos + 1] == ':') {
                        add_token(stream, TOKEN_DOUBLE_COLON, "::", line, column);
                        pos++;
                    } else {
                        add_token(stream, TOKEN_COLON, ":", line, column);
                    }
                    break;
                case '.': add_token(stream, TOKEN_DOT, ".", line, column); break;
                case '?': add_token(stream, TOKEN_QUESTION, "?", line, column); break;
                case '@': add_token(stream, TOKEN_AT, "@", line, column); break;
                case '#': add_token(stream, TOKEN_HASH, "#", line, column); break;
                case '$': add_token(stream, TOKEN_DOLLAR, "$", line, column); break;
                default:
                    add_token(stream, TOKEN_ERROR, "Unknown character", line, column);
                    break;
            }
            pos++;
            column++;
        }
    }
    
    add_token(stream, TOKEN_EOF, NULL, line, column);
    return stream;
}

TokenStream* lex_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s: %s\n", filename, strerror(errno));
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read file content
    char* source = malloc(size + 1);
    if (!source) {
        fclose(file);
        return NULL;
    }
    
    fread(source, 1, size, file);
    source[size] = '\0';
    fclose(file);
    
    TokenStream* stream = lex_string(source);
    free(source);
    return stream;
}

void free_token_stream(TokenStream* stream) {
    if (!stream) return;
    
    for (uint32_t i = 0; i < stream->count; i++) {
        free(stream->tokens[i].lexeme);
    }
    
    free(stream->tokens);
    free(stream);
}

void print_tokens(const TokenStream* stream) {
    printf("Tokens:\n");
    for (uint32_t i = 0; i < stream->count; i++) {
        const Token* token = &stream->tokens[i];
        printf("  %s", token_type_to_string(token->type));
        
        if (token->lexeme) {
            printf(" '%s'", token->lexeme);
        }
        
        printf(" at %u:%u\n", token->line, token->column);
    }
}

Token* peek_token(const TokenStream* stream) {
    if (stream->current >= stream->count) return NULL;
    return &stream->tokens[stream->current];
}

Token* next_token(TokenStream* stream) {
    if (stream->current >= stream->count) return NULL;
    return &stream->tokens[stream->current++];
}

bool has_tokens(const TokenStream* stream) {
    return stream->current < stream->count;
}

void reset_token_stream(TokenStream* stream) {
    stream->current = 0;
}

const char* token_type_to_string(TokenType type) {
    static const char* names[] = {
        "INTEGER", "FLOAT", "STRING", "BOOLEAN", "IDENTIFIER",
        "FN", "LET", "MUT", "IF", "ELSE", "MATCH", "LOOP", "BREAK", "CONTINUE", "RETURN",
        "TYPE", "ENUM", "STRUCT", "TRAIT", "IMPL", "MOD", "USE", "PUB",
        "ACTOR", "MESSAGE", "SPAWN", "CHANNEL", "ASYNC", "AWAIT",
        "PLUS", "MINUS", "MULTIPLY", "DIVIDE", "MODULO", "ASSIGN", "EQUAL", "NOT_EQUAL",
        "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL", "AND", "OR", "NOT",
        "AMPERSAND", "PIPE", "CARET", "TILDE", "SHIFT_LEFT", "SHIFT_RIGHT",
        "MOVE", "BORROW", "DEREF", "REF", "MUT_REF",
        "LEFT_PAREN", "RIGHT_PAREN", "LEFT_BRACKET", "RIGHT_BRACKET",
        "LEFT_BRACE", "RIGHT_BRACE", "COMMA", "SEMICOLON", "COLON", "DOUBLE_COLON",
        "DOT", "ARROW", "FAT_ARROW", "QUESTION", "EXCLAMATION", "AT", "HASH", "DOLLAR",
        "LIFETIME", "STATIC_LIFETIME",
        "IO_EFFECT", "CONCURRENCY_EFFECT", "RESOURCE_EFFECT", "PURE_EFFECT",
        "EOF", "ERROR"
    };
    
    return names[type];
}

bool is_keyword(const char* lexeme) {
    return keyword_to_token_type(lexeme) != TOKEN_IDENTIFIER;
}
