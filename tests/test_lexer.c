#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../src/lexer.h"

void test_lexer_basic() {
    printf("Testing basic lexer functionality...\n");
    
    const char* source = "fn main() -> i32 { return 42; }";
    TokenStream* stream = lex_string(source);
    
    assert(stream != NULL);
    assert(stream->count > 0);
    
    // Check first token
    assert(stream->tokens[0].type == TOKEN_FN);
    assert(strcmp(stream->tokens[0].lexeme, "fn") == 0);
    
    // Check identifier
    assert(stream->tokens[1].type == TOKEN_IDENTIFIER);
    assert(strcmp(stream->tokens[1].lexeme, "main") == 0);
    
    // Check parentheses
    assert(stream->tokens[2].type == TOKEN_LEFT_PAREN);
    assert(stream->tokens[3].type == TOKEN_RIGHT_PAREN);
    
    // Check arrow
    assert(stream->tokens[4].type == TOKEN_ARROW);
    
    // Check type
    assert(stream->tokens[5].type == TOKEN_IDENTIFIER);
    assert(strcmp(stream->tokens[5].lexeme, "i32") == 0);
    
    printf("✓ Basic lexer test passed\n");
    
    free_token_stream(stream);
}

void test_lexer_keywords() {
    printf("Testing keyword recognition...\n");
    
    const char* source = "let mut if else match loop break continue return";
    TokenStream* stream = lex_string(source);
    
    assert(stream != NULL);
    
    TokenType expected[] = {
        TOKEN_LET, TOKEN_MUT, TOKEN_IF, TOKEN_ELSE, 
        TOKEN_MATCH, TOKEN_LOOP, TOKEN_BREAK, 
        TOKEN_CONTINUE, TOKEN_RETURN
    };
    
    for (int i = 0; i < 9; i++) {
        assert(stream->tokens[i].type == expected[i]);
    }
    
    printf("✓ Keyword recognition test passed\n");
    
    free_token_stream(stream);
}

void test_lexer_numbers() {
    printf("Testing number parsing...\n");
    
    const char* source = "42 3.14 0x1A 0o777 0b1010";
    TokenStream* stream = lex_string(source);
    
    assert(stream != NULL);
    
    // Check integer
    assert(stream->tokens[0].type == TOKEN_INTEGER);
    assert(strcmp(stream->tokens[0].lexeme, "42") == 0);
    
    // Check float
    assert(stream->tokens[1].type == TOKEN_FLOAT);
    assert(strcmp(stream->tokens[1].lexeme, "3.14") == 0);
    
    // Check hex
    assert(stream->tokens[2].type == TOKEN_INTEGER);
    assert(strcmp(stream->tokens[2].lexeme, "0x1A") == 0);
    
    // Check octal
    assert(stream->tokens[3].type == TOKEN_INTEGER);
    assert(strcmp(stream->tokens[3].lexeme, "0o777") == 0);
    
    // Check binary
    assert(stream->tokens[4].type == TOKEN_INTEGER);
    assert(strcmp(stream->tokens[4].lexeme, "0b1010") == 0);
    
    printf("✓ Number parsing test passed\n");
    
    free_token_stream(stream);
}

void test_lexer_strings() {
    printf("Testing string parsing...\n");
    
    const char* source = "\"hello world\" \"escaped\\nstring\"";
    TokenStream* stream = lex_string(source);
    
    assert(stream != NULL);
    
    // Check first string
    assert(stream->tokens[0].type == TOKEN_STRING);
    assert(strcmp(stream->tokens[0].lexeme, "hello world") == 0);
    
    // Check second string with escape
    assert(stream->tokens[1].type == TOKEN_STRING);
    assert(strcmp(stream->tokens[1].lexeme, "escaped\\nstring") == 0);
    
    printf("✓ String parsing test passed\n");
    
    free_token_stream(stream);
}

void test_lexer_operators() {
    printf("Testing operator parsing...\n");
    
    const char* source = "+ - * / % = == != < <= > >= && || ! & | ^ ~ << >>";
    TokenStream* stream = lex_string(source);
    
    assert(stream != NULL);
    
    TokenType expected[] = {
        TOKEN_PLUS, TOKEN_MINUS, TOKEN_MULTIPLY, TOKEN_DIVIDE, TOKEN_MODULO,
        TOKEN_ASSIGN, TOKEN_EQUAL, TOKEN_NOT_EQUAL,
        TOKEN_LESS, TOKEN_LESS_EQUAL, TOKEN_GREATER, TOKEN_GREATER_EQUAL,
        TOKEN_AND, TOKEN_OR, TOKEN_NOT,
        TOKEN_AMPERSAND, TOKEN_PIPE, TOKEN_CARET, TOKEN_TILDE,
        TOKEN_SHIFT_LEFT, TOKEN_SHIFT_RIGHT
    };
    
    for (int i = 0; i < 22; i++) {
        assert(stream->tokens[i].type == expected[i]);
    }
    
    printf("✓ Operator parsing test passed\n");
    
    free_token_stream(stream);
}

void test_lexer_ownership() {
    printf("Testing ownership syntax...\n");
    
    const char* source = "!String &'r String &'r mut String move borrow";
    TokenStream* stream = lex_string(source);
    
    assert(stream != NULL);
    
    // Check owned type
    assert(stream->tokens[0].type == TOKEN_EXCLAMATION);
    assert(stream->tokens[1].type == TOKEN_IDENTIFIER);
    assert(strcmp(stream->tokens[1].lexeme, "String") == 0);
    
    // Check reference type
    assert(stream->tokens[2].type == TOKEN_AMPERSAND);
    assert(stream->tokens[3].type == TOKEN_LIFETIME);
    assert(strcmp(stream->tokens[3].lexeme, "r") == 0);
    
    // Check mutable reference type
    assert(stream->tokens[5].type == TOKEN_AMPERSAND);
    assert(stream->tokens[6].type == TOKEN_LIFETIME);
    assert(stream->tokens[7].type == TOKEN_MUT);
    assert(stream->tokens[8].type == TOKEN_IDENTIFIER);
    
    // Check move and borrow keywords
    assert(stream->tokens[9].type == TOKEN_MOVE);
    assert(stream->tokens[10].type == TOKEN_BORROW);
    
    printf("✓ Ownership syntax test passed\n");
    
    free_token_stream(stream);
}

int main() {
    printf("Running A# Lexer Tests\n");
    printf("====================\n\n");
    
    test_lexer_basic();
    test_lexer_keywords();
    test_lexer_numbers();
    test_lexer_strings();
    test_lexer_operators();
    test_lexer_ownership();
    
    printf("\nAll lexer tests passed! ✓\n");
    return 0;
}
