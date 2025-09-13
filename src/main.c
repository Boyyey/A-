#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"
#include "parser.h"
#include "typecheck.h"
#include "ir.h"
#include "codegen.h"

typedef struct {
    char* input_file;
    char* output_file;
    bool verbose;
    bool debug;
    bool verify;
} CompilerOptions;

static void print_usage(const char* program_name) {
    printf("A# Compiler - Research Language with Formal Verification\n");
    printf("Usage: %s [options] <input_file>\n", program_name);
    printf("\nOptions:\n");
    printf("  -o <file>     Output file (default: a.out)\n");
    printf("  -v            Verbose output\n");
    printf("  -d            Debug mode\n");
    printf("  --verify      Enable formal verification\n");
    printf("  -h, --help    Show this help\n");
    printf("\nExample:\n");
    printf("  %s -o program program.ash\n", program_name);
}

static CompilerOptions parse_args(int argc, char* argv[]) {
    CompilerOptions opts = {0};
    opts.output_file = "a.out";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                opts.output_file = argv[++i];
            } else {
                fprintf(stderr, "Error: -o requires a filename\n");
                exit(1);
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            opts.verbose = true;
        } else if (strcmp(argv[i], "-d") == 0) {
            opts.debug = true;
        } else if (strcmp(argv[i], "--verify") == 0) {
            opts.verify = true;
        } else if (argv[i][0] != '-') {
            opts.input_file = argv[i];
        } else {
            fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
            exit(1);
        }
    }
    
    if (!opts.input_file) {
        fprintf(stderr, "Error: No input file specified\n");
        print_usage(argv[0]);
        exit(1);
    }
    
    return opts;
}

static void compile_file(const CompilerOptions* opts) {
    if (opts->verbose) {
        printf("Compiling %s -> %s\n", opts->input_file, opts->output_file);
    }
    
    // Phase 1: Lexical Analysis
    if (opts->verbose) printf("Phase 1: Lexical Analysis\n");
    TokenStream* tokens = lex_file(opts->input_file);
    if (!tokens) {
        fprintf(stderr, "Error: Failed to tokenize input file\n");
        exit(1);
    }
    
    if (opts->debug) {
        print_tokens(tokens);
    }
    
    // Phase 2: Parsing
    if (opts->verbose) printf("Phase 2: Parsing\n");
    AST* ast = parse_tokens(tokens);
    if (!ast) {
        fprintf(stderr, "Error: Failed to parse input file\n");
        exit(1);
    }
    
    if (opts->debug) {
        print_ast(ast);
    }
    
    // Phase 3: Type Checking
    if (opts->verbose) printf("Phase 3: Type Checking\n");
    TypeContext* type_ctx = create_type_context();
    if (!typecheck_ast(ast, type_ctx)) {
        fprintf(stderr, "Error: Type checking failed\n");
        exit(1);
    }
    
    // Phase 4: IR Generation
    if (opts->verbose) printf("Phase 4: IR Generation\n");
    IRModule* ir = generate_ir(ast, type_ctx);
    if (!ir) {
        fprintf(stderr, "Error: Failed to generate IR\n");
        exit(1);
    }
    
    if (opts->debug) {
        print_ir(ir);
    }
    
    // Phase 5: Code Generation
    if (opts->verbose) printf("Phase 5: Code Generation\n");
    if (!generate_code(ir, opts->output_file)) {
        fprintf(stderr, "Error: Code generation failed\n");
        exit(1);
    }
    
    // Phase 6: Verification (if enabled)
    if (opts->verify) {
        if (opts->verbose) printf("Phase 6: Formal Verification\n");
        // TODO: Implement verification
        printf("Verification not yet implemented\n");
    }
    
    // Cleanup
    free_token_stream(tokens);
    free_ast(ast);
    free_type_context(type_ctx);
    free_ir_module(ir);
    
    if (opts->verbose) {
        printf("Compilation successful!\n");
    }
}

int main(int argc, char* argv[]) {
    printf("A# Compiler v0.1.0 - Research Language with Formal Verification\n");
    
    CompilerOptions opts = parse_args(argc, argv);
    compile_file(&opts);
    
    return 0;
}
