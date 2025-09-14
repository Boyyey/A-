#ifndef LIBRARY_SYSTEM_H
#define LIBRARY_SYSTEM_H

#include "parser.h"
#include "typecheck.h"
#include <stdbool.h>

// Library System for A# - Enables custom AI/ML libraries

typedef enum {
    LIBRARY_TYPE_STANDARD,
    LIBRARY_TYPE_AI_ML,
    LIBRARY_TYPE_WEB,
    LIBRARY_TYPE_GAME,
    LIBRARY_TYPE_CRYPTO,
    LIBRARY_TYPE_DATABASE,
    LIBRARY_TYPE_NETWORK,
    LIBRARY_TYPE_GRAPHICS
} LibraryType;

typedef struct {
    char* name;
    char* version;
    char* author;
    char* description;
    LibraryType type;
    char* license;
    char* repository;
    char* documentation;
} LibraryMetadata;

typedef struct {
    char* name;
    char* signature;
    char* documentation;
    void* implementation;
    bool is_public;
    bool is_async;
} LibraryFunction;

typedef struct {
    char* name;
    char* type_definition;
    char* documentation;
    bool is_public;
} LibraryType;

typedef struct {
    LibraryMetadata metadata;
    LibraryFunction* functions;
    uint32_t function_count;
    LibraryType* types;
    uint32_t type_count;
    char* source_code;
    char* compiled_code;
    bool is_compiled;
} Library;

typedef struct {
    Library** libraries;
    uint32_t library_count;
    char* cache_directory;
    char* registry_url;
} LibraryManager;

// Library Management Functions
LibraryManager* create_library_manager(const char* cache_dir);
void free_library_manager(LibraryManager* manager);

// Library Operations
Library* create_library(const char* name, const char* version, const char* author);
void free_library(Library* library);
bool add_function(Library* library, const char* name, const char* signature, 
                  const char* documentation, void* implementation);
bool add_type(Library* library, const char* name, const char* type_definition, 
              const char* documentation);
bool compile_library(Library* library);
bool load_library(LibraryManager* manager, const char* library_name);
bool unload_library(LibraryManager* manager, const char* library_name);

// Library Discovery
Library** search_libraries(LibraryManager* manager, const char* query);
Library** get_libraries_by_type(LibraryManager* manager, LibraryType type);
Library* get_library(LibraryManager* manager, const char* name);

// Library Registry
bool publish_library(Library* library, const char* registry_url);
bool install_library(LibraryManager* manager, const char* library_name, 
                     const char* version);
bool update_library(LibraryManager* manager, const char* library_name);
bool remove_library(LibraryManager* manager, const char* library_name);

// Library Dependencies
bool add_dependency(Library* library, const char* dependency_name, 
                   const char* version_constraint);
bool resolve_dependencies(Library* library, LibraryManager* manager);
bool check_dependency_conflicts(Library* library, LibraryManager* manager);

// Library Building
bool build_library_from_source(Library* library, const char* source_code);
bool build_library_from_ast(Library* library, const AST* ast);
bool optimize_library(Library* library);

// Library Testing
bool test_library(Library* library);
bool benchmark_library(Library* library);
bool validate_library(Library* library);

// Library Documentation
char* generate_library_docs(Library* library);
char* generate_api_reference(Library* library);
bool export_library_docs(Library* library, const char* output_dir);

// Library Packaging
bool package_library(Library* library, const char* output_file);
bool unpack_library(const char* package_file, const char* output_dir);
bool verify_library_package(const char* package_file);

// Library Versioning
bool set_library_version(Library* library, const char* version);
bool compare_library_versions(const char* version1, const char* version2);
bool is_library_compatible(Library* library1, Library* library2);

// Library Security
bool sign_library(Library* library, const char* private_key);
bool verify_library_signature(Library* library, const char* public_key);
bool check_library_permissions(Library* library);

// Library Performance
float get_library_performance_score(Library* library);
bool profile_library(Library* library);
bool optimize_library_performance(Library* library);

// Library Integration
bool integrate_library(Library* library, const AST* target_ast);
bool link_library(Library* library, const char* target_file);
bool embed_library(Library* library, const char* target_file);

// Library Utilities
char* get_library_info(Library* library);
char* list_library_functions(Library* library);
char* list_library_types(Library* library);
bool validate_library_interface(Library* library);

// A# Specific Library Features
bool create_ai_ml_library(Library* library, const char* model_type);
bool create_web_library(Library* library, const char* framework);
bool create_game_library(Library* library, const char* engine);
bool create_crypto_library(Library* library, const char* algorithm);

// Library Examples
Library* create_example_ai_library(void);
Library* create_example_web_library(void);
Library* create_example_game_library(void);
Library* create_example_crypto_library(void);

// Library Templates
Library* create_library_template(LibraryType type);
bool apply_library_template(Library* library, const char* template_name);

// Library Validation
bool validate_library_syntax(Library* library);
bool validate_library_semantics(Library* library);
bool validate_library_performance(Library* library);
bool validate_library_security(Library* library);

// Library Metrics
typedef struct {
    uint32_t function_count;
    uint32_t type_count;
    uint32_t line_count;
    uint32_t complexity_score;
    float performance_score;
    float security_score;
    float maintainability_score;
} LibraryMetrics;

LibraryMetrics* get_library_metrics(Library* library);
void free_library_metrics(LibraryMetrics* metrics);

// Library Comparison
typedef struct {
    Library* library1;
    Library* library2;
    float similarity_score;
    char** differences;
    uint32_t difference_count;
} LibraryComparison;

LibraryComparison* compare_libraries(Library* library1, Library* library2);
void free_library_comparison(LibraryComparison* comparison);

// Library Migration
bool migrate_library(Library* library, const char* target_version);
bool upgrade_library_dependencies(Library* library, LibraryManager* manager);
bool downgrade_library_dependencies(Library* library, LibraryManager* manager);

// Library Backup and Restore
bool backup_library(Library* library, const char* backup_path);
bool restore_library(const char* backup_path, const char* target_path);
bool list_library_backups(const char* library_name);

// Library Analytics
typedef struct {
    uint32_t download_count;
    uint32_t usage_count;
    float rating;
    char** user_reviews;
    uint32_t review_count;
} LibraryAnalytics;

LibraryAnalytics* get_library_analytics(Library* library);
void free_library_analytics(LibraryAnalytics* analytics);

// Library Community
bool submit_library_review(Library* library, const char* review, float rating);
bool report_library_issue(Library* library, const char* issue_description);
bool request_library_feature(Library* library, const char* feature_description);

// Library Integration with A# Compiler
bool integrate_library_with_compiler(Library* library, const char* compiler_path);
bool generate_library_bindings(Library* library, const char* target_language);
bool create_library_wrapper(Library* library, const char* target_language);

#endif // LIBRARY_SYSTEM_H
