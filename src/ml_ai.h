#ifndef ML_AI_H
#define ML_AI_H

#include "parser.h"
#include "typecheck.h"
#include <stdbool.h>

// ML/AI Types
typedef enum {
    ML_TENSOR,
    ML_NEURAL_NETWORK,
    ML_LAYER,
    ML_OPTIMIZER,
    ML_LOSS_FUNCTION,
    ML_ACTIVATION,
    ML_DATASET,
    ML_MODEL,
    ML_GRADIENT,
    ML_AUTO_DIFF
} MLType;

typedef enum {
    TENSOR_FLOAT32,
    TENSOR_FLOAT64,
    TENSOR_INT32,
    TENSOR_INT64,
    TENSOR_BOOL
} TensorDataType;

typedef enum {
    LAYER_DENSE,
    LAYER_CONV2D,
    LAYER_LSTM,
    LAYER_GRU,
    LAYER_ATTENTION,
    LAYER_TRANSFORMER,
    LAYER_DROPOUT,
    LAYER_BATCH_NORM
} LayerType;

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAGRAD,
    OPTIMIZER_ADADELTA
} OptimizerType;

typedef enum {
    LOSS_MSE,
    LOSS_MAE,
    LOSS_CROSSENTROPY,
    LOSS_BINARY_CROSSENTROPY,
    LOSS_HINGE,
    LOSS_HUBER
} LossType;

typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX,
    ACTIVATION_GELU,
    ACTIVATION_SWISH
} ActivationType;

// Tensor structure
typedef struct {
    TensorDataType dtype;
    int32_t* shape;
    uint32_t ndim;
    void* data;
    bool requires_grad;
    bool is_leaf;
} Tensor;

// Neural Network structures
typedef struct {
    LayerType type;
    char* name;
    Tensor** weights;
    Tensor** biases;
    uint32_t weight_count;
    uint32_t bias_count;
    void* config; // Layer-specific configuration
} Layer;

typedef struct {
    char* name;
    Layer** layers;
    uint32_t layer_count;
    bool training;
} NeuralNetwork;

typedef struct {
    OptimizerType type;
    char* name;
    float learning_rate;
    void* state; // Optimizer-specific state
} Optimizer;

typedef struct {
    LossType type;
    char* name;
    void* config; // Loss-specific configuration
} LossFunction;

typedef struct {
    char* name;
    Tensor** inputs;
    Tensor** targets;
    uint32_t sample_count;
    uint32_t batch_size;
} Dataset;

typedef struct {
    char* name;
    NeuralNetwork* network;
    Optimizer* optimizer;
    LossFunction* loss_function;
    bool trained;
} Model;

// Auto-differentiation structures
typedef struct {
    Tensor* value;
    Tensor* grad;
    char* operation;
    struct AutoDiffNode** inputs;
    uint32_t input_count;
    void (*backward)(struct AutoDiffNode*);
} AutoDiffNode;

typedef struct {
    AutoDiffNode** nodes;
    uint32_t node_count;
    bool requires_grad;
} ComputationGraph;

// ML/AI Functions
// Tensor operations
Tensor* create_tensor(TensorDataType dtype, int32_t* shape, uint32_t ndim);
Tensor* create_tensor_from_data(TensorDataType dtype, int32_t* shape, uint32_t ndim, void* data);
void free_tensor(Tensor* tensor);
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* tensor_reshape(Tensor* tensor, int32_t* new_shape, uint32_t new_ndim);
Tensor* tensor_transpose(Tensor* tensor, int32_t* axes, uint32_t axis_count);
Tensor* tensor_sum(Tensor* tensor, int32_t* axes, uint32_t axis_count);
Tensor* tensor_mean(Tensor* tensor, int32_t* axes, uint32_t axis_count);
Tensor* tensor_max(Tensor* tensor, int32_t* axes, uint32_t axis_count);
Tensor* tensor_min(Tensor* tensor, int32_t* axes, uint32_t axis_count);

// Neural Network operations
NeuralNetwork* create_neural_network(const char* name);
void free_neural_network(NeuralNetwork* network);
Layer* create_layer(LayerType type, const char* name, void* config);
void free_layer(Layer* layer);
void add_layer(NeuralNetwork* network, Layer* layer);
Tensor* forward_pass(NeuralNetwork* network, Tensor* input);
Tensor* backward_pass(NeuralNetwork* network, Tensor* grad_output);

// Layer implementations
Layer* create_dense_layer(uint32_t input_size, uint32_t output_size, const char* name);
Layer* create_conv2d_layer(uint32_t input_channels, uint32_t output_channels, 
                          uint32_t kernel_size, uint32_t stride, const char* name);
Layer* create_lstm_layer(uint32_t input_size, uint32_t hidden_size, const char* name);
Layer* create_attention_layer(uint32_t d_model, uint32_t num_heads, const char* name);
Layer* create_transformer_layer(uint32_t d_model, uint32_t num_heads, uint32_t d_ff, const char* name);

// Optimizers
Optimizer* create_optimizer(OptimizerType type, const char* name, float learning_rate);
void free_optimizer(Optimizer* optimizer);
void optimizer_step(Optimizer* optimizer, NeuralNetwork* network);

// Loss functions
LossFunction* create_loss_function(LossType type, const char* name, void* config);
void free_loss_function(LossFunction* loss);
Tensor* compute_loss(LossFunction* loss, Tensor* predictions, Tensor* targets);
Tensor* loss_backward(LossFunction* loss, Tensor* predictions, Tensor* targets);

// Activation functions
Tensor* apply_activation(ActivationType activation, Tensor* input);
Tensor* activation_backward(ActivationType activation, Tensor* input, Tensor* grad_output);

// Auto-differentiation
AutoDiffNode* create_autodiff_node(Tensor* value, const char* operation);
void free_autodiff_node(AutoDiffNode* node);
void backward(AutoDiffNode* node);
ComputationGraph* create_computation_graph(void);
void free_computation_graph(ComputationGraph* graph);
void add_node(ComputationGraph* graph, AutoDiffNode* node);

// Model training
Model* create_model(const char* name, NeuralNetwork* network, Optimizer* optimizer, LossFunction* loss);
void free_model(Model* model);
void train_model(Model* model, Dataset* dataset, uint32_t epochs);
Tensor* predict(Model* model, Tensor* input);
float evaluate_model(Model* model, Dataset* dataset);

// Dataset operations
Dataset* create_dataset(const char* name, Tensor** inputs, Tensor** targets, uint32_t sample_count);
void free_dataset(Dataset* dataset);
Dataset* load_dataset_from_file(const char* filename);
void save_dataset_to_file(Dataset* dataset, const char* filename);
Dataset* split_dataset(Dataset* dataset, float train_ratio, float val_ratio, float test_ratio);

// High-level ML operations
Model* create_linear_regression(uint32_t input_size, float learning_rate);
Model* create_logistic_regression(uint32_t input_size, uint32_t num_classes, float learning_rate);
Model* create_mlp(uint32_t* layer_sizes, uint32_t layer_count, ActivationType activation, float learning_rate);
Model* create_cnn(uint32_t input_channels, uint32_t* conv_layers, uint32_t* dense_layers, 
                 uint32_t conv_count, uint32_t dense_count, float learning_rate);
Model* create_rnn(uint32_t input_size, uint32_t hidden_size, uint32_t output_size, float learning_rate);
Model* create_transformer(uint32_t d_model, uint32_t num_heads, uint32_t num_layers, 
                         uint32_t d_ff, uint32_t vocab_size, float learning_rate);

// Advanced ML features
Tensor* attention_mechanism(Tensor* query, Tensor* key, Tensor* value, Tensor* mask);
Tensor* multi_head_attention(Tensor* input, uint32_t num_heads, uint32_t d_model);
Tensor* positional_encoding(Tensor* input, uint32_t max_len);
Tensor* layer_normalization(Tensor* input, float epsilon);
Tensor* dropout(Tensor* input, float rate, bool training);

// GPU acceleration (CUDA/OpenCL)
bool is_gpu_available(void);
Tensor* tensor_to_gpu(Tensor* tensor);
Tensor* tensor_to_cpu(Tensor* tensor);
Tensor* gpu_tensor_add(Tensor* a, Tensor* b);
Tensor* gpu_tensor_matmul(Tensor* a, Tensor* b);
Tensor* gpu_conv2d(Tensor* input, Tensor* kernel, uint32_t stride, uint32_t padding);

// Model serialization
void save_model(Model* model, const char* filename);
Model* load_model(const char* filename);
void export_model_to_onnx(Model* model, const char* filename);
Model* import_model_from_onnx(const char* filename);

// Utility functions
void print_tensor(Tensor* tensor);
void print_model_summary(Model* model);
void visualize_training_history(float* losses, uint32_t epoch_count);
Tensor* random_tensor(TensorDataType dtype, int32_t* shape, uint32_t ndim, float min_val, float max_val);
Tensor* zeros_tensor(TensorDataType dtype, int32_t* shape, uint32_t ndim);
Tensor* ones_tensor(TensorDataType dtype, int32_t* shape, uint32_t ndim);

// A# Language integration
bool typecheck_ml_expression(const ASTNode* expr, TypeContext* ctx, Type** result);
IRNode* generate_ml_ir(const ASTNode* expr, TypeContext* ctx);
bool compile_ml_to_llvm(IRNode* ir, const char* output_file);

#endif // ML_AI_H
