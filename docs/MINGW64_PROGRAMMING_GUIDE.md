# A# Programming with MinGW64
## Complete Guide for Windows Development

This guide shows you how to program in A# using MinGW64 on Windows. A# is the most advanced AI/ML language with formal verification, and MinGW64 provides an excellent development environment.

## Table of Contents

1. [Installation](#installation)
2. [Setting Up MinGW64](#setting-up-mingw64)
3. [Your First A# Program](#your-first-a-program)
4. [AI/ML Programming](#aiml-programming)
5. [Advanced Features](#advanced-features)
6. [Development Workflow](#development-workflow)
7. [Troubleshooting](#troubleshooting)

## Installation

### 1. Install MinGW64

Download and install MinGW64 from [https://www.mingw-w64.org/](https://www.mingw-w64.org/) or use a package manager:

```bash
# Using Chocolatey
choco install mingw

# Using Scoop
scoop install gcc

# Using MSYS2
pacman -S mingw-w64-x86_64-gcc
```

### 2. Install A# Compiler

```bash
# Clone the A# repository
git clone https://github.com/boyyey/A-sharp
cd A-sharp

# Run the MinGW64 installation script
./install_mingw64.bat

# Add A# to your PATH
export PATH=$PATH:$(pwd)/bin
```

### 3. Verify Installation

```bash
# Check if A# compiler is working
ashc --version

# Should output:
# A# Compiler v0.1.0 - Research Language with Formal Verification
```

## Setting Up MinGW64

### 1. Environment Setup

Add MinGW64 to your PATH:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export PATH="/c/Program Files/mingw-w64/x86_64-8.1.0-posix-seh-rt_v6-rev0/mingw64/bin:$PATH"

# Or add to Windows PATH through System Properties
# Control Panel > System > Advanced > Environment Variables
```

### 2. Verify MinGW64

```bash
# Check GCC version
gcc --version

# Check if make is available
make --version

# Check if gdb is available (for debugging)
gdb --version
```

### 3. Create Development Directory

```bash
# Create your A# project directory
mkdir my_ash_project
cd my_ash_project

# Initialize with A# template
ashc --init my_project
```

## Your First A# Program

### 1. Hello World

Create `hello.ash`:

```a#
mod hello_world {
    fn main() -> i32 @ Pure {
        let message: !String = "Hello, A# from MinGW64!".to_string();
        println(message);
        return 0;
    }
}
```

### 2. Compile and Run

```bash
# Compile the program
ashc hello.ash -o hello.exe

# Run the program
./hello.exe

# Output:
# Hello, A# from MinGW64!
```

### 3. Advanced Hello World

Create `advanced_hello.ash`:

```a#
mod advanced_hello {
    use std::collections::*;
    
    struct Person {
        name: !String,
        age: i32,
        skills: Vec<!String>
    }
    
    impl Person {
        fn new(name: !String, age: i32) -> !Person @ Resource {
            Person {
                name,
                age,
                skills: Vec::new()
            }
        }
        
        fn add_skill<'r>(&mut self, skill: &'r str) -> () @ Resource {
            self.skills.push(skill.to_string());
        }
        
        fn introduce(&self) -> !String @ Pure {
            let mut intro = format!("Hello! I'm {} and I'm {} years old.", self.name, self.age);
            
            if !self.skills.is_empty() {
                intro.push_str(" My skills include: ");
                for (i, skill) in self.skills.iter().enumerate() {
                    if i > 0 {
                        intro.push_str(", ");
                    }
                    intro.push_str(skill);
                }
                intro.push_str(".");
            }
            
            return intro;
        }
    }
    
    fn main() -> () @ Resource {
        let mut person = Person::new("Alice".to_string(), 25);
        person.add_skill("A# Programming");
        person.add_skill("AI/ML");
        person.add_skill("Formal Verification");
        
        let introduction = person.introduce();
        println(introduction);
        
        // Demonstrate ownership
        let moved_person = person; // person is now moved
        // person.introduce(); // ERROR: person has been moved
        
        let new_intro = moved_person.introduce();
        println("After move: {}", new_intro);
    }
}
```

### 4. Compile and Run Advanced Example

```bash
# Compile with verbose output
ashc -v advanced_hello.ash -o advanced_hello.exe

# Run the program
./advanced_hello.exe

# Output:
# Hello! I'm Alice and I'm 25 years old. My skills include: A# Programming, AI/ML, Formal Verification.
# After move: Hello! I'm Alice and I'm 25 years old. My skills include: A# Programming, AI/ML, Formal Verification.
```

## AI/ML Programming

### 1. Simple Neural Network

Create `neural_network.ash`:

```a#
mod neural_network_demo {
    use std::ml::*;
    
    fn main() -> () @ Resource {
        println("🧠 A# Neural Network Demo with MinGW64");
        
        // Create a simple neural network
        let model = create_mlp([784, 128, 64, 10], 4, ACTIVATION_RELU, 0.001);
        let optimizer = create_optimizer(OPTIMIZER_ADAM, "adam", 0.001);
        let loss_fn = create_loss_function(LOSS_CROSSENTROPY, "crossentropy", null);
        
        println("Neural network created successfully!");
        println("Architecture: 784 -> 128 -> 64 -> 10");
        println("Activation: ReLU");
        println("Learning rate: 0.001");
        
        // Simulate training data
        let input_data = create_tensor(TENSOR_FLOAT32, [32, 784], 2).random_normal();
        let target_data = create_tensor(TENSOR_FLOAT32, [32, 10], 2).random_normal();
        
        // Training loop
        for epoch in 0..10 {
            // Forward pass
            let predictions = model.forward(&input_data);
            let loss = loss_fn.compute(predictions, &target_data);
            
            // Backward pass
            loss.backward();
            optimizer.step();
            
            let loss_value = loss.value();
            println("Epoch {}: Loss = {:.4}", epoch + 1, loss_value);
        }
        
        println("Training completed! Model ready for inference.");
        
        // Test inference
        let test_input = create_tensor(TENSOR_FLOAT32, [1, 784], 2).random_normal();
        let test_output = model.forward(&test_input);
        let probabilities = test_output.softmax(-1);
        
        println("Test inference completed!");
        println("Output shape: {:?}", probabilities.shape());
    }
}
```

### 2. Compile and Run AI Example

```bash
# Compile with ML support
ashc --ml neural_network.ash -o neural_network.exe

# Run the AI program
./neural_network.exe

# Output:
# 🧠 A# Neural Network Demo with MinGW64
# Neural network created successfully!
# Architecture: 784 -> 128 -> 64 -> 10
# Activation: ReLU
# Learning rate: 0.001
# Epoch 1: Loss = 2.3026
# Epoch 2: Loss = 2.1456
# ...
# Training completed! Model ready for inference.
# Test inference completed!
# Output shape: [1, 10]
```

### 3. Advanced AI Example - Transformer

Create `transformer_demo.ash`:

```a#
mod transformer_demo {
    use std::ml::*;
    
    fn main() -> () @ Resource {
        println("🤖 A# Transformer Demo with MinGW64");
        
        // Create a transformer model
        let transformer = create_transformer(
            512,    // d_model
            8,      // num_heads
            6,      // num_layers
            2048,   // d_ff
            10000,  // vocab_size
            0.0001  // learning rate
        );
        
        println("Transformer created successfully!");
        println("Model size: 512 dimensions");
        println("Attention heads: 8");
        println("Layers: 6");
        println("Feed-forward size: 2048");
        println("Vocabulary size: 10,000");
        
        // Simulate text processing
        let input_ids = create_tensor(TENSOR_INT32, [1, 128], 2).random_uniform(0, 10000);
        let attention_mask = create_tensor(TENSOR_FLOAT32, [1, 128], 2).fill(1.0);
        
        // Forward pass
        let output = transformer.forward(&input_ids, &attention_mask);
        
        println("Text processing completed!");
        println("Input shape: {:?}", input_ids.shape());
        println("Output shape: {:?}", output.shape());
        
        // Demonstrate attention mechanisms
        let attention_weights = transformer.get_attention_weights();
        println("Attention weights shape: {:?}", attention_weights.shape());
        
        println("Transformer demo completed successfully!");
    }
}
```

## Advanced Features

### 1. Concurrency with Actors

Create `actor_demo.ash`:

```a#
mod actor_demo {
    use std::concurrency::*;
    
    actor Counter {
        state: i32,
        
        message Increment(i32) -> i32;
        message Get() -> i32;
        message Reset() -> ();
    }
    
    impl Counter {
        fn new(initial: i32) -> !Counter @ Resource {
            Counter { state: initial }
        }
        
        fn handle_increment(&mut self, value: i32) -> i32 {
            self.state += value;
            return self.state;
        }
        
        fn handle_get(&self) -> i32 {
            return self.state;
        }
        
        fn handle_reset(&mut self) -> () {
            self.state = 0;
        }
    }
    
    fn main() -> () @ Concurrency {
        println("🎭 A# Actor Demo with MinGW64");
        
        let counter = Counter::new(0);
        let counter_ref = counter.spawn();
        
        // Send messages concurrently
        let task1 = spawn(|| {
            for i in 0..5 {
                let value = counter_ref.send(Increment(i + 1));
                println("Task 1: Counter = {}", value);
            }
        });
        
        let task2 = spawn(|| {
            for i in 0..3 {
                let value = counter_ref.send(Increment(i * 2));
                println("Task 2: Counter = {}", value);
            }
        });
        
        // Wait for tasks to complete
        task1.join();
        task2.join();
        
        let final_value = counter_ref.send(Get());
        println("Final counter value: {}", final_value);
        
        // Reset counter
        counter_ref.send(Reset());
        let reset_value = counter_ref.send(Get());
        println("After reset: {}", reset_value);
    }
}
```

### 2. Metaprogramming with Macros

Create `macro_demo.ash`:

```a#
mod macro_demo {
    use std::ml::*;
    
    // Define a macro for creating neural layers
    macro_rules! neural_layer {
        ($input_size:expr, $output_size:expr, $activation:expr) => {
            {
                let weights = create_tensor(TENSOR_FLOAT32, [$input_size, $output_size], 2);
                let biases = create_tensor(TENSOR_FLOAT32, [$output_size], 1);
                let activation = $activation;
                
                Layer {
                    weights: weights,
                    biases: biases,
                    activation: activation,
                    input_size: $input_size,
                    output_size: $output_size
                }
            }
        };
    }
    
    // Define a macro for creating optimizers
    macro_rules! create_optimizer_with_lr {
        ($name:ident, $learning_rate:expr) => {
            fn $name() -> Optimizer {
                Optimizer {
                    name: stringify!($name),
                    learning_rate: $learning_rate,
                    state: HashMap::new()
                }
            }
        };
    }
    
    fn main() -> () @ Resource {
        println("🔧 A# Macro Demo with MinGW64");
        
        // Use the neural layer macro
        let layer1 = neural_layer!(784, 128, ACTIVATION_RELU);
        let layer2 = neural_layer!(128, 64, ACTIVATION_RELU);
        let layer3 = neural_layer!(64, 10, ACTIVATION_SOFTMAX);
        
        println("Neural layers created with macros:");
        println("Layer 1: {} -> {} with ReLU", layer1.input_size, layer1.output_size);
        println("Layer 2: {} -> {} with ReLU", layer2.input_size, layer2.output_size);
        println("Layer 3: {} -> {} with Softmax", layer3.input_size, layer3.output_size);
        
        // Use the optimizer macro
        create_optimizer_with_lr!(adam_optimizer, 0.001);
        create_optimizer_with_lr!(sgd_optimizer, 0.01);
        
        let adam = adam_optimizer();
        let sgd = sgd_optimizer();
        
        println("Optimizers created with macros:");
        println("Adam: {} with LR {}", adam.name, adam.learning_rate);
        println("SGD: {} with LR {}", sgd.name, sgd.learning_rate);
        
        println("Macro demo completed successfully!");
    }
}
```

## Development Workflow

### 1. Using Make

```bash
# Use the provided Makefile
make -f Makefile.mingw64

# Clean build
make -f Makefile.mingw64 clean

# Install
make -f Makefile.mingw64 install
```

### 2. Using Vim

```bash
# Install Vim syntax highlighting
cp vim/ash.vim ~/.vim/syntax/

# Add to ~/.vimrc
echo 'au BufNewFile,BufRead *.ash set filetype=ash' >> ~/.vimrc

# Edit A# files with syntax highlighting
vim hello.ash
```

### 3. Debugging with GDB

```bash
# Compile with debug symbols
ashc -g hello.ash -o hello_debug.exe

# Debug with GDB
gdb ./hello_debug.exe

# In GDB:
# (gdb) break main
# (gdb) run
# (gdb) step
# (gdb) print variable_name
# (gdb) quit
```

### 4. Project Structure

```
my_ash_project/
├── src/
│   ├── main.ash
│   ├── neural_network.ash
│   └── utils.ash
├── examples/
│   ├── hello_world.ash
│   └── ai_demo.ash
├── tests/
│   └── test_main.ash
├── docs/
│   └── README.md
├── Makefile
└── .gitignore
```

## Troubleshooting

### 1. Common Issues

**Issue**: `ashc: command not found`
```bash
# Solution: Add A# to PATH
export PATH=$PATH:/path/to/A-sharp/bin
```

**Issue**: `gcc: command not found`
```bash
# Solution: Install MinGW64 or add to PATH
export PATH=$PATH:/c/Program\ Files/mingw-w64/x86_64-8.1.0-posix-seh-rt_v6-rev0/mingw64/bin
```

**Issue**: Compilation errors
```bash
# Solution: Check syntax and dependencies
ashc -v your_program.ash -o output.exe
```

### 2. Performance Optimization

```bash
# Compile with optimizations
ashc -O2 your_program.ash -o optimized.exe

# Compile for specific architecture
ashc -march=native your_program.ash -o native.exe

# Profile your program
gprof your_program.exe
```

### 3. Memory Debugging

```bash
# Compile with memory debugging
ashc -fsanitize=address your_program.ash -o debug.exe

# Run with Valgrind (if available)
valgrind --leak-check=full ./debug.exe
```

## Conclusion

A# with MinGW64 provides a powerful development environment for:

- **AI/ML Programming** - Built-in neural networks and tensors
- **Systems Programming** - Memory safety with ownership
- **Concurrent Programming** - Actor-based concurrency
- **Metaprogramming** - Hygienic macros and code generation
- **Formal Verification** - Mechanized proofs of correctness

**Start building the future of AI with A# and MinGW64 today!** 🚀

---

**A# - Where Safety Meets AI, and Research Meets Production!** 🧠✨
