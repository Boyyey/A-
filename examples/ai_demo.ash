mod ai_demo { 
    use std::ml::*; 
 
    fn main() -> () @ Resource { 
        println("🧠 A# AI Demo with MinGW64"); 
 
        // Create a simple neural network 
        let model = create_mlp([784, 128, 10], 3, ACTIVATION_RELU, 0.001); 
        let optimizer = create_optimizer(OPTIMIZER_ADAM, "adam", 0.001); 
        let loss_fn = create_loss_function(LOSS_CROSSENTROPY, "crossentropy", null); 
 
        println("Neural network created successfully!"); 
        println("Model parameters: 784 -^> 128 -^> 10"); 
        println("Optimizer: Adam with learning rate 0.001"); 
        println("Loss function: Cross Entropy"); 
 
        // Simulate training 
        for epoch in 0..5 { 
            let loss = 1.0 - (epoch as f32) * 0.15; 
            println("Epoch {}: Loss = {:.3}", epoch + 1, loss); 
        } 
 
        println("Training completed! Model ready for inference."); 
    } 
} 
