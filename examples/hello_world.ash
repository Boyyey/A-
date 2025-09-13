// Hello World in A#
// Demonstrates basic syntax and ownership

mod main {
    // Main function with ownership annotations
    fn main() -> i32 @ Pure {
        let message: !String = "Hello, A#!".to_string();
        println(message);
        return 0;
    }
    
    // Function demonstrating borrowing
    fn greet(name: &'r String) -> !String @ Pure {
        let greeting = "Hello, ".to_string();
        greeting.append(name);
        return greeting;
    }
    
    // Function with region polymorphism
    fn process_data<'r>(data: &'r mut Vec<i32>) -> &'r i32 {
        data.push(42);
        return &data[0];
    }
}
