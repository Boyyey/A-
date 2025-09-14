mod hello_world { 
    fn main() -> i32 @ Pure { 
        let message: !String = "Hello, A# from MinGW64!".to_string(); 
        println(message); 
        return 0; 
    } 
} 
