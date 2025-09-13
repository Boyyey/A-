// Ownership and Borrowing Demo
// Demonstrates A#'s novel ownership model

mod ownership {
    // Unique ownership with move semantics
    fn move_demo() -> !String @ Pure {
        let x: !String = "Hello".to_string();
        let y = x; // x is moved, no longer accessible
        // x.len(); // This would be a compile error
        return y;
    }
    
    // Borrowing with lifetime tracking
    fn borrow_demo<'r>(s: &'r String) -> &'r i32 {
        let len = s.len();
        return &len;
    }
    
    // Mutable borrowing
    fn mutate_demo<'r>(vec: &'r mut Vec<i32>) -> &'r i32 {
        vec.push(42);
        return &vec[0];
    }
    
    // Region polymorphism
    fn region_poly<'r, 's>(a: &'r i32, b: &'s i32) -> i32 @ Pure {
        return *a + *b;
    }
    
    // Ownership transfer in function calls
    fn take_ownership(s: !String) -> i32 @ Pure {
        return s.len();
    }
    
    // Borrowing in function calls
    fn borrow_immutable(s: &String) -> i32 @ Pure {
        return s.len();
    }
    
    // Mutable borrowing in function calls
    fn borrow_mutable(s: &mut String) -> i32 @ Pure {
        s.push_str(" World");
        return s.len();
    }
}
