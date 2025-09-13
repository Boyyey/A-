// Concurrency Demo
// Demonstrates actor-based concurrency with session types

mod concurrency {
    // Actor definition with message types
    actor Counter {
        state: i32,
        
        message Increment(i32) -> i32;
        message Get() -> i32;
        message Reset() -> ();
    }
    
    // Message handler implementation
    impl Counter {
        fn handle_increment(&mut self, amount: i32) -> i32 {
            self.state += amount;
            return self.state;
        }
        
        fn handle_get(&self) -> i32 {
            return self.state;
        }
        
        fn handle_reset(&mut self) -> () {
            self.state = 0;
        }
    }
    
    // Channel-based communication
    fn channel_demo() -> i32 @ Concurrency {
        let (tx, rx) = channel::<i32>();
        
        // Spawn a task
        spawn(|| {
            tx.send(42);
        });
        
        // Receive the value
        let result = rx.recv();
        return result;
    }
    
    // Async/await pattern
    async fn async_demo() -> i32 @ Concurrency {
        let future = async_operation();
        let result = await future;
        return result;
    }
    
    // Session types for protocol verification
    fn session_demo() -> () @ Concurrency {
        let (client, server) = create_session();
        
        // Client sends request
        client.send_request("Hello");
        
        // Server processes and responds
        let response = server.recv_request();
        server.send_response("World");
        
        // Client receives response
        let result = client.recv_response();
        println(result);
    }
}
