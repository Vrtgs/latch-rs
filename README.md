# Latch

A lightweight, no_std compatible synchronization primitive for asynchronous Rust code.

## Overview

This crate provides an async-aware latch synchronization primitive that allows multiple tasks to wait for a signal that unlocks them all simultaneously. It's designed to be lightweight, memory-efficient, and compatible with all major Rust async runtimes including Tokio, async-std, and smol.

## Features

- **No standard library requirement**: Works in `no_std` environments with `alloc` support
- **Runtime agnostic**: Compatible with Tokio, async-std, smol, and pollster
- **Memory efficient**: Optimized internal representation with minimal overhead (the Latch struct is a single usize)
- **Flexible waiting strategies**: Support for spinning, yielding, or busy-looping
- **Thread safety**: Fully `Send` and `Sync` for concurrent access
- **Synchronous API**: Optional synchronous blocking API when using the `std` feature

## Usage

Add this crate to your `Cargo.toml`:

```toml
[dependencies]
latch = "0.1.0"
```



## Examples

### Basic waiting with async/await

```rust
use latch::{Latch, spin::Spin};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let latch = Arc::new(Latch::<Spin>::new());
    let latch_clone = latch.clone();

    // Spawn a task to wait on the latch
    let handle = tokio::spawn(async move {
        println!("Waiting for latch...");
        latch_clone.wait().await;
        println!("Latch opened!");
    });

    // Give the task time to start waiting
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Open the latch, releasing the waiting task
    latch.open();

    // Wait for the task to complete
    handle.await.unwrap();
}
```


### Multiple waiters with Tokio

```rust
use latch::{Latch, spin::Spin};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let latch = Arc::new(Latch::<Spin>::new());
    let mut handles = vec![];

    // Spawn multiple waiters
    for i in 0..10 {
        let latch_clone = latch.clone();
        handles.push(tokio::spawn(async move {
            println!("Task {i} waiting");
            latch_clone.wait().await;
            println!("Task {i} resumed");
        }));
    }

    // Wait briefly to ensure tasks are waiting
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Open the latch, releasing all waiters
    println!("Opening latch");
    latch.open();

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
}
```


### Using the synchronous API

```no_compile
use latch::{Latch, spin::Yield};
use std::sync::Arc;
use std::thread;

fn main() {
    let latch = Arc::new(Latch::<Yield>::new());
    let latch_clone = latch.clone();

    // Spawn a thread that will wait synchronously
    let thread_handle = thread::spawn(move || {
        println!("Thread waiting");
        latch_clone.wait_sync();
        println!("Thread resumed");
    });

    // Give the thread time to start waiting
    thread::sleep(std::time::Duration::from_millis(50));

    // Open the latch, releasing the waiting thread
    println!("Opening latch");
    latch.open();

    // Wait for the thread to complete
    thread_handle.join().unwrap();
}
```


## Feature Flags

- **`std`**: Enables features that require the standard library, such as the `Yield` relaxation strategy and the synchronous `wait_sync()` method. Enabled by default.

## Safety

This crate uses `unsafe` internally for memory efficiency but provides a safe public API.
The unsafe code has been extensively tested with miri, but I wouldn't trust this in production


## Performance Considerations

First, this `Latch` is only one machine word (usize),
and is intended to spend A LOT more time waiting than registering to wait.

It internally uses a spin loop to manage a singly linked waiter list, and closed

When choosing a `RelaxStrategy`:

- `Spin` is typically most efficient for very short wait times
- `Yield` (with the `std` feature) is better for longer waits as it reduces CPU usage
- `Loop` should only be used in specialized environments where spinning or yielding is not possible

## Minimum-Supported Rust Version (MSRV)

This crate requires Rust 1.84.0 or later due to its use of the `expose_provenance` feature.

## License

This project is licensed under the MIT License.
