#![doc = include_str!("../README.md")]
#![no_std]

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub extern crate spin;

use alloc::boxed::Box;
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem;
use core::mem::ManuallyDrop;
use core::pin::Pin;
use core::ptr::{with_exposed_provenance_mut, NonNull};
use core::sync::atomic::{AtomicUsize, Ordering};
use core::task::{Context, Poll, Waker};
use spin::RelaxStrategy;

#[repr(align(2))]
struct Node {
    next: Link,
    waker: UnsafeCell<Waker>,
}

type Link = Option<NonNull<Node>>;


// guaranteed https://doc.rust-lang.org/std/boxed/index.html#memory-layout
const _: () = assert!(
    size_of::<Link>() == size_of::<usize>()
    && size_of::<Link>() == size_of::<Link>()
    && size_of::<Link>() == size_of::<Link>()
    && align_of::<Node>() >= 2
);

const LOCK_BIT: usize = 0b1;
const ADDR_BITS: usize = !LOCK_BIT;

// this is an invalid address as when trying to read, it will wrap around and overflow

const OPEN_LATCH: usize = usize::MAX;
const OPEN_LATCH_NORMALIZED: usize = OPEN_LATCH & ADDR_BITS;
// new list
const NULL: usize = 0;


#[inline(always)]
fn is_open(ptr: usize) -> bool {
    ptr == OPEN_LATCH
}



// Err(()) => locked
// Ok(Some(x)) => list x
// Ok(None) => open
fn from_state(addr: usize) -> Result<Option<Link>, ()> {
    if addr & LOCK_BIT == 0 {
        let link = match addr {
            NULL => None,
            addr => {
                let bx = unsafe { NonNull::new_unchecked(with_exposed_provenance_mut(addr)) };
                Some(bx)
            },
        };

        return  Ok(Some(link))
    }

    if is_open(addr) {
        return Ok(None)
    }

    Err(())
}

fn to_state(link: Link) -> usize {
    match link {
        None => NULL,
        Some(ptr) => ptr.as_ptr().expose_provenance(),
    }
}


/// A lightweight, memory-efficient synchronization primitive for asynchronous Rust code.
///
/// `Latch` is a synchronization primitive that allows multiple tasks to wait for a signal
/// that unlocks them all simultaneously. It's designed with minimal memory overhead,
/// using only a single `usize` for its internal state.
///
/// When the `Latch` is open, calling all waiting methods is lock-free and wait-free
/// 
/// # Memory Layout
///
/// The `Latch` struct is marked with `#[repr(transparent)]` ensuring it has the same
/// memory layout as a single `AtomicUsize`, making it extremely lightweight.
///
/// # Type Parameters
///
/// * `S` - The relaxation strategy used when spinning on the internal lock.
///   Must implement the `RelaxStrategy` trait from the `spin` crate.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use latch::{Latch, spin::Spin};
/// use std::sync::Arc;
///
/// # async fn example() {
/// let latch = Arc::new(Latch::<Spin>::new());
/// let latch_clone = latch.clone();
///
/// // Spawn a task to wait on the latch
/// let task = tokio::spawn(async move {
///     println!("Waiting for latch...");
///     latch_clone.wait().await;
///     println!("Latch opened!");
/// });
///
/// // Open the latch, releasing any waiting tasks
/// latch.open();
/// # }
/// ```
///
/// Multiple waiters:
///
/// ```rust
/// use latch::{Latch, spin::Spin};
/// use std::sync::Arc;
///
/// # async fn example() {
/// let latch = Arc::new(Latch::<Spin>::new());
/// let mut handles = vec![];
///
/// // Spawn multiple waiters
/// for i in 0..10 {
///     let latch_clone = latch.clone();
///     handles.push(tokio::spawn(async move {
///         println!("Task {i} waiting");
///         latch_clone.wait().await;
///         println!("Task {i} resumed");
///     }));
/// }
///
/// // Open the latch, releasing all waiters at once
/// latch.open();
///
/// // Wait for all tasks to complete
/// for handle in handles {
///     handle.await.unwrap();
/// }
/// # }
/// ```
///
/// Synchronous waiting (with the `std` feature):
///
/// ```rust
/// # #[cfg(feature = "std")]
/// # {
/// use latch::{Latch, spin::Spin};
/// use std::sync::Arc;
/// use std::thread;
///
/// let latch = Arc::new(Latch::<Spin>::new());
/// let latch_clone = latch.clone();
///
/// // Spawn a thread that will wait synchronously
/// let thread_handle = thread::spawn(move || {
///     println!("Thread waiting");
///     latch_clone.wait_sync();
///     println!("Thread resumed");
/// });
///
/// // Open the latch, releasing the waiting thread
/// latch.open();
/// # }
/// ```
#[repr(transparent)]
pub struct Latch<S: RelaxStrategy> {
    // the lowest bit is for lock
    // a Null pointer for just an empty list
    // All bits on means the latch is open
    ptr: AtomicUsize,
    _strategy: PhantomData<S>
}

impl<S: RelaxStrategy> Drop for Latch<S> {
    fn drop(&mut self) {
        let addr = (*self.ptr.get_mut()) & ADDR_BITS;

        match addr {
            OPEN_LATCH_NORMALIZED | NULL => {}
            addr => drop_drain(Some(unsafe { NonNull::new_unchecked(with_exposed_provenance_mut(addr)) }))
        }
    }
}


/// # Safety
/// `node.next` must be None
unsafe fn push(list: &mut Link, node: Box<Node>) -> NonNull<UnsafeCell<Waker>> {
    let node = unsafe { NonNull::new_unchecked(Box::into_raw(node)) };
    let waker = NonNull::from(unsafe { &(*node.as_ptr()).waker });
    let next = unsafe { &mut (*node.as_ptr()).next };
    unsafe {
        core::hint::assert_unchecked(next.is_none());
        core::ptr::write(next, list.take())
    }

    *list = Some(node);
    waker
}

/// # Safety
/// `node.next` must be None
/// must not have any outgoing waiters
unsafe fn pop(list: &mut Link) -> Option<Waker> {
    list.take().map(|head| {
        let node = unsafe { head.read() };
        *list = node.next;
        let waker = node.waker;
        drop(unsafe { Box::from_raw(head.as_ptr().cast::<ManuallyDrop<Node>>()) });
        waker.into_inner()
    })
}

struct DropGuard<'a>(&'a mut Link);

impl<'a> Drop for DropGuard<'a> {
    fn drop(&mut self) {
        // Continue the same loop we do below. This only runs when a destructor has
        // panicked. If another one panics, this will abort.
        while unsafe { pop(self.0) }.is_some() {}
    }
}

fn drop_drain(mut list: Link) {
    // don't drop on the stack,
    // this is from std::collections::LinkedList

    // Wrap self so that if a destructor panics, we can try to keep looping
    let guard = DropGuard(&mut list);
    while unsafe { pop(guard.0) }.is_some() {}
    mem::forget(guard);
}

struct LatchLock<'a, S: RelaxStrategy> {
    latch: &'a Latch<S>,
    link: Link
}

impl<S: RelaxStrategy> LatchLock<'_, S> {
    fn as_mut_link(&mut self) -> &mut Link {
        // Safety: RawLink and Link have the same layout
        unsafe { &mut *((&raw mut self.link) as *mut Link) }
    }
}

impl<S: RelaxStrategy> Drop for LatchLock<'_, S> {
    fn drop(&mut self) {
        self.latch.ptr.store(to_state(self.link), Ordering::Release)
    }
}


impl<S: RelaxStrategy> Latch<S> {
    /// Creates a new instance of the [`Latch`] structure.
    /// # Returns
    ///
    /// A new instance of [`Latch`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use latch::{Latch, spin::Spin};
    /// let latch = Latch::<Spin>::new();
    /// ```
    pub const fn new() -> Self {
        Latch {
            ptr: AtomicUsize::new(NULL),
            _strategy: PhantomData,
        }
    }

    /// Checks if the [`Latch`] is in an "open" state.
    pub fn opened(&self) -> bool {
        is_open(self.ptr.load(Ordering::Acquire))
    }
    
    fn lock(&self) -> Option<LatchLock<S>> {
        loop {
            let ptr = self.ptr.fetch_or(LOCK_BIT, Ordering::AcqRel);

            if let Ok(prev_unlocked) = from_state(ptr) {
                return prev_unlocked.map(|link| LatchLock {
                    latch: self,
                    link,
                })
            }

            S::relax()
        }
    }

    /// Opens the latch if it is currently locked.
    pub fn open(&self) {
        if let Some(lock) = self.lock() {
            // open up the latch
            self.ptr.store(OPEN_LATCH, Ordering::Release);

            let lock = ManuallyDrop::new(lock);
            let mut list = lock.link;

            let guard = DropGuard(&mut list);
            while let Some(waker) = unsafe { pop(guard.0) } {
                waker.wake()
            }
        }
    }

    /// Waits for the latch to open, returning a `LatchWait` object that allows managing the latch wait state.
    ///
    /// # Returns
    /// A `LatchWait<S>` instance, which represents the state of the waiting node within the latch.
    pub fn wait(&self) -> LatchWait<S> {
        macro_rules! fast_fail {
            () => {
                return LatchWait {
                    entry: NonNull::dangling(),
                    latch: self,
                }
            };
        }

        if self.opened() {
            fast_fail!()
        }

        let node = Box::new(Node {
            next: None,
            waker: UnsafeCell::new(Waker::noop().clone()),
        });

        let Some(mut lock) = self.lock() else {
            fast_fail!()
        };


        // node unmodified, and .next is none
        let entry = unsafe { push(lock.as_mut_link(), node) };
        LatchWait {
            entry,
            latch: self,
        }
    }

    /// Blocks the current thread until the latch is open.
    #[cfg(feature = "std")]
    pub fn wait_sync(&self) {
        if self.opened() {
            return;
        }

        use std::sync::Arc;
        
        struct ThreadWaker(std::thread::Thread);

        impl std::task::Wake for ThreadWaker {
            fn wake(self: Arc<Self>) {
                self.wake_by_ref()
            }
            
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark()
            }
        }

        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let node = Box::new(Node {
            next: None,
            waker: UnsafeCell::new(waker),
        });

        let Some(mut lock) = self.lock() else {
            return;
        };


        // node unmodified, and .next is none
        unsafe { push(lock.as_mut_link(), node) };
        drop(lock);
        std::thread::park()
    }
}


/// The Future returned by [`Latch::wait`].
pub struct LatchWait<'a, S: RelaxStrategy> {
    entry: NonNull<UnsafeCell<Waker>>,
    latch: &'a Latch<S>
}

unsafe impl<S: RelaxStrategy> Send for Latch<S> {}
unsafe impl<S: RelaxStrategy> Sync for Latch<S> {}

// Latch is threadsafe, and the entries are only modified by the waiters that added them
unsafe impl<S: RelaxStrategy> Send for LatchWait<'_, S> {}
// no way to access immutably
unsafe impl<S: RelaxStrategy> Sync for LatchWait<'_, S> {}


impl<S: RelaxStrategy> Future for LatchWait<'_, S> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = Pin::into_inner(self);

        let Some(lock) = this.latch.lock() else {
            // fast path, already open
            return Poll::Ready(())
        };

        match cfg!(debug_assertions) {
            true => assert!(lock.link.is_some()),
            false => unsafe { core::hint::assert_unchecked(lock.link.is_some()) },
        }

        // this isn't a dangling pointer and is valid
        let waker_entry = unsafe { &mut *UnsafeCell::raw_get(this.entry.as_ptr()) };

        waker_entry.clone_from(cx.waker());

        drop(lock);

        Poll::Pending
    }
}


#[cfg(test)]
mod tests {
    extern crate std;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use super::*;
    use std::time::Duration;

    #[cfg(feature = "std")]
    use std::thread;
    use spin::relax::Loop;
    use spin::Spin;

    // Basic sync tests that don't rely on async runtimes
    #[test]
    fn test_latch_initial_state() {
        let latch = Latch::<Spin>::new();
        assert!(!latch.opened());
    }

    #[test]
    fn test_latch_open() {
        let latch = Latch::<Spin>::new();
        latch.open();
        assert!(latch.opened());
    }

    #[test]
    fn test_latch_open_idempotent() {
        let latch = Latch::<Spin>::new();
        latch.open();
        latch.open(); // Should be a no-op
        assert!(latch.opened());
    }


    const NUM_LATCHES: usize = 10;
    const WAITERS_PER_LATCH: usize = 3;

    fn latches() -> Vec<Arc<Latch<Spin>>> {
        (0..NUM_LATCHES)
            .map(|_| Arc::new(Latch::<Spin>::new()))
            .collect()
    }

    // Test using Loop strategy instead of Spin
    #[test]
    fn test_loop_strategy() {
        let latch = Latch::<Loop>::new();
        assert!(!latch.opened());
        latch.open();
        assert!(latch.opened());
    }

    // Test memory layout guarantees
    #[test]
    fn test_memory_layout() {
        assert_eq!(size_of::<Link>(), size_of::<usize>());
        assert_eq!(size_of::<Link>(), size_of::<Link>());
        assert!(align_of::<Node>() >= 2);
    }


    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_smol_latch_wait() {
        smol::block_on(async {
            let latch = Arc::new(Latch::<Spin>::new());
            let latch_clone = latch.clone();

            let handle = smol::spawn(async move {
                smol::Timer::after(Duration::from_millis(10)).await;
                latch_clone.open();
            });

            // Wait for the latch to be opened
            latch.wait().await;
            assert!(latch.opened());

            // Make sure the spawned task completes
            handle.await;
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_smol_multiple_waiters() {
        smol::block_on(async {
            let latch = Arc::new(Latch::<Spin>::new());

            // Create multiple waiters
            let mut waiter_tasks = Vec::new();
            for _ in 0..5 {
                let latch_clone = latch.clone();
                let handle = smol::spawn(async move {
                    latch_clone.wait().await;
                    assert!(latch_clone.opened());
                });
                waiter_tasks.push(handle);
            }

            // Small delay to ensure all waiters are registered
            smol::Timer::after(Duration::from_millis(5)).await;

            // Open the latch to release all waiters
            latch.open();

            // Wait for all tasks to complete
            for handle in waiter_tasks {
                handle.await;
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_smol_wait_already_open() {
        smol::block_on(async {
            let latch = Latch::<Spin>::new();
            latch.open();

            // Wait should return immediately
            latch.wait().await;
            assert!(latch.opened());
        });
    }


    #[test]
    fn test_pollster_wait_already_open() {
        pollster::block_on(async {
            let latch = Latch::<Spin>::new();
            latch.open();

            // Wait should return immediately
            latch.wait().await;
            assert!(latch.opened());
        });
    }

    #[test]
    fn test_pollster_open_after_wait() {
        // For this test, we'll use a std::thread since pollster doesn't have task spawning
        use std::thread;
        use std::time::Duration;

        let latch = Arc::new(Latch::<Spin>::new());
        let latch_clone = latch.clone();

        let thread_handle = thread::spawn(move || {
            // Sleep for a bit before opening the latch
            thread::sleep(Duration::from_millis(10));
            latch_clone.open();
        });

        // Wait for the latch in the main thread
        pollster::block_on(async {
            latch.wait().await;
            assert!(latch.opened());
        });

        thread_handle.join().unwrap();
    }


    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_async_std_latch_wait() {
        async_std::task::block_on(async {
            let latch = Arc::new(Latch::<Spin>::new());
            let latch_clone = latch.clone();

            let handle = async_std::task::spawn(async move {
                async_std::task::sleep(Duration::from_millis(10)).await;
                latch_clone.open();
            });

            // Wait for the latch to be opened
            latch.wait().await;
            assert!(latch.opened());

            // Make sure the spawned task completes
            handle.await;
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_async_std_multiple_waiters() {
        async_std::task::block_on(async {
            let latch = Arc::new(Latch::<Spin>::new());

            // Create multiple waiters
            let mut waiter_tasks = Vec::new();
            for _ in 0..5 {
                let latch_clone = latch.clone();
                let handle = async_std::task::spawn(async move {
                    latch_clone.wait().await;
                    assert!(latch_clone.opened());
                });
                waiter_tasks.push(handle);
            }

            // Small delay to ensure all waiters are registered
            async_std::task::sleep(Duration::from_millis(5)).await;

            // Open the latch to release all waiters
            latch.open();

            // Wait for all tasks to complete
            for handle in waiter_tasks {
                handle.await;
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_async_std_wait_already_open() {
        async_std::task::block_on(async {
            let latch = Latch::<Spin>::new();
            latch.open();

            // Wait should return immediately
            latch.wait().await;
            assert!(latch.opened());
        });
    }


    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tokio_latch_wait() {
        // Use a multi-threaded runtime
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let latch = Arc::new(Latch::<Spin>::new());
            let latch_clone = latch.clone();

            let handle = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                latch_clone.open();
            });

            // Wait for the latch to be opened
            latch.wait().await;
            assert!(latch.opened());

            // Make sure the spawned task completes
            handle.await.unwrap();
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tokio_multiple_waiters() {
        // Use a multi-threaded runtime
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let latch = Arc::new(Latch::<Spin>::new());

            // Create multiple waiters
            let mut waiter_tasks = Vec::new();
            for _ in 0..5 {
                let latch_clone = latch.clone();
                let handle = tokio::spawn(async move {
                    latch_clone.wait().await;
                    assert!(latch_clone.opened());
                });
                waiter_tasks.push(handle);
            }

            // Small delay to ensure all waiters are registered
            tokio::time::sleep(Duration::from_millis(5)).await;

            // Open the latch to release all waiters
            latch.open();

            // Wait for all tasks to complete
            for handle in waiter_tasks {
                handle.await.unwrap();
            }
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tokio_wait_already_open() {
        // Use a multi-threaded runtime
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let latch = Latch::<Spin>::new();
            latch.open();

            // Wait should return immediately
            latch.wait().await;
            assert!(latch.opened());
        });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tokio_concurrent_waiters_and_openers() {
        // Use a multi-threaded runtime
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            // Test multiple latches concurrently
            let latches = latches();

            let mut all_handles = Vec::new();

            // Create waiters for each latch
            for latch in &latches {
                for _ in 0..WAITERS_PER_LATCH {
                    let latch_clone = latch.clone();
                    let handle = tokio::spawn(async move {
                        latch_clone.wait().await;
                        assert!(latch_clone.opened());
                    });
                    all_handles.push(handle);
                }
            }

            // Small delay to ensure waiters are registered
            tokio::time::sleep(Duration::from_millis(5)).await;

            // Spawn tasks to open each latch
            for latch in &latches {
                let latch_clone = latch.clone();
                let handle = tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    latch_clone.open();
                });
                all_handles.push(handle);
            }

            // Wait for all tasks to complete
            for handle in all_handles {
                handle.await.unwrap();
            }

            // Verify all latches are open
            for latch in &latches {
                assert!(latch.opened());
            }
        });
    }

    // Add tests for the wait_sync method
    #[test]
    #[cfg(feature = "std")]
    fn test_wait_sync_basic() {
        let latch = Arc::new(Latch::<Spin>::new());
        let latch_clone = latch.clone();

        // Spawn a thread that will wait on the latch
        let waiter_thread = thread::spawn(move || {
            latch_clone.wait_sync();
            assert!(latch_clone.opened());
        });

        // Give the waiter thread time to start waiting
        thread::sleep(Duration::from_millis(10));

        // Open the latch
        latch.open();

        // Ensure the waiter thread completes
        waiter_thread.join().unwrap();
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_wait_sync_already_open() {
        let latch = Latch::<Spin>::new();

        // Open the latch before waiting
        latch.open();

        // Wait should return immediately
        latch.wait_sync();

        // Still open
        assert!(latch.opened());
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_wait_sync_multiple_waiters() {
        let latch = Arc::new(Latch::<Spin>::new());

        // Create multiple waiter threads
        let mut waiter_threads = Vec::new();
        for _ in 0..5 {
            let latch_clone = latch.clone();
            let thread = thread::spawn(move || {
                latch_clone.wait_sync();
                assert!(latch_clone.opened());
            });
            waiter_threads.push(thread);
        }

        // Give the waiter threads time to start waiting
        thread::sleep(Duration::from_millis(10));

        // Open the latch
        latch.open();

        // Ensure all waiter threads complete
        for thread in waiter_threads {
            thread.join().unwrap();
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_wait_sync_concurrent_operations() {
        let latches = latches();

        // Create waiter threads for each latch
        let mut all_threads = Vec::new();

        for latch in &latches {
            for _ in 0..WAITERS_PER_LATCH {
                let latch_clone = latch.clone();
                let thread = thread::spawn(move || {
                    latch_clone.wait_sync();
                    assert!(latch_clone.opened());
                });
                all_threads.push(thread);
            }
        }

        // Give the waiter threads time to start waiting
        thread::sleep(Duration::from_millis(20));

        // Create open threads for each latch
        let mut opener_threads = Vec::new();
        for latch in &latches {
            let latch_clone = latch.clone();
            let thread = thread::spawn(move || {
                // Add a small random delay to increase contention
                thread::sleep(Duration::from_millis(5));
                latch_clone.open();
            });
            opener_threads.push(thread);
        }

        // Wait for all opener threads to complete
        for thread in opener_threads {
            thread.join().unwrap();
        }

        // Wait for all waiter threads to complete
        for thread in all_threads {
            thread.join().unwrap();
        }

        // Verify all latches are open
        for latch in &latches {
            assert!(latch.opened());
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_wait_sync_and_async_interop() {
        let latch = Arc::new(Latch::<Spin>::new());
        let latch_clone_1 = latch.clone();
        let latch_clone_2 = latch.clone();

        // Spawn a thread that uses wait_sync
        let sync_thread = thread::spawn(move || {
            latch_clone_1.wait_sync();
            assert!(latch_clone_1.opened());
        });

        let async_handle = thread::spawn(move || pollster::block_on(async move {
            latch_clone_2.wait().await;
            assert!(latch_clone_2.opened());
        }));

        // Give threads time to start waiting
        thread::sleep(Duration::from_millis(10));

        // Open the latch
        latch.open();

        // Ensure both threads complete
        sync_thread.join().unwrap();
        async_handle.join().unwrap();
    }
}