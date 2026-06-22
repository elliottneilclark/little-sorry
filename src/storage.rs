//! Pluggable accumulator storage.
//!
//! At scale the matchers *are* the memory of a solve, and how their cells are
//! read and written decides whether many threads can cooperate. The storage
//! layer therefore abstracts a single accumulator cell behind plain
//! load/store, letting one update routine run unchanged over two backends:
//!
//! - [`Local`] — an ordinary `Cell`. No synchronization, no atomics. Because
//!   `Cell` is `!Sync`, a `Local`-backed matcher cannot be shared across
//!   threads, so single-threaded misuse is caught at compile time.
//! - [`Atomic`] — an `AtomicU32` (the `f32` bit pattern) read and written with
//!   `Relaxed` ordering, so updates can go through a shared `&self` from many
//!   worker threads.
//!
//! **On the concurrent backend's correctness.** Updates are a non-atomic
//! read-modify-write — load a cell, add, store — so two threads touching the
//! same cell can lose one update. We tolerate this deliberately. There is *no*
//! published theorem that CFR converges under racy accumulation; the
//! justification is the Hogwild result (Recht et al. 2011), where lock-free
//! stochastic-gradient updates converge *in expectation* under sparse,
//! bounded-staleness contention, reinforced by CFR's demonstrated tolerance of
//! the high-variance noise that Monte-Carlo CFR sampling injects. A lost update
//! is a small, bounded perturbation. This is empirically-motivated robustness —
//! strongest when threads touch mostly disjoint information sets — and is not a
//! guarantee. Single-threaded `Local` runs are fully deterministic; use them
//! when reproducibility matters.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// One `f32` accumulator cell: relaxed load/store, zero-initialized.
pub trait FloatCell: Default {
    /// Read the current value.
    fn load(&self) -> f32;
    /// Overwrite the value.
    fn store(&self, v: f32);
}

/// One iteration-counter cell: load and post-increment.
pub trait CounterCell: Default {
    /// Read the current count.
    fn load(&self) -> usize;
    /// Increment by one, returning the value *before* the increment.
    fn fetch_incr(&self) -> usize;
}

/// A choice of cell types for a matcher's accumulators and iteration clock.
pub trait StorageBackend {
    /// Cell type for the per-action accumulators.
    type Float: FloatCell;
    /// Cell type for the shared iteration counter.
    type Counter: CounterCell;
}

// ── Local: single-threaded, zero-overhead ───────────────────────────────────

impl FloatCell for Cell<f32> {
    fn load(&self) -> f32 {
        self.get()
    }
    fn store(&self, v: f32) {
        self.set(v);
    }
}

impl CounterCell for Cell<usize> {
    fn load(&self) -> usize {
        self.get()
    }
    fn fetch_incr(&self) -> usize {
        let prev = self.get();
        self.set(prev + 1);
        prev
    }
}

/// Single-threaded backend. `Cell` is `!Sync`, so a matcher built on it is
/// confined to one thread by the type system.
pub struct Local;

impl StorageBackend for Local {
    type Float = Cell<f32>;
    type Counter = Cell<usize>;
}

// ── Atomic: lock-free, Sync ──────────────────────────────────────────────────

impl FloatCell for AtomicU32 {
    fn load(&self) -> f32 {
        // We store the raw bit pattern, not a numeric encoding, so `to_bits`/
        // `from_bits` round-trips every value (including NaN and signed zero).
        f32::from_bits(AtomicU32::load(self, Ordering::Relaxed))
    }
    fn store(&self, v: f32) {
        AtomicU32::store(self, v.to_bits(), Ordering::Relaxed);
    }
}

impl CounterCell for AtomicUsize {
    fn load(&self) -> usize {
        AtomicUsize::load(self, Ordering::Relaxed)
    }
    fn fetch_incr(&self) -> usize {
        self.fetch_add(1, Ordering::Relaxed)
    }
}

/// Lock-free backend. Cells are `Sync`, so the matcher can be updated through a
/// shared reference from many threads (see the module note on benign races).
pub struct Atomic;

impl StorageBackend for Atomic {
    type Float = AtomicU32;
    type Counter = AtomicUsize;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_cell_roundtrips<C: FloatCell>() {
        let c = C::default();
        assert_eq!(c.load(), 0.0);
        for &v in &[1.0f32, -2.5, 0.0, 1e9, -1e-9] {
            c.store(v);
            assert_eq!(c.load(), v, "round-trip {v}");
        }
    }

    #[test]
    fn local_float_cell_roundtrips() {
        float_cell_roundtrips::<<Local as StorageBackend>::Float>();
    }

    #[test]
    fn atomic_float_cell_roundtrips_via_bits() {
        float_cell_roundtrips::<<Atomic as StorageBackend>::Float>();
    }

    fn counter_increments<C: CounterCell>() {
        let c = C::default();
        assert_eq!(c.load(), 0);
        assert_eq!(c.fetch_incr(), 0); // returns previous
        assert_eq!(c.fetch_incr(), 1);
        assert_eq!(c.load(), 2);
    }

    #[test]
    fn local_counter_increments() {
        counter_increments::<<Local as StorageBackend>::Counter>();
    }

    #[test]
    fn atomic_counter_increments() {
        counter_increments::<<Atomic as StorageBackend>::Counter>();
    }

    #[test]
    fn atomic_backend_cells_are_shareable() {
        // A compile-time proof that the concurrent backend's cells can be
        // shared across threads; the single-threaded `Local` cells cannot
        // (Cell is !Sync), which is exactly the guard we want.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<<Atomic as StorageBackend>::Float>();
        assert_send_sync::<<Atomic as StorageBackend>::Counter>();
    }
}
