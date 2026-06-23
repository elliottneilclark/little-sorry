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
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering};

/// A raw storage word: the integer widths a cell can hold (16- or 32-bit), each
/// with its lock-free atomic counterpart. The numeric *interpretation* of the
/// bits (f32 pattern, unit-fixed, scaled int) lives in the lane store, not here.
pub trait Word: Copy + Default {
    /// The atomic type that provides lock-free access to a `Self`-valued cell.
    type Atomic: Default;
    /// Load from an atomic cell using `Relaxed` ordering.
    fn load_atomic(a: &Self::Atomic) -> Self;
    /// Store into an atomic cell using `Relaxed` ordering.
    fn store_atomic(a: &Self::Atomic, w: Self);
}

impl Word for u16 {
    type Atomic = AtomicU16;
    fn load_atomic(a: &AtomicU16) -> u16 {
        a.load(Ordering::Relaxed)
    }
    fn store_atomic(a: &AtomicU16, w: u16) {
        a.store(w, Ordering::Relaxed);
    }
}

impl Word for u32 {
    type Atomic = AtomicU32;
    fn load_atomic(a: &AtomicU32) -> u32 {
        a.load(Ordering::Relaxed)
    }
    fn store_atomic(a: &AtomicU32, w: u32) {
        a.store(w, Ordering::Relaxed);
    }
}

/// One raw-word accumulator cell: relaxed load/store, zero-initialized.
pub trait AccumCell<W: Word>: Default {
    /// Read the current word value.
    fn load(&self) -> W;
    /// Overwrite the word value.
    fn store(&self, w: W);
}

impl<W: Word> AccumCell<W> for Cell<W> {
    fn load(&self) -> W {
        self.get()
    }
    fn store(&self, w: W) {
        self.set(w);
    }
}

/// Lock-free word cell for the concurrent backend.
pub struct AtomicCell<W: Word>(W::Atomic);

impl<W: Word> Default for AtomicCell<W> {
    fn default() -> Self {
        Self(W::Atomic::default())
    }
}

impl<W: Word> AccumCell<W> for AtomicCell<W> {
    fn load(&self) -> W {
        W::load_atomic(&self.0)
    }
    fn store(&self, w: W) {
        W::store_atomic(&self.0, w);
    }
}

/// One `f32` accumulator cell: relaxed load/store, zero-initialized.
pub trait FloatCell: Default {
    /// Read the current value.
    fn load(&self) -> f32;
    /// Overwrite the value.
    fn store(&self, v: f32);
}

/// One iteration-counter cell: load, post-increment, and direct store.
pub trait CounterCell: Default {
    /// Read the current count.
    fn load(&self) -> usize;
    /// Increment by one, returning the value *before* the increment.
    fn fetch_incr(&self) -> usize;
    /// Overwrite the counter with `v`.
    fn store(&self, v: usize);
}

/// A choice of cell types for a matcher's accumulators and iteration clock.
pub trait StorageBackend {
    /// Cell type for the per-action accumulators.
    type Float: FloatCell;
    /// Cell type for the shared iteration counter.
    type Counter: CounterCell;
    /// Cell type for a raw integer word of width `W`.
    type Cell<W: Word>: AccumCell<W>;
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
    fn store(&self, v: usize) {
        self.set(v);
    }
}

/// Single-threaded backend. `Cell` is `!Sync`, so a matcher built on it is
/// confined to one thread by the type system.
pub struct Local;

impl StorageBackend for Local {
    type Float = Cell<f32>;
    type Counter = Cell<usize>;
    type Cell<W: Word> = Cell<W>;
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
    fn store(&self, v: usize) {
        AtomicUsize::store(self, v, Ordering::Relaxed);
    }
}

/// Lock-free backend. Cells are `Sync`, so the matcher can be updated through a
/// shared reference from many threads (see the module note on benign races).
pub struct Atomic;

impl StorageBackend for Atomic {
    type Float = AtomicU32;
    type Counter = AtomicUsize;
    type Cell<W: Word> = AtomicCell<W>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_cells_round_trip_both_widths() {
        fn check<B: StorageBackend>() {
            let c16 = <B::Cell<u16>>::default();
            assert_eq!(c16.load(), 0);
            c16.store(40_000);
            assert_eq!(c16.load(), 40_000);

            let c32 = <B::Cell<u32>>::default();
            c32.store(0xDEAD_BEEF);
            assert_eq!(c32.load(), 0xDEAD_BEEF);
        }
        check::<Local>();
        check::<Atomic>();
    }

    #[test]
    fn counter_store_sets_value() {
        fn check<C: CounterCell>() {
            let c = C::default();
            c.store(42);
            assert_eq!(c.load(), 42);
            assert_eq!(c.fetch_incr(), 42);
        }
        check::<<Local as StorageBackend>::Counter>();
        check::<<Atomic as StorageBackend>::Counter>();
    }

    #[test]
    fn atomic_word_cells_are_shareable() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<<Atomic as StorageBackend>::Cell<u16>>();
        assert_send_sync::<<Atomic as StorageBackend>::Cell<u32>>();
    }

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
