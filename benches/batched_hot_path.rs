//! Hot-path bench at the downstream consumer profile: one preflop decision
//! node — 169 hand-class rows × 3 actions, DCFR (α=3, β=0, γ=20), lock-free
//! `Atomic` cells — across the layouts a solver would choose between, plus
//! the masked (partial-action) update path.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use little_sorry::{
    Atomic, BatchedMatcher, Dcfr, DiscountParams, HalfStrategyShared, Int32Full, Int32HalfShared,
    Int32NoAverage, Layout, Scratch,
};

const NUM_ROWS: usize = 169;
const NUM_ACTIONS: usize = 3;

fn new_matcher<L: Layout<Dcfr, Atomic>>() -> BatchedMatcher<Dcfr, Atomic, L> {
    BatchedMatcher::new(NUM_ROWS, NUM_ACTIONS, DiscountParams::new(3.0, 0.0, 20.0))
}

/// Fixed, cheap, row-varying reward so rows do not collapse to one strategy.
fn reward(action: usize, row: usize) -> f32 {
    [1.0f32, -0.5, 0.25][action] * (1.0 + (row % 13) as f32 * 0.05)
}

/// ~25% of (action, row) cells inactive, spread across rows.
fn quarter_inactive(action: usize, row: usize) -> bool {
    (action + row) % 4 != 0
}

fn bench_update_batch(c: &mut Criterion) {
    let m = new_matcher::<HalfStrategyShared>();
    let mut expected = vec![0.0f32; NUM_ROWS];
    c.bench_function("batched_update_batch_169x3", |b| {
        b.iter(|| {
            m.update_batch(reward, &mut expected);
            black_box(expected[0])
        });
    });
}

fn bench_update_batch_with_layout<L: Layout<Dcfr, Atomic>>(c: &mut Criterion, name: &str) {
    let m = new_matcher::<L>();
    let mut scratch = Scratch::new(NUM_ACTIONS);
    let mut expected = vec![0.0f32; NUM_ROWS];
    c.bench_function(&format!("batched_update_batch_with_169x3/{name}"), |b| {
        b.iter(|| {
            m.update_batch_with(&mut scratch, reward, &mut expected);
            black_box(expected[0])
        });
    });
}

fn bench_update_batch_with(c: &mut Criterion) {
    bench_update_batch_with_layout::<HalfStrategyShared>(c, "HalfStrategyShared");
    bench_update_batch_with_layout::<Int32Full>(c, "Int32Full");
    bench_update_batch_with_layout::<Int32HalfShared>(c, "Int32HalfShared");
    bench_update_batch_with_layout::<Int32NoAverage>(c, "Int32NoAverage");
}

fn bench_masked_layout<L: Layout<Dcfr, Atomic>>(c: &mut Criterion, name: &str) {
    let m = new_matcher::<L>();
    let mut scratch = Scratch::new(NUM_ACTIONS);
    let mut expected = vec![0.0f32; NUM_ROWS];
    c.bench_function(
        &format!("batched_update_batch_masked_all_active_169x3/{name}"),
        |b| {
            b.iter(|| {
                m.update_batch_masked_with(&mut scratch, reward, |_, _| true, &mut expected);
                black_box(expected[0])
            });
        },
    );
    c.bench_function(
        &format!("batched_update_batch_masked_25pct_169x3/{name}"),
        |b| {
            b.iter(|| {
                m.update_batch_masked_with(&mut scratch, reward, quarter_inactive, &mut expected);
                black_box(expected[0])
            });
        },
    );
}

fn bench_masked(c: &mut Criterion) {
    bench_masked_layout::<HalfStrategyShared>(c, "HalfStrategyShared");
    bench_masked_layout::<Int32NoAverage>(c, "Int32NoAverage");
}

fn bench_current_into(c: &mut Criterion) {
    let m = new_matcher::<HalfStrategyShared>();
    let mut expected = vec![0.0f32; NUM_ROWS];
    for _ in 0..100 {
        m.update_batch(reward, &mut expected);
    }
    let mut out = [0.0f32; NUM_ACTIONS];
    let mut row = 0;
    c.bench_function("batched_current_into_169x3", |b| {
        b.iter(|| {
            row = (row + 1) % NUM_ROWS;
            m.current_into(row, &mut out);
            black_box(out[0])
        });
    });
}

criterion_group!(
    benches,
    bench_update_batch,
    bench_update_batch_with,
    bench_masked,
    bench_current_into
);
criterion_main!(benches);
