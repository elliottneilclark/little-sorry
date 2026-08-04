//! Hot-path bench at the downstream consumer profile: one preflop decision
//! node — 169 hand-class rows × 3 actions, DCFR (α=3, β=0, γ=20), lock-free
//! `Atomic` cells, `HalfStrategyShared` layout.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use little_sorry::{Atomic, BatchedMatcher, Dcfr, DiscountParams, HalfStrategyShared, Scratch};

const NUM_ROWS: usize = 169;
const NUM_ACTIONS: usize = 3;

type Matcher = BatchedMatcher<Dcfr, Atomic, HalfStrategyShared>;

fn new_matcher() -> Matcher {
    Matcher::new(NUM_ROWS, NUM_ACTIONS, DiscountParams::new(3.0, 0.0, 20.0))
}

/// Fixed, cheap, row-varying reward so rows do not collapse to one strategy.
fn reward(action: usize, row: usize) -> f32 {
    [1.0f32, -0.5, 0.25][action] * (1.0 + (row % 13) as f32 * 0.05)
}

fn bench_update_batch(c: &mut Criterion) {
    let m = new_matcher();
    let mut expected = vec![0.0f32; NUM_ROWS];
    c.bench_function("batched_update_batch_169x3", |b| {
        b.iter(|| {
            m.update_batch(reward, &mut expected);
            black_box(expected[0])
        });
    });
}

fn bench_update_batch_with(c: &mut Criterion) {
    let m = new_matcher();
    let mut scratch = Scratch::new(NUM_ACTIONS);
    let mut expected = vec![0.0f32; NUM_ROWS];
    c.bench_function("batched_update_batch_with_169x3", |b| {
        b.iter(|| {
            m.update_batch_with(&mut scratch, reward, &mut expected);
            black_box(expected[0])
        });
    });
}

fn bench_current_into(c: &mut Criterion) {
    let m = new_matcher();
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
    bench_current_into
);
criterion_main!(benches);
