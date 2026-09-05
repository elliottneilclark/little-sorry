//! Concurrent `Atomic` equivalence tests for the half-width memory layouts.
//!
//! rs-poker drives `BatchedMatcher<Dcfr, Atomic>` under lock-free Hogwild
//! updates. The half-width strategy lanes maintain a RACED read-modify-write
//! denominator `W` (Relaxed, no CAS): unlike the f32 sum lane's benign
//! additive races, a raced denominator can bias the recency fraction high.
//! Single-threaded (`Local`) equivalence is covered elsewhere; this file adds
//! the CONCURRENT (`Atomic` + threads) equivalence that the real workload
//! depends on, plus `Send`/`Sync` assertions for every layout.

use little_sorry::lane::{
    F32Full, HalfBoth, HalfBothShared, HalfStrategy, HalfStrategyShared, Int32Full,
    Int32HalfShared, Int32NoAverage, Layout,
};
use little_sorry::{Atomic, BatchedMatcher, Dcfr, DiscountParams};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn atomic_half_layouts_are_send_sync() {
    assert_send_sync::<BatchedMatcher<Dcfr, Atomic, F32Full>>();
    assert_send_sync::<BatchedMatcher<Dcfr, Atomic, HalfStrategy>>();
    assert_send_sync::<BatchedMatcher<Dcfr, Atomic, HalfStrategyShared>>();
    assert_send_sync::<BatchedMatcher<Dcfr, Atomic, HalfBoth>>();
    assert_send_sync::<BatchedMatcher<Dcfr, Atomic, HalfBothShared>>();
}

#[test]
fn int32_layouts_are_send_sync() {
    assert_send_sync::<BatchedMatcher<Dcfr, Atomic, Int32Full>>();
    assert_send_sync::<BatchedMatcher<Dcfr, Atomic, Int32HalfShared>>();
    assert_send_sync::<BatchedMatcher<Dcfr, Atomic, Int32NoAverage>>();
}

/// rs-poker's DCFR discount parameters.
fn rs_poker_params() -> DiscountParams {
    DiscountParams::new(2.3, 0.0, 10.0)
}

/// A small table of DISTINCT, stationary per-row 3-action payoff vectors. Each
/// row sees a fixed reward vector (NOT self-play) so the regret-matching target
/// is deterministic and identical across layouts and thread counts. Distinct
/// per-row payoffs make any row-addressing bug surface as a mismatch.
const PAYOFFS: [[f32; 3]; 8] = [
    [1.0, -0.5, 0.2],
    [-0.3, 0.7, -0.4],
    [0.4, 0.4, -0.8],
    [-0.6, -0.1, 0.9],
    [0.2, -0.9, 0.6],
    [0.8, -0.2, -0.7],
    [-0.5, 0.5, 0.1],
    [0.3, -0.4, 0.0],
];

/// Rows whose payoff ties two actions (`PAYOFFS[2]`: actions 0 and 1 both pay
/// 0.4). On such a row the instantaneous regret of the tied pair is exactly
/// zero once the row sits on the tie, so f32 stays frozen at an exact 50/50
/// while the int32 lane's stochastic rounding random-walks along the flat
/// direction of that non-unique equilibrium — an arbitrary split of the tied
/// pair, not a lane error. The int32 comparisons skip these rows; every other
/// row matches f32 exactly under both `Local` and 8-thread `Atomic` runs.
fn is_tied_row(row: usize) -> bool {
    row % PAYOFFS.len() == 2
}

fn reward(action: usize, row: usize) -> f32 {
    PAYOFFS[row % PAYOFFS.len()][action]
}

/// Spawn `threads` workers, all sharing one `Arc<BatchedMatcher<…, Atomic, L>>`,
/// each advancing every row `iters_per_thread` ticks with the stationary
/// `reward`. Returns the per-row average strategy (rows × actions), or the
/// current strategy when `current` (for layouts with no average lane).
fn solve_concurrent<L>(
    threads: usize,
    iters_per_thread: usize,
    rows: usize,
    params: DiscountParams,
    current: bool,
) -> Vec<Vec<f32>>
where
    L: Layout<Dcfr, Atomic>,
    L::Regret: Send + Sync,
    L::Strategy: Send + Sync,
{
    let m = std::sync::Arc::new(BatchedMatcher::<Dcfr, Atomic, L>::new(rows, 3, params));

    std::thread::scope(|scope| {
        for _ in 0..threads {
            let m = std::sync::Arc::clone(&m);
            scope.spawn(move || {
                // Output scratch for `update_batch` — written each tick, then discarded.
                let mut expected = vec![0.0f32; rows];
                for _ in 0..iters_per_thread {
                    m.update_batch(reward, &mut expected);
                }
            });
        }
    });

    (0..rows)
        .map(|row| {
            let mut out = vec![0.0f32; 3];
            if current {
                m.current_into(row, &mut out);
            } else {
                m.average_into(row, &mut out);
            }
            out
        })
        .collect()
}

/// Assert every component of `got` is within `tol` of `baseline` on `rows`,
/// printing the worst gap.
fn assert_rows_match(
    name: &str,
    rows: impl Iterator<Item = usize>,
    got: &[Vec<f32>],
    baseline: &[Vec<f32>],
    tol: f32,
) {
    let mut max_gap = 0.0f32;
    for row in rows {
        for a in 0..3 {
            let gap = (got[row][a] - baseline[row][a]).abs();
            max_gap = max_gap.max(gap);
            assert!(
                gap <= tol,
                "{name}: row {row} action {a} gap {gap} exceeds {tol} (got={}, f32={})",
                got[row][a],
                baseline[row][a],
            );
        }
    }
    eprintln!("{name}: max per-component gap vs f32 = {max_gap}");
}

#[test]
fn half_layouts_match_f32_under_concurrency() {
    const THREADS: usize = 8;
    const ITERS: usize = 4_000; // 8 * 4_000 = 32_000 ticks per row
    const ROWS: usize = 16;
    // Per-component tolerance. Observed worst per-component gaps over 6 runs:
    // HalfStrategy 0.0020, HalfStrategyShared 0.0013, HalfBoth 0.0076,
    // HalfBothShared 0.0013 — all well under 5e-2, which leaves comfortable
    // Hogwild headroom while still catching gross regression (a raced
    // denominator would blow far past this).
    const TOL: f32 = 5e-2;

    let params = rs_poker_params();
    let baseline = solve_concurrent::<F32Full>(THREADS, ITERS, ROWS, params, false);

    assert_rows_match(
        "HalfStrategy",
        0..ROWS,
        &solve_concurrent::<HalfStrategy>(THREADS, ITERS, ROWS, params, false),
        &baseline,
        TOL,
    );
    assert_rows_match(
        "HalfStrategyShared",
        0..ROWS,
        &solve_concurrent::<HalfStrategyShared>(THREADS, ITERS, ROWS, params, false),
        &baseline,
        TOL,
    );
    assert_rows_match(
        "HalfBoth",
        0..ROWS,
        &solve_concurrent::<HalfBoth>(THREADS, ITERS, ROWS, params, false),
        &baseline,
        TOL,
    );
    assert_rows_match(
        "HalfBothShared",
        0..ROWS,
        &solve_concurrent::<HalfBothShared>(THREADS, ITERS, ROWS, params, false),
        &baseline,
        TOL,
    );
}

#[test]
fn int32_layouts_match_f32_under_concurrency() {
    const THREADS: usize = 8;
    const ITERS: usize = 4_000;
    const ROWS: usize = 16;
    const TOL: f32 = 5e-2;

    let params = rs_poker_params();
    let baseline = solve_concurrent::<F32Full>(THREADS, ITERS, ROWS, params, false);
    assert_rows_match(
        "Int32Full",
        (0..ROWS).filter(|&r| !is_tied_row(r)),
        &solve_concurrent::<Int32Full>(THREADS, ITERS, ROWS, params, false),
        &baseline,
        TOL,
    );
    assert_rows_match(
        "Int32HalfShared",
        (0..ROWS).filter(|&r| !is_tied_row(r)),
        &solve_concurrent::<Int32HalfShared>(THREADS, ITERS, ROWS, params, false),
        &baseline,
        TOL,
    );
}

/// `Int32NoAverage` has no average to compare, so its concurrent check is on
/// the current strategy every row settles to under the stationary payoffs.
#[test]
fn int32_no_average_matches_f32_current_under_concurrency() {
    const THREADS: usize = 8;
    const ITERS: usize = 4_000;
    const ROWS: usize = 16;
    const TOL: f32 = 5e-2;

    let params = rs_poker_params();
    let baseline = solve_concurrent::<F32Full>(THREADS, ITERS, ROWS, params, true);
    let got = solve_concurrent::<Int32NoAverage>(THREADS, ITERS, ROWS, params, true);
    assert_rows_match(
        "Int32NoAverage",
        (0..ROWS).filter(|&r| !is_tied_row(r)),
        &got,
        &baseline,
        TOL,
    );
}

/// Time-varying reward: on top of each row's fixed base payoff, a `+BIAS` bonus
/// rotates through the three actions, advancing one step every `period`
/// iterations. The favored action — and therefore the regret-matched current
/// strategy — SHIFTS over the run, so the exported average σ̄ tracks a MOVING
/// target. This is exactly the regime where the u16 running-average lane's
/// recency weighting (and its RACED `W` denominator) can bias the result, which
/// the stationary test — where every layout converges to one fixed point —
/// cannot exercise.
///
/// The base payoffs keep a clear long-run ordering, so the average does not sit
/// on a knife-edge near-tie. (A FULL cyclic rotation of the payoff vector, which
/// drives the long-run average onto a 3-way near-tie, makes the u16 strategy
/// lane's recency bias blow up to 0.1–0.45 per component — a real finding about
/// the lossy running average, documented in the followup report. This test
/// deliberately uses the additive-bias variant so it is a non-flaky equivalence
/// gate while still genuinely time-varying the target.)
const BIAS: f32 = 0.15;
fn reward_ns(action: usize, row: usize, iter: usize, period: usize) -> f32 {
    let base = PAYOFFS[row % PAYOFFS.len()];
    let favored = (iter / period) % 3;
    base[action] + if action == favored { BIAS } else { 0.0 }
}

/// Like `solve_concurrent`, but each worker drives the matcher with the
/// time-varying `reward_ns` keyed off its LOCAL iteration index, so the target
/// the average tracks moves over the run. Thread count, iteration count, rows,
/// params, and the reward function are IDENTICAL across every layout, so the
/// only difference between baseline and half layouts is the lane storage.
fn solve_concurrent_nonstationary<L>(
    threads: usize,
    iters_per_thread: usize,
    rows: usize,
    period: usize,
    params: DiscountParams,
) -> Vec<Vec<f32>>
where
    L: Layout<Dcfr, Atomic>,
    L::Regret: Send + Sync,
    L::Strategy: Send + Sync,
{
    let m = std::sync::Arc::new(BatchedMatcher::<Dcfr, Atomic, L>::new(rows, 3, params));

    std::thread::scope(|scope| {
        for _ in 0..threads {
            let m = std::sync::Arc::clone(&m);
            scope.spawn(move || {
                let mut expected = vec![0.0f32; rows];
                for iter in 0..iters_per_thread {
                    m.update_batch(
                        |action, row| reward_ns(action, row, iter, period),
                        &mut expected,
                    );
                }
            });
        }
    });

    (0..rows)
        .map(|row| {
            let mut out = vec![0.0f32; 3];
            m.average_into(row, &mut out);
            out
        })
        .collect()
}

#[test]
fn half_layouts_match_f32_under_concurrency_nonstationary() {
    const THREADS: usize = 8;
    const ITERS: usize = 4_000;
    const ROWS: usize = 16;
    // Rotate the favored action every quarter of a thread's run, so the average
    // tracks four distinct phases — recency weighting genuinely matters here.
    const PERIOD: usize = ITERS / 4;
    // Per-component tolerance vs the f32 baseline. Under this time-varying load
    // the gap is expected to be LARGER than the stationary test (that's the
    // point): the running average chases a moving target, so a biased recency
    // fraction from the raced `W` has somewhere to show. Observed worst
    // per-component gaps over 50 runs (8 threads, 4_000 iters/thread):
    //   HalfStrategy       0.035
    //   HalfStrategyShared 0.034
    //   HalfBoth           0.033
    //   HalfBothShared     0.034
    // i.e. non-stationary (~0.035) >> stationary (~0.0003), confirming the test
    // exercises recency weighting; and the Shared (single-W) variants are NOT
    // materially worse than their per-row counterparts — the contention on one
    // shared `W` does not amplify the bias here. 5e-2 sits comfortably above the
    // 50-run worst with Hogwild headroom while still catching gross regression.
    const TOL: f32 = 5e-2;

    let params = rs_poker_params();
    let baseline = solve_concurrent_nonstationary::<F32Full>(THREADS, ITERS, ROWS, PERIOD, params);

    assert_rows_match(
        "HalfStrategy (non-stationary)",
        0..ROWS,
        &solve_concurrent_nonstationary::<HalfStrategy>(THREADS, ITERS, ROWS, PERIOD, params),
        &baseline,
        TOL,
    );
    assert_rows_match(
        "HalfStrategyShared (non-stationary)",
        0..ROWS,
        &solve_concurrent_nonstationary::<HalfStrategyShared>(THREADS, ITERS, ROWS, PERIOD, params),
        &baseline,
        TOL,
    );
    assert_rows_match(
        "HalfBoth (non-stationary)",
        0..ROWS,
        &solve_concurrent_nonstationary::<HalfBoth>(THREADS, ITERS, ROWS, PERIOD, params),
        &baseline,
        TOL,
    );
    assert_rows_match(
        "HalfBothShared (non-stationary)",
        0..ROWS,
        &solve_concurrent_nonstationary::<HalfBothShared>(THREADS, ITERS, ROWS, PERIOD, params),
        &baseline,
        TOL,
    );
}

#[test]
fn int32_layouts_match_f32_under_concurrency_nonstationary() {
    const THREADS: usize = 8;
    const ITERS: usize = 4_000;
    const ROWS: usize = 16;
    const PERIOD: usize = ITERS / 4;
    const TOL: f32 = 5e-2;

    let params = rs_poker_params();
    let baseline = solve_concurrent_nonstationary::<F32Full>(THREADS, ITERS, ROWS, PERIOD, params);
    assert_rows_match(
        "Int32Full (non-stationary)",
        (0..ROWS).filter(|&r| !is_tied_row(r)),
        &solve_concurrent_nonstationary::<Int32Full>(THREADS, ITERS, ROWS, PERIOD, params),
        &baseline,
        TOL,
    );
    assert_rows_match(
        "Int32HalfShared (non-stationary)",
        (0..ROWS).filter(|&r| !is_tied_row(r)),
        &solve_concurrent_nonstationary::<Int32HalfShared>(THREADS, ITERS, ROWS, PERIOD, params),
        &baseline,
        TOL,
    );
}

#[test]
fn reset_average_through_halfstrategy_dispatch() {
    use little_sorry::Local;

    let m = BatchedMatcher::<Dcfr, Local, HalfStrategy>::new(2, 3, DiscountParams::RECOMMENDED);
    let mut expected = [0.0f32; 2];
    for _ in 0..200 {
        m.update_batch(|action, _row| [1.0, -0.5, 0.2][action], &mut expected);
    }

    // Reset the strategy lane through the trait dispatch, then confirm every
    // row's average is uniform again (exercises `U16AvgStrategy::reset`).
    m.reset_average();

    for row in 0..2 {
        let mut probs = [0.0f32; 3];
        m.average_into(row, &mut probs);
        for &p in &probs {
            assert!(
                (p - 1.0 / 3.0).abs() < 1e-6,
                "row {row} not uniform after reset_average: {probs:?}",
            );
        }
    }
}
