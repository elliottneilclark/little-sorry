//! Convergence-equivalence test: `F32Full` and `HalfStrategy` layouts must
//! both converge to the RPS Nash equilibrium ([1/3, 1/3, 1/3]) and agree with
//! each other within tolerance.

#![cfg(feature = "rps")]
use little_sorry::lane::{F32Full, HalfStrategy};
use little_sorry::{BatchedMatcher, Dcfr, DiscountParams, Local};

fn solve_rps<L>(iters: usize) -> [f32; 3]
where
    L: little_sorry::lane::Layout<Dcfr, Local>,
{
    // RPS payoff: row action a vs column action b. Reuse the crate's reward table.
    let m = BatchedMatcher::<Dcfr, Local, L>::new(1, 3, DiscountParams::RECOMMENDED);
    let mut avg = [0.0f32; 3];
    let mut opp = [1.0f32 / 3.0; 3];
    let mut ev = [0.0f32; 1];
    for _ in 0..iters {
        m.update_batch(|a, _| rps_value(a, &opp), &mut ev);
        m.average_into(0, &mut avg);
        opp = avg; // self-play
    }
    avg
}

#[test]
fn half_strategy_converges_like_f32() {
    let a = solve_rps::<F32Full>(5_000);
    let b = solve_rps::<HalfStrategy>(5_000);
    for i in 0..3 {
        assert!(
            (a[i] - b[i]).abs() < 1e-2,
            "layout divergence at {i}: {a:?} vs {b:?}"
        );
        assert!(
            (b[i] - 1.0 / 3.0).abs() < 3e-2,
            "not near equilibrium: {b:?}"
        );
        assert!(
            (a[i] - 1.0 / 3.0).abs() < 3e-2,
            "F32Full not near equilibrium: {a:?}"
        );
    }
}

#[test]
fn half_both_converges_like_f32() {
    let a = solve_rps::<F32Full>(5_000);
    let b = solve_rps::<little_sorry::lane::HalfBoth>(5_000);
    for i in 0..3 {
        assert!(
            (a[i] - b[i]).abs() < 3e-2,
            "i16 layout divergence at {i}: {a:?} vs {b:?}"
        );
        assert!(
            (b[i] - 1.0 / 3.0).abs() < 5e-2,
            "not near equilibrium: {b:?}"
        );
        assert!(
            (a[i] - 1.0 / 3.0).abs() < 5e-2,
            "F32Full not near equilibrium: {a:?}"
        );
    }
}

// rps_value(a, opp): expected payoff of action a vs opponent distribution `opp`.
fn rps_value(a: usize, opp: &[f32; 3]) -> f32 {
    const PAY: [[f32; 3]; 3] = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
    (0..3).map(|b| opp[b] * PAY[a][b]).sum()
}
