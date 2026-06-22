//! Integration tests for the batched matcher: lock-free convergence parity and
//! the strategy-export round trip, both exercised through the public API.

use little_sorry::{
    Atomic, BatchedMatcher, Dcfr, DcfrPlus, DiscountParams, LinearCfr, Local, PcfrPlus, PdcfrPlus,
    PlusDiscount, UpdateRule, dequantize_dist, quantize_dist,
};
use std::sync::Arc;
use std::thread;

/// Rock-Paper-Scissors payoff: `reward[a] = Σ_b M[a][b] · opponent[b]`, where
/// `M` is the zero-sum win/lose/draw matrix. In symmetric self-play the
/// opponent is the matcher's own current strategy, and the unique equilibrium —
/// the average strategy this drives toward — is uniform `[1/3, 1/3, 1/3]`.
fn rps_reward(opponent: &[f32]) -> [f32; 3] {
    const M: [[f32; 3]; 3] = [
        [0.0, -1.0, 1.0], // rock
        [1.0, 0.0, -1.0], // paper
        [-1.0, 1.0, 0.0], // scissors
    ];
    let mut reward = [0.0f32; 3];
    for (a, row) in M.iter().enumerate() {
        reward[a] = row.iter().zip(opponent).map(|(&m, &p)| m * p).sum();
    }
    reward
}

/// Single-threaded RPS self-play to convergence, returning the average strategy.
fn solve_rps_local<R: UpdateRule>(params: R::Params, iters: usize) -> Vec<f32> {
    let m = BatchedMatcher::<R, Local>::new(1, 3, params);
    let mut strategy = [0.0f32; 3];
    for _ in 0..iters {
        m.current_into(0, &mut strategy);
        let reward = rps_reward(&strategy);
        m.update_row(0, |a| reward[a]);
    }
    let mut average = vec![0.0f32; 3];
    m.average_into(0, &mut average);
    average
}

fn assert_near_uniform(strategy: &[f32], tolerance: f32) {
    for (a, &p) in strategy.iter().enumerate() {
        assert!(
            (p - 1.0 / 3.0).abs() < tolerance,
            "action {a}: {p} not within {tolerance} of 1/3 (strategy {strategy:?})"
        );
    }
}

#[test]
fn lockfree_concurrent_solve_converges_like_single_threaded() {
    // Single-threaded reference: deterministic, must reach the equilibrium.
    let reference = solve_rps_local::<Dcfr>(DiscountParams::RECOMMENDED, 20_000);
    assert_near_uniform(&reference, 0.02);

    // Concurrent: several threads hammer one shared matcher through `&self`,
    // racing on its cells. We assert convergence to the same equilibrium within
    // tolerance — not bit-equality, which contention deliberately forgoes.
    let matcher = Arc::new(BatchedMatcher::<Dcfr, Atomic>::new(
        1,
        3,
        DiscountParams::RECOMMENDED,
    ));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let m = Arc::clone(&matcher);
            thread::spawn(move || {
                let mut strategy = [0.0f32; 3];
                for _ in 0..50_000 {
                    m.current_into(0, &mut strategy);
                    let reward = rps_reward(&strategy);
                    m.update_row(0, |a| reward[a]);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }

    let mut concurrent = vec![0.0f32; 3];
    matcher.average_into(0, &mut concurrent);
    assert_near_uniform(&concurrent, 0.05);
}

#[test]
fn strategy_export_round_trip_preserves_every_rule() {
    // Each rule solves RPS, then its average strategy is exported to u16 and
    // reloaded. Quantization must not meaningfully move the solution: every
    // component stays within one quantum, and the reloaded policy is still the
    // equilibrium. New rules join this table automatically.
    let dcfr = solve_rps_local::<Dcfr>(DiscountParams::RECOMMENDED, 20_000);
    let dcfr_plus = solve_rps_local::<DcfrPlus>(DcfrPlus::RECOMMENDED, 20_000);
    let linear = solve_rps_local::<LinearCfr>((), 20_000);
    let pcfr_plus = solve_rps_local::<PcfrPlus>((), 20_000);
    let pdcfr_plus = solve_rps_local::<PdcfrPlus>(PdcfrPlus::RECOMMENDED, 20_000);

    let table: [(&str, &[f32]); 5] = [
        ("DCFR", &dcfr),
        ("DCFR+", &dcfr_plus),
        ("LinearCFR", &linear),
        ("PCFR+", &pcfr_plus),
        ("PDCFR+", &pdcfr_plus),
    ];

    let quantum = 1.0 / 65535.0;
    for (name, average) in table {
        assert_near_uniform(average, 0.05);
        let reloaded = dequantize_dist::<u16>(&quantize_dist::<u16>(average));
        for (a, (&before, &after)) in average.iter().zip(&reloaded).enumerate() {
            assert!(
                (before - after).abs() < quantum,
                "{name} action {a}: {before} -> {after} exceeds one quantum"
            );
        }
        assert!(
            (reloaded.iter().sum::<f32>() - 1.0).abs() < 1e-6,
            "{name}: reloaded strategy not normalized"
        );
    }
}

/// A second matcher of the same rule with different parameters must coexist in
/// the same process — parameters are per instance, never global.
#[test]
fn matchers_of_one_rule_with_different_params_coexist() {
    let aggressive = solve_rps_local::<DcfrPlus>(
        PlusDiscount {
            alpha: 2.0,
            gamma: 6.0,
        },
        20_000,
    );
    let gentle = solve_rps_local::<DcfrPlus>(
        PlusDiscount {
            alpha: 1.0,
            gamma: 2.0,
        },
        20_000,
    );
    assert_near_uniform(&aggressive, 0.05);
    assert_near_uniform(&gentle, 0.05);
}
