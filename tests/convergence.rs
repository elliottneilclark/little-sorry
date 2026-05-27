//! Convergence tests for regret minimization algorithms.
//!
//! These tests verify that all CFR variants converge to Nash equilibrium
//! for the Rock-Paper-Scissors game.

use little_sorry::RegretMinimizer;
use little_sorry::cfr_plus::CfrPlusRegretMatcher;
use little_sorry::dcfr::DiscountedRegretMatcher;
use little_sorry::dcfr_plus::DcfrPlusRegretMatcher;
use little_sorry::linear_cfr::LinearCfrRegretMatcher;
use little_sorry::pcfr_plus::PcfrPlusRegretMatcher;
use little_sorry::pdcfr_plus::PdcfrPlusRegretMatcher;
use little_sorry::rps::RPSRunnerGeneric;

const NASH_WEIGHT: f32 = 1.0 / 3.0;
const CONVERGENCE_TOLERANCE: f32 = 0.05;
const NUM_ITERATIONS: usize = 10_000;

/// Computes the exploitability of a strategy in RPS.
fn compute_exploitability(weights: &[f32]) -> f32 {
    weights
        .iter()
        .map(|w| (w - NASH_WEIGHT).abs())
        .fold(0.0_f32, f32::max)
}

/// Helper to test convergence for any RegretMinimizer
fn test_convergence<M: RegretMinimizer>(name: &str) {
    let mut runner = RPSRunnerGeneric::<M>::new();
    let mut rng = rand::rng();

    for _ in 0..NUM_ITERATIONS {
        runner.run_one(&mut rng);
        runner.update_regret();
    }

    let weights = runner.best_weight();
    let exploitability = compute_exploitability(&weights);

    assert!(
        exploitability < CONVERGENCE_TOLERANCE,
        "{} did not converge to Nash equilibrium. Weights: {:?}, Exploitability: {}",
        name,
        weights,
        exploitability
    );
}

#[test]
fn test_cfr_plus_converges_to_nash_rps() {
    test_convergence::<CfrPlusRegretMatcher>("CFR+");
}

#[test]
fn test_dcfr_converges_to_nash_rps() {
    test_convergence::<DiscountedRegretMatcher>("DCFR");
}

#[test]
fn test_dcfr_plus_converges_to_nash_rps() {
    test_convergence::<DcfrPlusRegretMatcher>("DCFR+");
}

#[test]
fn test_linear_cfr_converges_to_nash_rps() {
    test_convergence::<LinearCfrRegretMatcher>("Linear CFR");
}

#[test]
fn test_pcfr_plus_converges_to_nash_rps() {
    test_convergence::<PcfrPlusRegretMatcher>("PCFR+");
}

#[test]
fn test_pdcfr_plus_converges_to_nash_rps() {
    test_convergence::<PdcfrPlusRegretMatcher>("PDCFR+");
}

#[test]
fn test_exploitability_converges_to_low_value() {
    let mut runner = RPSRunnerGeneric::<DiscountedRegretMatcher>::new();
    let mut rng = rand::rng();

    for _ in 0..10000 {
        runner.run_one(&mut rng);
        runner.update_regret();
    }
    let final_exploitability = compute_exploitability(&runner.best_weight());

    assert!(
        final_exploitability < CONVERGENCE_TOLERANCE,
        "Exploitability should be low after many iterations. Got: {}",
        final_exploitability
    );
}

#[test]
fn test_all_algorithms_converge() {
    const TEST_ITERATIONS: usize = 5000;

    let algorithms: Vec<(&str, f32)> = vec![
        (
            "CFR+",
            run_and_get_exploitability::<CfrPlusRegretMatcher>(TEST_ITERATIONS),
        ),
        (
            "DCFR",
            run_and_get_exploitability::<DiscountedRegretMatcher>(TEST_ITERATIONS),
        ),
        (
            "DCFR+",
            run_and_get_exploitability::<DcfrPlusRegretMatcher>(TEST_ITERATIONS),
        ),
        (
            "Linear CFR",
            run_and_get_exploitability::<LinearCfrRegretMatcher>(TEST_ITERATIONS),
        ),
        (
            "PCFR+",
            run_and_get_exploitability::<PcfrPlusRegretMatcher>(TEST_ITERATIONS),
        ),
        (
            "PDCFR+",
            run_and_get_exploitability::<PdcfrPlusRegretMatcher>(TEST_ITERATIONS),
        ),
    ];

    eprintln!("\nExploitability at {} iterations:", TEST_ITERATIONS);
    for (name, exploitability) in &algorithms {
        eprintln!("  {}: {:.4}", name, exploitability);
        assert!(
            *exploitability < 0.1,
            "{} should have low exploitability. Got: {}",
            name,
            exploitability
        );
    }
}

fn run_and_get_exploitability<M: RegretMinimizer>(iterations: usize) -> f32 {
    let mut runner = RPSRunnerGeneric::<M>::new();
    let mut rng = rand::rng();

    for _ in 0..iterations {
        runner.run_one(&mut rng);
        runner.update_regret();
    }

    compute_exploitability(&runner.best_weight())
}

/// Average regret is the CFR convergence diagnostic: once normalized by each
/// matcher's own regret weight it should shrink to a small value as the matcher
/// approaches the RPS Nash equilibrium, regardless of the weighting scheme.
fn assert_average_regret_converges<M: RegretMinimizer>(name: &str) {
    let mut runner = RPSRunnerGeneric::<M>::new();
    let mut rng = rand::rng();

    for _ in 0..NUM_ITERATIONS {
        runner.run_one(&mut rng);
        runner.update_regret();
    }

    let ar = runner.matcher_one.average_regret();
    assert!(ar >= 0.0, "{name}: average regret must be >= 0");
    assert!(
        ar < CONVERGENCE_TOLERANCE,
        "{name}: average regret should be small after convergence, got {ar}"
    );
}

#[test]
fn test_average_regret_converges_cfr_plus() {
    assert_average_regret_converges::<CfrPlusRegretMatcher>("CFR+");
}

#[test]
fn test_average_regret_converges_dcfr() {
    assert_average_regret_converges::<DiscountedRegretMatcher>("DCFR");
}

#[test]
fn test_average_regret_converges_dcfr_plus() {
    assert_average_regret_converges::<DcfrPlusRegretMatcher>("DCFR+");
}

// Regression guard: Linear CFR weights regret by `t`, so dividing by `T`
// (the old behavior) left this growing without bound (~43 at T=10k). With the
// correct `T(T+1)/2` normalizer it must converge like the other matchers.
#[test]
fn test_average_regret_converges_linear_cfr() {
    assert_average_regret_converges::<LinearCfrRegretMatcher>("LinearCFR");
}

#[test]
fn test_average_regret_converges_pcfr_plus() {
    assert_average_regret_converges::<PcfrPlusRegretMatcher>("PCFR+");
}

#[test]
fn test_average_regret_converges_pdcfr_plus() {
    assert_average_regret_converges::<PdcfrPlusRegretMatcher>("PDCFR+");
}
