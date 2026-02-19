#![deny(clippy::all)]

use little_sorry::RegretMinimizer;
use little_sorry::cfr_plus::CfrPlusRegretMatcher;
use little_sorry::dcfr::DiscountedRegretMatcher;
use little_sorry::dcfr_plus::DcfrPlusRegretMatcher;
use little_sorry::linear_cfr::LinearCfrRegretMatcher;
use little_sorry::pcfr_plus::PcfrPlusRegretMatcher;
use little_sorry::pdcfr_plus::PdcfrPlusRegretMatcher;
use little_sorry::rps::RPSRunnerGeneric;

fn compute_exploitability(weights: &[f32]) -> f32 {
    const NASH: f32 = 1.0 / 3.0;
    weights
        .iter()
        .map(|w| (w - NASH).abs())
        .fold(0.0_f32, f32::max)
}

fn run_algorithm<M: RegretMinimizer>(iterations: usize) -> f32 {
    let mut runner = RPSRunnerGeneric::<M>::new();
    let mut rng = rand::rng();
    for _ in 0..iterations {
        runner.run_one(&mut rng);
        runner.update_regret();
    }
    compute_exploitability(&runner.best_weight())
}

fn main() {
    let iterations = [1000, 2500, 5000, 10000, 25000];

    println!("Exploitability by iteration count (lower is better, Nash = 0.0):\n");
    println!(
        "{:15} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Algorithm", "1K", "2.5K", "5K", "10K", "25K"
    );
    println!("{}", "-".repeat(75));

    // Run each algorithm
    print!("{:15}", "CFR+");
    for &iters in &iterations {
        let avg: f32 = (0..3)
            .map(|_| run_algorithm::<CfrPlusRegretMatcher>(iters))
            .sum::<f32>()
            / 3.0;
        print!(" {:>10.4}", avg);
    }
    println!();

    print!("{:15}", "DCFR");
    for &iters in &iterations {
        let avg: f32 = (0..3)
            .map(|_| run_algorithm::<DiscountedRegretMatcher>(iters))
            .sum::<f32>()
            / 3.0;
        print!(" {:>10.4}", avg);
    }
    println!();

    print!("{:15}", "DCFR+");
    for &iters in &iterations {
        let avg: f32 = (0..3)
            .map(|_| run_algorithm::<DcfrPlusRegretMatcher>(iters))
            .sum::<f32>()
            / 3.0;
        print!(" {:>10.4}", avg);
    }
    println!();

    print!("{:15}", "Linear CFR");
    for &iters in &iterations {
        let avg: f32 = (0..3)
            .map(|_| run_algorithm::<LinearCfrRegretMatcher>(iters))
            .sum::<f32>()
            / 3.0;
        print!(" {:>10.4}", avg);
    }
    println!();

    print!("{:15}", "PCFR+");
    for &iters in &iterations {
        let avg: f32 = (0..3)
            .map(|_| run_algorithm::<PcfrPlusRegretMatcher>(iters))
            .sum::<f32>()
            / 3.0;
        print!(" {:>10.4}", avg);
    }
    println!();

    print!("{:15}", "PDCFR+");
    for &iters in &iterations {
        let avg: f32 = (0..3)
            .map(|_| run_algorithm::<PdcfrPlusRegretMatcher>(iters))
            .sum::<f32>()
            / 3.0;
        print!(" {:>10.4}", avg);
    }
    println!();
}
