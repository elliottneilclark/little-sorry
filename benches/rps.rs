use criterion::{Criterion, criterion_group, criterion_main};

use std::time::Instant;

use little_sorry::RegretMinimizer;
use little_sorry::cfr_plus::CfrPlusRegretMatcher;
use little_sorry::dcfr::DiscountedRegretMatcher;
use little_sorry::dcfr_plus::DcfrPlusRegretMatcher;
use little_sorry::linear_cfr::LinearCfrRegretMatcher;
use little_sorry::pcfr_plus::PcfrPlusRegretMatcher;
use little_sorry::pdcfr_plus::PdcfrPlusRegretMatcher;
use little_sorry::rps::{RPSRunner, RPSRunnerGeneric};

pub fn bench_cfr_plus(c: &mut Criterion) {
    c.bench_function("cfr_plus_iter", move |b| {
        b.iter_custom(|iters| {
            let mut runner = RPSRunner::new().unwrap();
            let mut rng = rand::rng();
            let start = Instant::now();
            for _i in 0..iters {
                runner.run_one(&mut rng);
                runner.update_regret().unwrap();
            }
            start.elapsed()
        })
    });
}

pub fn bench_dcfr_recommended(c: &mut Criterion) {
    c.bench_function("dcfr_recommended_iter", move |b| {
        b.iter_custom(|iters| {
            let mut runner = RPSRunnerGeneric::<DiscountedRegretMatcher>::new().unwrap();
            let mut rng = rand::rng();
            let start = Instant::now();
            for _i in 0..iters {
                runner.run_one(&mut rng);
                runner.update_regret().unwrap();
            }
            start.elapsed()
        })
    });
}

pub fn bench_dcfr_plus(c: &mut Criterion) {
    c.bench_function("dcfr_plus_iter", move |b| {
        b.iter_custom(|iters| {
            let mut runner = RPSRunnerGeneric::<DcfrPlusRegretMatcher>::new().unwrap();
            let mut rng = rand::rng();
            let start = Instant::now();
            for _i in 0..iters {
                runner.run_one(&mut rng);
                runner.update_regret().unwrap();
            }
            start.elapsed()
        })
    });
}

pub fn bench_linear_cfr(c: &mut Criterion) {
    c.bench_function("linear_cfr_iter", move |b| {
        b.iter_custom(|iters| {
            let mut runner = RPSRunnerGeneric::<LinearCfrRegretMatcher>::new().unwrap();
            let mut rng = rand::rng();
            let start = Instant::now();
            for _i in 0..iters {
                runner.run_one(&mut rng);
                runner.update_regret().unwrap();
            }
            start.elapsed()
        })
    });
}

pub fn bench_pcfr_plus(c: &mut Criterion) {
    c.bench_function("pcfr_plus_iter", move |b| {
        b.iter_custom(|iters| {
            let mut runner = RPSRunnerGeneric::<PcfrPlusRegretMatcher>::new().unwrap();
            let mut rng = rand::rng();
            let start = Instant::now();
            for _i in 0..iters {
                runner.run_one(&mut rng);
                runner.update_regret().unwrap();
            }
            start.elapsed()
        })
    });
}

pub fn bench_pdcfr_plus(c: &mut Criterion) {
    c.bench_function("pdcfr_plus_iter", move |b| {
        b.iter_custom(|iters| {
            let mut runner = RPSRunnerGeneric::<PdcfrPlusRegretMatcher>::new().unwrap();
            let mut rng = rand::rng();
            let start = Instant::now();
            for _i in 0..iters {
                runner.run_one(&mut rng);
                runner.update_regret().unwrap();
            }
            start.elapsed()
        })
    });
}

fn compute_exploitability(weights: &[f32]) -> f32 {
    const NASH_WEIGHT: f32 = 1.0 / 3.0;
    weights
        .iter()
        .map(|w| (w - NASH_WEIGHT).abs())
        .fold(0.0_f32, f32::max)
}

fn bench_convergence_for<M: RegretMinimizer>(name: &str, c: &mut Criterion) {
    c.bench_function(&format!("convergence_{}", name), |b| {
        b.iter_custom(|_iters| {
            let mut runner = RPSRunnerGeneric::<M>::new().unwrap();
            let mut rng = rand::rng();
            let start = Instant::now();

            loop {
                runner.run_one(&mut rng);
                runner.update_regret().unwrap();

                if compute_exploitability(&runner.best_weight()) < 0.01 {
                    break;
                }
            }
            start.elapsed()
        })
    });
}

pub fn bench_convergence_comparison(c: &mut Criterion) {
    bench_convergence_for::<CfrPlusRegretMatcher>("cfr_plus", c);
    bench_convergence_for::<DiscountedRegretMatcher>("dcfr", c);
    bench_convergence_for::<DcfrPlusRegretMatcher>("dcfr_plus", c);
    bench_convergence_for::<LinearCfrRegretMatcher>("linear_cfr", c);
    bench_convergence_for::<PcfrPlusRegretMatcher>("pcfr_plus", c);
    bench_convergence_for::<PdcfrPlusRegretMatcher>("pdcfr_plus", c);
}

criterion_group!(
    benches,
    bench_cfr_plus,
    bench_dcfr_recommended,
    bench_dcfr_plus,
    bench_linear_cfr,
    bench_pcfr_plus,
    bench_pdcfr_plus,
    bench_convergence_comparison
);
criterion_main!(benches);
