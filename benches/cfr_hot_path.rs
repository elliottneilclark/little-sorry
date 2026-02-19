use criterion::{Criterion, criterion_group, criterion_main};

use little_sorry::RegretMinimizer;
use little_sorry::cfr_plus::CfrPlusRegretMatcher;
use little_sorry::dcfr::DiscountedRegretMatcher;
use little_sorry::dcfr_plus::DcfrPlusRegretMatcher;
use little_sorry::linear_cfr::LinearCfrRegretMatcher;
use little_sorry::pcfr_plus::PcfrPlusRegretMatcher;
use little_sorry::pdcfr_plus::PdcfrPlusRegretMatcher;

fn bench_update_only<M: RegretMinimizer>(name: &str, c: &mut Criterion) {
    let num_experts = 52;
    let mut rewards = vec![-100.0_f32; num_experts];
    rewards[0] = 10.0;
    rewards[1] = 50.0;
    rewards[51] = 200.0;

    c.bench_function(&format!("{}_update_x1000", name), |b| {
        b.iter(|| {
            let mut matcher = M::new(num_experts);
            for _ in 0..1000 {
                matcher.update_regret(&rewards);
            }
        });
    });
}

fn bench_update_then_best_weight<M: RegretMinimizer>(name: &str, c: &mut Criterion) {
    let num_experts = 52;
    let mut rewards = vec![-100.0_f32; num_experts];
    rewards[0] = 10.0;
    rewards[1] = 50.0;
    rewards[51] = 200.0;

    c.bench_function(&format!("{}_update_x1000_best_weight", name), |b| {
        b.iter(|| {
            let mut matcher = M::new(num_experts);
            for _ in 0..1000 {
                matcher.update_regret(&rewards);
            }
            let weights = matcher.best_weight();
            let _ = weights[0] + weights[1] + weights[51];
        });
    });
}

fn bench_update_and_sample<M: RegretMinimizer>(name: &str, c: &mut Criterion) {
    let num_experts = 52;
    let mut rewards = vec![-100.0_f32; num_experts];
    rewards[0] = 10.0;
    rewards[1] = 50.0;
    rewards[51] = 200.0;

    c.bench_function(&format!("{}_update_sample_x1000", name), |b| {
        b.iter(|| {
            let mut rng = rand::rng();
            let mut matcher = M::new(num_experts);
            for _ in 0..1000 {
                matcher.update_regret(&rewards);
                let _ = matcher.next_action(&mut rng);
            }
        });
    });
}

fn bench_all(c: &mut Criterion) {
    bench_update_only::<CfrPlusRegretMatcher>("cfr_plus", c);
    bench_update_only::<DiscountedRegretMatcher>("dcfr", c);
    bench_update_only::<DcfrPlusRegretMatcher>("dcfr_plus", c);
    bench_update_only::<LinearCfrRegretMatcher>("linear_cfr", c);
    bench_update_only::<PcfrPlusRegretMatcher>("pcfr_plus", c);
    bench_update_only::<PdcfrPlusRegretMatcher>("pdcfr_plus", c);

    bench_update_then_best_weight::<CfrPlusRegretMatcher>("cfr_plus", c);
    bench_update_then_best_weight::<DiscountedRegretMatcher>("dcfr", c);

    bench_update_and_sample::<CfrPlusRegretMatcher>("cfr_plus", c);
    bench_update_and_sample::<DiscountedRegretMatcher>("dcfr", c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
