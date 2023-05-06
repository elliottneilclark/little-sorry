use criterion::{criterion_group, criterion_main, Criterion};

use std::time::Instant;

use little_sorry::rps::RPSRunner;

pub fn bench(c: &mut Criterion) {
    c.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let mut runner = RPSRunner::new().unwrap();
            let start = Instant::now();
            for _i in 0..iters {
                runner.run_one();
                runner.update_regret().unwrap();
            }
            start.elapsed()
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
