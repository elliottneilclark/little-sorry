use criterion::{Criterion, criterion_group, criterion_main};

use std::time::Instant;

use little_sorry::rps::RPSRunner;

pub fn bench(c: &mut Criterion) {
    c.bench_function("iter", move |b| {
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

criterion_group!(benches, bench);
criterion_main!(benches);
