#![deny(clippy::all)]
#![deny(clippy::pedantic)]

use little_sorry::rps::RPSRunner;

static NUM_ITERS: usize = 100_000_000;

fn main() {
    let mut runner = RPSRunner::new().unwrap();
    dbg!(&runner.matcher_one);
    let mut rng = rand::rng();
    for i in 0..NUM_ITERS {
        runner.run_one(&mut rng);

        if i % 50 == 0 || i == NUM_ITERS - 1 {
            runner.update_regret().unwrap();
        }
    }
    dbg!(&runner);
    dbg!(runner.best_weight());
}
