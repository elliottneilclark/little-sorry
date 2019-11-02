#![deny(clippy::all)]
#![deny(clippy::pedantic)]

use little_sorry::rps::RPSRunner;

fn main() {
    let mut runner = RPSRunner::new();
    dbg!(&runner.matcher_one);
    for _i in 0..1_000_000 {
        runner.run_one();
    }
    dbg!(&runner.matcher_one);
    dbg!(runner.best_weight());
}
