use crate::cfr_plus::CfrPlusRegretMatcher;
use crate::regret_minimizer::RegretMinimizer;
use std::cmp;
use std::mem;

/// Represents the actions in Rock-Paper-Scissors game.
///
/// This uses unsafe code to do the conversion so it shows as dead code.
#[allow(dead_code)]
#[repr(usize)]
#[derive(Debug, Clone, Copy)]
enum RPSAction {
    /// Rock action.
    Rock = 0,
    /// Paper action.
    Paper = 1,
    /// Scissors action.
    Scissors = 2,
}

const ROCK_REWARD: [f32; 3] = [0.0, 1.0, -1.0];
const PAPER_REWARD: [f32; 3] = [-1.0, 0.0, 1.0];
const SCISSOR_REWARD: [f32; 3] = [1.0, -1.0, 0.0];

impl RPSAction {
    /// Converts the action to its corresponding reward array.
    pub fn to_reward(self) -> &'static [f32] {
        match self {
            Self::Rock => &ROCK_REWARD,
            Self::Paper => &PAPER_REWARD,
            Self::Scissors => &SCISSOR_REWARD,
        }
    }
}

impl From<usize> for RPSAction {
    fn from(i: usize) -> Self {
        unsafe {
            mem::transmute(cmp::max(
                cmp::min(i, Self::Scissors as usize),
                Self::Rock as usize,
            ))
        }
    }
}

/// Runner for the Rock-Paper-Scissors game using regret matching.
///
/// This struct is generic over the regret minimization algorithm,
/// allowing use of CFR+, DCFR, or other implementations.
#[derive(Debug, Clone)]
pub struct RPSRunnerGeneric<M: RegretMinimizer> {
    /// Regret matcher for the first player.
    pub matcher_one: M,
    /// Regret matcher for the second player.
    pub matcher_two: M,
    pending_reward_one: Vec<f32>,
    pending_reward_two: Vec<f32>,
}

impl<M: RegretMinimizer> RPSRunnerGeneric<M> {
    /// Creates a new `RPSRunnerGeneric`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            matcher_one: M::new(3),
            matcher_two: M::new(3),
            pending_reward_one: vec![0.0; 3],
            pending_reward_two: vec![0.0; 3],
        }
    }

    /// Creates a new `RPSRunnerGeneric` with pre-configured matchers.
    #[must_use]
    pub fn new_with_matchers(matcher_one: M, matcher_two: M) -> Self {
        Self {
            matcher_one,
            matcher_two,
            pending_reward_one: vec![0.0; 3],
            pending_reward_two: vec![0.0; 3],
        }
    }

    /// Runs one iteration of the Rock-Paper-Scissors game.
    pub fn run_one<R: rand::Rng>(&mut self, rng: &mut R) {
        let a1 = RPSAction::from(self.matcher_one.next_action(rng));
        let a2 = RPSAction::from(self.matcher_two.next_action(rng));

        let r2 = a2.to_reward();
        let r1 = a1.to_reward();
        for (pr, &r) in self.pending_reward_one.iter_mut().zip(r2) {
            *pr += r;
        }
        for (pr, &r) in self.pending_reward_two.iter_mut().zip(r1) {
            *pr += r;
        }
    }

    /// Updates the regret values for both players.
    pub fn update_regret(&mut self) {
        self.matcher_one.update_regret(&self.pending_reward_one);
        self.matcher_two.update_regret(&self.pending_reward_two);

        self.pending_reward_one.fill(0.0);
        self.pending_reward_two.fill(0.0);
    }

    /// Returns the best weight for the first player.
    #[must_use]
    pub fn best_weight(&self) -> Vec<f32> {
        self.matcher_one.best_weight()
    }

    /// Returns the best weight for the second player.
    pub fn opponent_best_weight(&self) -> Vec<f32> {
        self.matcher_two.best_weight()
    }
}

/// Type alias for backwards compatibility with CFR+.
pub type RPSRunner = RPSRunnerGeneric<CfrPlusRegretMatcher>;

impl Default for RPSRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dcfr::DiscountedRegretMatcher;

    #[test]
    fn test_rps_cfr_plus() {
        let mut runner = RPSRunner::new();
        let mut rng = rand::rng();
        for _ in 0..100 {
            runner.run_one(&mut rng);
            runner.update_regret();
        }
        dbg!(runner.best_weight());
    }

    #[test]
    fn test_rps_dcfr() {
        let mut runner = RPSRunnerGeneric::<DiscountedRegretMatcher>::new();
        let mut rng = rand::rng();
        for _ in 0..100 {
            runner.run_one(&mut rng);
            runner.update_regret();
        }
        dbg!(runner.best_weight());
    }
}
