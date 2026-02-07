use crate::cfr_plus::CfrPlusRegretMatcher;
use crate::errors::LittleError;
use crate::regret_minimizer::RegretMinimizer;
use ndarray::prelude::*;
use std::cmp;
use std::mem;
use std::sync::LazyLock;

use std::vec::Vec;

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

static ROCK_REWARD: LazyLock<Array1<f32>> = LazyLock::new(|| array![0.0_f32, 1.0_f32, -1.0_f32]);
static PAPER_REWARD: LazyLock<Array1<f32>> = LazyLock::new(|| array![-1.0_f32, 0.0_f32, 1.0_f32]);
static SCISSOR_REWARD: LazyLock<Array1<f32>> = LazyLock::new(|| array![1.0_f32, -1.0_f32, 0.0_f32]);

impl RPSAction {
    /// Converts the action to its corresponding reward array view.
    ///
    /// # Returns
    ///
    /// An array view of rewards for the action.
    pub fn to_reward(self) -> ArrayView1<'static, f32> {
        match self {
            Self::Rock => ROCK_REWARD.view(),
            Self::Paper => PAPER_REWARD.view(),
            Self::Scissors => SCISSOR_REWARD.view(),
        }
    }
}

impl From<usize> for RPSAction {
    /// Converts a usize to an RPSAction.
    ///
    /// # Arguments
    ///
    /// * `i` - The usize value to convert.
    ///
    /// # Safety
    ///
    /// This function uses `unsafe` to transmute the usize value to an RPSAction.
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
    pending_reward_one: Array1<f32>,
    pending_reward_two: Array1<f32>,
}

impl<M: RegretMinimizer> RPSRunnerGeneric<M> {
    /// Creates a new `RPSRunnerGeneric`.
    ///
    /// # Returns
    ///
    /// A result containing the new runner or a `LittleError`.
    pub fn new() -> Result<Self, LittleError> {
        Ok(Self {
            matcher_one: M::new(3)?,
            matcher_two: M::new(3)?,
            pending_reward_one: Array1::zeros(3),
            pending_reward_two: Array1::zeros(3),
        })
    }

    /// Creates a new `RPSRunnerGeneric` with pre-configured matchers.
    ///
    /// This allows using specific configurations of the regret minimizer.
    #[must_use]
    pub fn new_with_matchers(matcher_one: M, matcher_two: M) -> Self {
        Self {
            matcher_one,
            matcher_two,
            pending_reward_one: Array1::zeros(3),
            pending_reward_two: Array1::zeros(3),
        }
    }

    /// Runs one iteration of the Rock-Paper-Scissors game.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a random number generator.
    pub fn run_one<R: rand::Rng>(&mut self, rng: &mut R) {
        let a1 = RPSAction::from(self.matcher_one.next_action(rng));
        let a2 = RPSAction::from(self.matcher_two.next_action(rng));

        self.pending_reward_one += &a2.to_reward();
        self.pending_reward_two += &a1.to_reward();
    }

    /// Updates the regret values for both players.
    ///
    /// # Returns
    ///
    /// A result indicating success or a `LittleError`.
    pub fn update_regret(&mut self) -> Result<(), LittleError> {
        self.matcher_one
            .update_regret(self.pending_reward_one.view())?;
        self.matcher_two
            .update_regret(self.pending_reward_two.view())?;

        self.pending_reward_one.fill(0.0);
        self.pending_reward_two.fill(0.0);
        Ok(())
    }

    /// Returns the best weight for the first player.
    ///
    /// # Returns
    ///
    /// A vector of floats representing the best weight.
    #[must_use]
    pub fn best_weight(&self) -> Vec<f32> {
        self.matcher_one.best_weight()
    }

    /// Returns the best weight for the second player.
    ///
    /// # Returns
    ///
    /// A vector of floats representing the best weight for the opponent player.
    pub fn opponent_best_weight(&self) -> Vec<f32> {
        self.matcher_two.best_weight()
    }
}

/// Type alias for backwards compatibility with CFR+.
pub type RPSRunner = RPSRunnerGeneric<CfrPlusRegretMatcher>;

impl Default for RPSRunner {
    /// Creates a new `RPSRunner` with default values.
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dcfr::DiscountedRegretMatcher;

    /// Tests the Rock-Paper-Scissors runner with CFR+.
    #[test]
    fn test_rps_cfr_plus() {
        let mut runner = RPSRunner::new().unwrap();
        let mut rng = rand::rng();
        for _ in 0..100 {
            runner.run_one(&mut rng);
            runner.update_regret().unwrap();
        }
        dbg!(runner.best_weight());
    }

    /// Tests the Rock-Paper-Scissors runner with DCFR.
    #[test]
    fn test_rps_dcfr() {
        let mut runner = RPSRunnerGeneric::<DiscountedRegretMatcher>::new().unwrap();
        let mut rng = rand::rng();
        for _ in 0..100 {
            runner.run_one(&mut rng);
            runner.update_regret().unwrap();
        }
        dbg!(runner.best_weight());
    }
}
