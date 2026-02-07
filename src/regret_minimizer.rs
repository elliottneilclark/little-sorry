//! Trait definition for regret minimization algorithms.
//!
//! This module provides the [`RegretMinimizer`] trait which defines
//! the common interface for CFR variants like CFR+ and DCFR.

use ndarray::prelude::*;

use crate::errors::LittleError;

/// A trait for regret minimization algorithms.
///
/// Implementors of this trait can be used interchangeably in game-solving
/// algorithms that rely on regret matching. The trait provides a common
/// interface for different CFR variants.
pub trait RegretMinimizer: Clone {
    /// Creates a new regret minimizer with the given number of actions/experts.
    ///
    /// # Arguments
    ///
    /// * `num_experts` - The number of available actions to choose from.
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if initialization fails (e.g., invalid weights).
    fn new(num_experts: usize) -> Result<Self, LittleError>
    where
        Self: Sized;

    /// Samples the next action according to the current strategy.
    ///
    /// # Arguments
    ///
    /// * `rng` - A random number generator.
    ///
    /// # Returns
    ///
    /// The index of the selected action.
    fn next_action<R: rand::Rng>(&self, rng: &mut R) -> usize;

    /// Updates regrets based on the reward array.
    ///
    /// This method should:
    /// 1. Compute instantaneous regret for each action
    /// 2. Update cumulative regrets (with algorithm-specific discounting)
    /// 3. Derive new strategy from regret matching
    /// 4. Update average strategy weights
    ///
    /// # Arguments
    ///
    /// * `reward_array` - A view of rewards for each action.
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if the update fails.
    fn update_regret(&mut self, reward_array: ArrayView1<f32>) -> Result<(), LittleError>;

    /// Returns the average strategy weights.
    ///
    /// This represents the Nash equilibrium approximation computed
    /// as the time-weighted average of strategies played.
    #[must_use]
    fn best_weight(&self) -> Vec<f32>;

    /// Returns the number of updates performed.
    #[must_use]
    fn num_updates(&self) -> usize;
}
