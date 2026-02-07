#![allow(clippy::cast_precision_loss)]
//! CFR+ regret minimization implementation.
//!
//! This module provides [`CfrPlusRegretMatcher`], an implementation of the
//! CFR+ algorithm which floors negative regrets at zero.

use ndarray::prelude::*;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedAliasIndex;

use crate::errors::LittleError;
use crate::regret_minimizer::RegretMinimizer;

/// A regret matcher implementing CFR+ (regret matching plus).
///
/// CFR+ differs from vanilla CFR by flooring cumulative regrets at zero.
/// This can be viewed as DCFR_{∞, -∞, 2}: infinite positive discount
/// (keeping positive regrets), infinite negative discount (flooring at 0),
/// and linear average strategy weighting.
///
/// # Fields
///
/// * `p` - The probability distribution over experts.
/// * `sum_p` - The cumulative sum of probabilities over time.
/// * `expert_reward` - The accumulated reward for each expert.
/// * `cumulative_reward` - The total reward accumulated over time.
/// * `dist` - The weighted alias distribution for O(1) sampling.
/// * `num_updates` - The number of updates performed.
#[derive(Debug, Clone)]
pub struct CfrPlusRegretMatcher {
    p: Array1<f32>,
    sum_p: Array1<f32>,
    expert_reward: Array1<f32>,
    cumulative_reward: f32,
    dist: WeightedAliasIndex<f32>,
    num_updates: usize,
}

impl CfrPlusRegretMatcher {
    fn init_weights(num_experts: usize) -> Vec<f32> {
        vec![1.0 / num_experts as f32; num_experts]
    }

    /// Creates a new `CfrPlusRegretMatcher` with custom initial probabilities.
    ///
    /// # Arguments
    ///
    /// * `p` - Initial probability distribution over actions.
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if the probability distribution is invalid.
    pub fn new_from_p(p: Vec<f32>) -> Result<Self, LittleError> {
        let num_experts = p.len();
        let dist = WeightedAliasIndex::new(p.clone())?;
        Ok(Self {
            p: Array1::from(p),
            sum_p: Array1::zeros(num_experts),
            cumulative_reward: 0.0_f32,
            expert_reward: Array1::from(vec![0.0_f32; num_experts]),
            dist,
            num_updates: 0,
        })
    }
}

impl RegretMinimizer for CfrPlusRegretMatcher {
    fn new(num_experts: usize) -> Result<Self, LittleError> {
        let p = Self::init_weights(num_experts);
        Self::new_from_p(p)
    }

    fn next_action<R: rand::Rng>(&self, rng: &mut R) -> usize {
        self.dist.sample(rng)
    }

    fn update_regret(&mut self, reward_array: ArrayView1<f32>) -> Result<(), LittleError> {
        let num_experts = self.p.len();
        // Compute expected reward
        let r = self.p.dot(&reward_array);
        self.cumulative_reward += r;
        // Accumulate expert rewards
        self.expert_reward += &reward_array;
        // Regret = what expert would have earned - what we earned
        let regret = &self.expert_reward - self.cumulative_reward;
        // CFR+: floor negative regrets at 0
        let capped_regret: Array1<f32> = regret.iter().map(|v: &f32| f32::max(0.0, *v)).collect();
        let regret_sum = capped_regret.sum();

        if regret_sum <= 0.0 {
            // Reset if all regrets are zero or negative
            self.p = Array1::from(Self::init_weights(num_experts));
            self.cumulative_reward = 0.0;
            self.expert_reward = Array1::zeros(num_experts);
            self.num_updates = 0;
        } else {
            // Normalize to get new strategy
            self.p = capped_regret / regret_sum;
            // Accumulate for average strategy
            self.sum_p += &self.p;
            self.num_updates += 1;
        }
        self.dist = WeightedAliasIndex::new(self.p.to_vec())?;
        Ok(())
    }

    fn best_weight(&self) -> Vec<f32> {
        (self.sum_p.clone() / self.num_updates as f32).to_vec()
    }

    fn num_updates(&self) -> usize {
        self.num_updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    #[test]
    fn test_cfr_plus_new() {
        let _rg = CfrPlusRegretMatcher::new(3);
    }

    #[test]
    fn test_next_action() {
        let rg = CfrPlusRegretMatcher::new(100).unwrap();
        let mut rng = rng();
        for _i in 0..500 {
            let a = rg.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_num_updates_increments() {
        let mut rm = CfrPlusRegretMatcher::new(3).unwrap();
        assert_eq!(rm.num_updates(), 0);

        // After update, num_updates should increase
        let rewards = array![1.0_f32, 0.0_f32, -1.0_f32];
        rm.update_regret(rewards.view()).unwrap();
        assert_eq!(rm.num_updates(), 1);
    }
}
