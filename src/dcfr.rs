#![allow(clippy::cast_precision_loss)]
//! Discounted CFR (DCFR) implementation.
//!
//! This module provides [`DiscountedRegretMatcher`], a configurable
//! implementation of discounted counterfactual regret minimization.

use ndarray::prelude::*;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedAliasIndex;

use crate::discount::DiscountParams;
use crate::errors::LittleError;
use crate::regret_minimizer::RegretMinimizer;

/// A regret matcher implementing discounted CFR (DCFR).
///
/// DCFR applies time-based discounting to cumulative regrets and
/// average strategy weights. This can accelerate convergence compared
/// to vanilla CFR or CFR+.
///
/// The discount factor at iteration `t` for exponent `exp` is:
/// `t^exp / (t^exp + 1)`
#[derive(Debug, Clone)]
pub struct DiscountedRegretMatcher {
    params: DiscountParams,
    p: Array1<f32>,
    sum_p: Array1<f32>,
    cumulative_regret: Array1<f32>,
    dist: WeightedAliasIndex<f32>,
    num_updates: usize,
}

impl DiscountedRegretMatcher {
    fn init_weights(num_experts: usize) -> Vec<f32> {
        vec![1.0 / num_experts as f32; num_experts]
    }

    /// Creates a new `DiscountedRegretMatcher` with custom discount parameters.
    ///
    /// # Arguments
    ///
    /// * `num_experts` - The number of available actions.
    /// * `params` - The discount parameters (alpha, beta, gamma).
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if initialization fails.
    pub fn new_with_params(
        num_experts: usize,
        params: DiscountParams,
    ) -> Result<Self, LittleError> {
        let p = Self::init_weights(num_experts);
        let dist = WeightedAliasIndex::new(p.clone())?;
        Ok(Self {
            params,
            p: Array1::from(p),
            sum_p: Array1::zeros(num_experts),
            cumulative_regret: Array1::zeros(num_experts),
            dist,
            num_updates: 0,
        })
    }

    /// Creates a new `DiscountedRegretMatcher` using Linear CFR (LCFR).
    ///
    /// LCFR uses DCFR_{1,1,1}: all discounts are linear.
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if initialization fails.
    pub fn lcfr(num_experts: usize) -> Result<Self, LittleError> {
        Self::new_with_params(num_experts, DiscountParams::LCFR)
    }

    /// Creates a new `DiscountedRegretMatcher` using recommended parameters.
    ///
    /// Uses DCFR_{1.5,0,2} which has been shown to provide fast convergence.
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if initialization fails.
    pub fn recommended(num_experts: usize) -> Result<Self, LittleError> {
        Self::new_with_params(num_experts, DiscountParams::RECOMMENDED)
    }

    /// Creates a new `DiscountedRegretMatcher` with pruning-safe parameters.
    ///
    /// Uses DCFR_{1.5,0.5,2} which is safer for regret-based pruning.
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if initialization fails.
    pub fn pruning_safe(num_experts: usize) -> Result<Self, LittleError> {
        Self::new_with_params(num_experts, DiscountParams::PRUNING_SAFE)
    }

    /// Returns the discount parameters used by this matcher.
    #[must_use]
    pub fn params(&self) -> DiscountParams {
        self.params
    }
}

impl RegretMinimizer for DiscountedRegretMatcher {
    fn new(num_experts: usize) -> Result<Self, LittleError> {
        Self::recommended(num_experts)
    }

    fn next_action<R: rand::Rng>(&self, rng: &mut R) -> usize {
        self.dist.sample(rng)
    }

    fn update_regret(&mut self, reward_array: ArrayView1<f32>) -> Result<(), LittleError> {
        let num_experts = self.p.len();
        let t = self.num_updates + 1;

        // Compute discount factors
        let positive_discount = DiscountParams::discount_factor(t, self.params.alpha);
        let negative_discount = DiscountParams::discount_factor(t, self.params.beta);
        let strategy_discount = (t as f32 / (t as f32 + 1.0)).powf(self.params.gamma);

        // Compute expected reward and instantaneous regret
        let expected_reward = self.p.dot(&reward_array);
        let instantaneous_regret = &reward_array - expected_reward;

        // Apply discounting to cumulative regrets based on sign, then add new regret
        for i in 0..num_experts {
            let discount = if self.cumulative_regret[i] > 0.0 {
                positive_discount
            } else {
                negative_discount
            };
            self.cumulative_regret[i] =
                self.cumulative_regret[i] * discount + instantaneous_regret[i];
        }

        // Compute new strategy via regret matching
        let positive_regret: Array1<f32> = self
            .cumulative_regret
            .iter()
            .map(|&v| f32::max(0.0, v))
            .collect();
        let regret_sum = positive_regret.sum();

        if regret_sum <= 0.0 {
            // All regrets non-positive: use uniform strategy
            self.p = Array1::from(Self::init_weights(num_experts));
        } else {
            self.p = positive_regret / regret_sum;
        }

        // Update average strategy with discount
        self.sum_p = &self.sum_p * strategy_discount + &self.p;
        self.num_updates += 1;

        self.dist = WeightedAliasIndex::new(self.p.to_vec())?;
        Ok(())
    }

    fn best_weight(&self) -> Vec<f32> {
        let sum = self.sum_p.sum();
        if sum <= 0.0 {
            Self::init_weights(self.p.len())
        } else {
            (self.sum_p.clone() / sum).to_vec()
        }
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
    fn test_dcfr_new() {
        let _rm = DiscountedRegretMatcher::new(3).unwrap();
    }

    #[test]
    fn test_dcfr_lcfr() {
        let rm = DiscountedRegretMatcher::lcfr(3).unwrap();
        assert_eq!(rm.params(), DiscountParams::LCFR);
    }

    #[test]
    fn test_dcfr_recommended() {
        let rm = DiscountedRegretMatcher::recommended(3).unwrap();
        assert_eq!(rm.params(), DiscountParams::RECOMMENDED);
    }

    #[test]
    fn test_dcfr_pruning_safe() {
        let rm = DiscountedRegretMatcher::pruning_safe(3).unwrap();
        assert_eq!(rm.params(), DiscountParams::PRUNING_SAFE);
    }

    #[test]
    fn test_next_action() {
        let rm = DiscountedRegretMatcher::new(100).unwrap();
        let mut rng = rng();
        for _i in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_num_updates_increments() {
        let mut rm = DiscountedRegretMatcher::new(3).unwrap();
        assert_eq!(rm.num_updates(), 0);

        let rewards = array![1.0_f32, 0.0_f32, -1.0_f32];
        rm.update_regret(rewards.view()).unwrap();
        assert_eq!(rm.num_updates(), 1);
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = DiscountedRegretMatcher::new(3).unwrap();
        let rewards = array![1.0_f32, 0.0_f32, -1.0_f32];

        for _ in 0..10 {
            rm.update_regret(rewards.view()).unwrap();
        }

        let weights = rm.best_weight();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
