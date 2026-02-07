#![allow(clippy::cast_precision_loss)]
//! DCFR+ (Discounted CFR+) implementation.
//!
//! This module provides [`DcfrPlusRegretMatcher`], which combines the
//! discounting approach of DCFR with the regret clipping of CFR+.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use ndarray::prelude::*;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedAliasIndex;

use crate::discount::DiscountParams;
use crate::errors::LittleError;
use crate::regret_minimizer::RegretMinimizer;

/// A regret matcher implementing DCFR+.
///
/// DCFR+ combines the discounting of old regrets from DCFR with the
/// non-negative regret constraint from CFR+. The key difference from
/// regular DCFR is that the clipping to non-negative values happens
/// AFTER adding the instantaneous regret, not before.
///
/// Update rule: `R^t = [R^{t-1} * d(t-1, α) + r^t]^+`
///
/// where `d(t, α) = t^α / (t^α + 1)`.
///
/// Recommended parameters: α = 1.5, γ = 4
#[derive(Debug, Clone)]
pub struct DcfrPlusRegretMatcher {
    alpha: f32,
    gamma: f32,
    p: Array1<f32>,
    sum_p: Array1<f32>,
    cumulative_regret: Array1<f32>,
    dist: WeightedAliasIndex<f32>,
    num_updates: usize,
}

impl DcfrPlusRegretMatcher {
    fn init_weights(num_experts: usize) -> Vec<f32> {
        vec![1.0 / num_experts as f32; num_experts]
    }

    /// Creates a new `DcfrPlusRegretMatcher` with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `num_experts` - The number of available actions.
    /// * `alpha` - Regret discount exponent (recommended: 1.5).
    /// * `gamma` - Strategy discount exponent (recommended: 4.0).
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if initialization fails.
    pub fn new_with_params(
        num_experts: usize,
        alpha: f32,
        gamma: f32,
    ) -> Result<Self, LittleError> {
        let p = Self::init_weights(num_experts);
        let dist = WeightedAliasIndex::new(p.clone())?;
        Ok(Self {
            alpha,
            gamma,
            p: Array1::from(p),
            sum_p: Array1::zeros(num_experts),
            cumulative_regret: Array1::zeros(num_experts),
            dist,
            num_updates: 0,
        })
    }

    /// Creates a new `DcfrPlusRegretMatcher` with recommended parameters.
    ///
    /// Uses α = 1.5, γ = 4.0 as recommended in the paper.
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if initialization fails.
    pub fn recommended(num_experts: usize) -> Result<Self, LittleError> {
        Self::new_with_params(num_experts, 1.5, 4.0)
    }

    /// Returns the alpha (regret discount) parameter.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the gamma (strategy discount) parameter.
    #[must_use]
    pub fn gamma(&self) -> f32 {
        self.gamma
    }
}

impl RegretMinimizer for DcfrPlusRegretMatcher {
    fn new(num_experts: usize) -> Result<Self, LittleError> {
        Self::recommended(num_experts)
    }

    fn next_action<R: rand::Rng>(&self, rng: &mut R) -> usize {
        self.dist.sample(rng)
    }

    fn update_regret(&mut self, reward_array: ArrayView1<f32>) -> Result<(), LittleError> {
        let num_experts = self.p.len();
        let t = self.num_updates + 1;

        // Compute discount factor for regrets: (t-1)^α / ((t-1)^α + 1)
        let regret_discount = if t > 1 {
            DiscountParams::discount_factor(t - 1, self.alpha)
        } else {
            0.0 // No previous regret to discount on first iteration
        };

        // Compute strategy discount: ((t-1)/t)^γ
        let strategy_discount = if t > 1 {
            ((t - 1) as f32 / t as f32).powf(self.gamma)
        } else {
            0.0
        };

        // Compute instantaneous regret
        let expected_reward = self.p.dot(&reward_array);
        let instantaneous_regret = &reward_array - expected_reward;

        // DCFR+ update: R^t = [R^{t-1} * d(t-1, α) + r^t]^+
        // The clipping happens AFTER adding instantaneous regret
        for i in 0..num_experts {
            let discounted = self.cumulative_regret[i] * regret_discount;
            self.cumulative_regret[i] = f32::max(0.0, discounted + instantaneous_regret[i]);
        }

        // Compute new strategy via regret matching
        let regret_sum = self.cumulative_regret.sum();

        if regret_sum <= 0.0 {
            self.p = Array1::from(Self::init_weights(num_experts));
        } else {
            self.p = self.cumulative_regret.clone() / regret_sum;
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
    fn test_dcfr_plus_new() {
        let rm = DcfrPlusRegretMatcher::new(3).unwrap();
        assert!((rm.alpha() - 1.5).abs() < 1e-6);
        assert!((rm.gamma() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_dcfr_plus_custom_params() {
        let rm = DcfrPlusRegretMatcher::new_with_params(3, 2.0, 3.0).unwrap();
        assert!((rm.alpha() - 2.0).abs() < 1e-6);
        assert!((rm.gamma() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_next_action() {
        let rm = DcfrPlusRegretMatcher::new(100).unwrap();
        let mut rng = rng();
        for _ in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = DcfrPlusRegretMatcher::new(3).unwrap();
        let rewards = array![1.0_f32, 0.0_f32, -1.0_f32];

        for _ in 0..10 {
            rm.update_regret(rewards.view()).unwrap();
        }

        let weights = rm.best_weight();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
