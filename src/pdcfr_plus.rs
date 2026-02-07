#![allow(clippy::cast_precision_loss)]
//! PDCFR+ (Predictive Discounted CFR+) implementation.
//!
//! This module provides [`PdcfrPlusRegretMatcher`], which combines the
//! discounting of DCFR+ with the predictive approach of PCFR+.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use ndarray::prelude::*;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedAliasIndex;

use crate::discount::DiscountParams;
use crate::errors::LittleError;
use crate::regret_minimizer::RegretMinimizer;

/// A regret matcher implementing PDCFR+ (Predictive Discounted CFR+).
///
/// PDCFR+ combines the best of both DCFR+ and PCFR+:
/// - Regret discounting from DCFR+
/// - Predictive strategy selection from PCFR+
///
/// Update rules:
/// - Regret: `R^t = [R^{t-1} * d(t-1, α) + r^t]^+`
/// - Predicted regret: `R̃^{t+1} = [R^t * d(t, α) + v^{t+1}]^+`
/// - Strategy: `x^{t+1} = R̃^{t+1} / ||R̃^{t+1}||_1`
/// - Cumulative: `X^t = X^{t-1} * ((t-1)/t)^γ + x^t`
///
/// where `d(t, α) = t^α / (t^α + 1)`.
///
/// Recommended parameters: α = 2.3, γ = 5
#[derive(Debug, Clone)]
pub struct PdcfrPlusRegretMatcher {
    alpha: f32,
    gamma: f32,
    p: Array1<f32>,
    sum_p: Array1<f32>,
    cumulative_regret: Array1<f32>,
    last_instantaneous_regret: Array1<f32>,
    dist: WeightedAliasIndex<f32>,
    num_updates: usize,
}

impl PdcfrPlusRegretMatcher {
    fn init_weights(num_experts: usize) -> Vec<f32> {
        vec![1.0 / num_experts as f32; num_experts]
    }

    /// Creates a new `PdcfrPlusRegretMatcher` with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `num_experts` - The number of available actions.
    /// * `alpha` - Regret discount exponent (recommended: 2.3).
    /// * `gamma` - Strategy discount exponent (recommended: 5.0).
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
            last_instantaneous_regret: Array1::zeros(num_experts),
            dist,
            num_updates: 0,
        })
    }

    /// Creates a new `PdcfrPlusRegretMatcher` with recommended parameters.
    ///
    /// Uses α = 2.3, γ = 5.0 as recommended in the paper.
    ///
    /// # Errors
    ///
    /// Returns [`LittleError`] if initialization fails.
    pub fn recommended(num_experts: usize) -> Result<Self, LittleError> {
        Self::new_with_params(num_experts, 2.3, 5.0)
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

impl RegretMinimizer for PdcfrPlusRegretMatcher {
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
        let prev_regret_discount = if t > 1 {
            DiscountParams::discount_factor(t - 1, self.alpha)
        } else {
            0.0
        };

        let curr_regret_discount = DiscountParams::discount_factor(t, self.alpha);

        let strategy_discount = if t > 1 {
            ((t - 1) as f32 / t as f32).powf(self.gamma)
        } else {
            0.0
        };

        // Compute instantaneous regret
        let expected_reward = self.p.dot(&reward_array);
        let instantaneous_regret = &reward_array - expected_reward;

        // DCFR+ regret update: R^t = [R^{t-1} * d(t-1, α) + r^t]^+
        for i in 0..num_experts {
            let discounted = self.cumulative_regret[i] * prev_regret_discount;
            self.cumulative_regret[i] = f32::max(0.0, discounted + instantaneous_regret[i]);
        }

        // Store for prediction
        self.last_instantaneous_regret = instantaneous_regret.to_owned();

        // Compute predicted regrets: R̃^{t+1} = [R^t * d(t, α) + v^{t+1}]^+
        // where v^{t+1} = r^t (current regret as prediction)
        let predicted_regret: Array1<f32> = self
            .cumulative_regret
            .iter()
            .zip(self.last_instantaneous_regret.iter())
            .map(|(&r, &v)| f32::max(0.0, r * curr_regret_discount + v))
            .collect();

        // Compute new strategy from predicted regrets
        let regret_sum = predicted_regret.sum();

        if regret_sum <= 0.0 {
            self.p = Array1::from(Self::init_weights(num_experts));
        } else {
            self.p = predicted_regret / regret_sum;
        }

        // DCFR-style discounted cumulative strategy
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
    fn test_pdcfr_plus_new() {
        let rm = PdcfrPlusRegretMatcher::new(3).unwrap();
        assert!((rm.alpha() - 2.3).abs() < 1e-6);
        assert!((rm.gamma() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_pdcfr_plus_custom_params() {
        let rm = PdcfrPlusRegretMatcher::new_with_params(3, 2.0, 4.0).unwrap();
        assert!((rm.alpha() - 2.0).abs() < 1e-6);
        assert!((rm.gamma() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_next_action() {
        let rm = PdcfrPlusRegretMatcher::new(100).unwrap();
        let mut rng = rng();
        for _ in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = PdcfrPlusRegretMatcher::new(3).unwrap();
        let rewards = array![1.0_f32, 0.0_f32, -1.0_f32];

        for _ in 0..10 {
            rm.update_regret(rewards.view()).unwrap();
        }

        let weights = rm.best_weight();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_num_updates_increments() {
        let mut rm = PdcfrPlusRegretMatcher::new(3).unwrap();
        assert_eq!(rm.num_updates(), 0);

        let rewards = array![1.0_f32, 0.0_f32, -1.0_f32];
        rm.update_regret(rewards.view()).unwrap();
        assert_eq!(rm.num_updates(), 1);
    }
}
