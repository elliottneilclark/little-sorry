#![allow(clippy::cast_precision_loss)]
//! PDCFR+ (Predictive Discounted CFR+) implementation.
//!
//! This module provides [`PdcfrPlusRegretMatcher`], which combines the
//! discounting of DCFR+ with the predictive approach of PCFR+.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use crate::discount::DiscountParams;
use crate::regret_minimizer::{self, RegretMinimizer};

/// A regret matcher implementing PDCFR+ (Predictive Discounted CFR+).
///
/// PDCFR+ combines the best of both DCFR+ and PCFR+:
/// - Regret discounting from DCFR+
/// - Predictive strategy selection from PCFR+
///
/// Update rules:
/// - Regret: `R^t = [R^{t-1} * d(t-1, alpha) + r^t]^+`
/// - Predicted regret: `R_tilde^{t+1} = [R^t * d(t, alpha) + v^{t+1}]^+`
/// - Strategy: `x^{t+1} = R_tilde^{t+1} / ||R_tilde^{t+1}||_1`
/// - Cumulative: `X^t = X^{t-1} * ((t-1)/t)^gamma + x^t`
///
/// where `d(t, alpha) = t^alpha / (t^alpha + 1)`.
///
/// Recommended parameters: alpha = 2.3, gamma = 5
#[derive(Debug, Clone)]
pub struct PdcfrPlusRegretMatcher {
    alpha: f32,
    gamma: f32,
    p: Vec<f32>,
    sum_p: Vec<f32>,
    cumulative_regret: Vec<f32>,
    last_instantaneous_regret: Vec<f32>,
    num_updates: usize,
}

impl PdcfrPlusRegretMatcher {
    /// Creates a new `PdcfrPlusRegretMatcher` with custom parameters.
    ///
    /// # Panics
    ///
    /// Panics if `num_experts` is 0.
    #[must_use]
    pub fn new_with_params(num_experts: usize, alpha: f32, gamma: f32) -> Self {
        let p = regret_minimizer::uniform_weights(num_experts);
        Self {
            alpha,
            gamma,
            p,
            sum_p: vec![0.0; num_experts],
            cumulative_regret: vec![0.0; num_experts],
            last_instantaneous_regret: vec![0.0; num_experts],
            num_updates: 0,
        }
    }

    /// Creates a new `PdcfrPlusRegretMatcher` with recommended parameters.
    ///
    /// Uses alpha = 2.3, gamma = 5.0 as recommended in the paper.
    #[must_use]
    pub fn recommended(num_experts: usize) -> Self {
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
    fn new(num_experts: usize) -> Self {
        Self::recommended(num_experts)
    }

    fn update_regret(&mut self, rewards: &[f32]) {
        let t = self.num_updates + 1;

        let prev_discount = if t > 1 {
            DiscountParams::discount_factor(t - 1, self.alpha)
        } else {
            0.0
        };
        let curr_discount = DiscountParams::discount_factor(t, self.alpha);
        let strategy_discount = if t > 1 {
            ((t - 1) as f32 / t as f32).powf(self.gamma)
        } else {
            0.0
        };

        let expected = regret_minimizer::dot(&self.p, rewards);

        // DCFR+ regret update + store instantaneous for prediction
        for ((cr, lr), &rw) in self
            .cumulative_regret
            .iter_mut()
            .zip(self.last_instantaneous_regret.iter_mut())
            .zip(rewards)
        {
            let inst = rw - expected;
            *cr = (*cr * prev_discount + inst).max(0.0);
            *lr = inst;
        }

        // Predicted regrets into p: R_tilde = [R^t * d(t, alpha) + v]^+
        for ((pi, &cr), &lr) in self
            .p
            .iter_mut()
            .zip(self.cumulative_regret.iter())
            .zip(self.last_instantaneous_regret.iter())
        {
            *pi = (cr * curr_discount + lr).max(0.0);
        }
        regret_minimizer::normalize_inplace(&mut self.p);

        // Discounted cumulative strategy
        for (sp, &pi) in self.sum_p.iter_mut().zip(self.p.iter()) {
            *sp = *sp * strategy_discount + pi;
        }
        self.num_updates += 1;
    }

    fn num_updates(&self) -> usize {
        self.num_updates
    }

    fn current_strategy(&self) -> &[f32] {
        &self.p
    }

    fn cumulative_strategy(&self) -> &[f32] {
        &self.sum_p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    #[test]
    fn test_pdcfr_plus_new() {
        let rm = PdcfrPlusRegretMatcher::new(3);
        assert!((rm.alpha() - 2.3).abs() < 1e-6);
        assert!((rm.gamma() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_pdcfr_plus_custom_params() {
        let rm = PdcfrPlusRegretMatcher::new_with_params(3, 2.0, 4.0);
        assert!((rm.alpha() - 2.0).abs() < 1e-6);
        assert!((rm.gamma() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_next_action() {
        let rm = PdcfrPlusRegretMatcher::new(100);
        let mut rng = rng();
        for _ in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = PdcfrPlusRegretMatcher::new(3);

        for _ in 0..10 {
            rm.update_regret(&[1.0, 0.0, -1.0]);
        }

        let weights = rm.best_weight();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_num_updates_increments() {
        let mut rm = PdcfrPlusRegretMatcher::new(3);
        assert_eq!(rm.num_updates(), 0);

        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert_eq!(rm.num_updates(), 1);
    }
}
