#![allow(clippy::cast_precision_loss)]
//! DCFR+ (Discounted CFR+) implementation.
//!
//! This module provides [`DcfrPlusRegretMatcher`], which combines the
//! discounting approach of DCFR with the regret clipping of CFR+.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use crate::discount::DiscountParams;
use crate::regret_minimizer::{self, RegretMinimizer};

/// A regret matcher implementing DCFR+.
///
/// DCFR+ combines the discounting of old regrets from DCFR with the
/// non-negative regret constraint from CFR+. The key difference from
/// regular DCFR is that the clipping to non-negative values happens
/// AFTER adding the instantaneous regret, not before.
///
/// Update rule: `R^t = [R^{t-1} * d(t-1, alpha) + r^t]^+`
///
/// where `d(t, alpha) = t^alpha / (t^alpha + 1)`.
///
/// Recommended parameters: alpha = 1.5, gamma = 4
#[derive(Debug, Clone)]
pub struct DcfrPlusRegretMatcher {
    alpha: f32,
    gamma: f32,
    p: Vec<f32>,
    sum_p: Vec<f32>,
    cumulative_regret: Vec<f32>,
    num_updates: usize,
}

impl DcfrPlusRegretMatcher {
    /// Creates a new `DcfrPlusRegretMatcher` with custom parameters.
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
            num_updates: 0,
        }
    }

    /// Creates a new `DcfrPlusRegretMatcher` with recommended parameters.
    ///
    /// Uses alpha = 1.5, gamma = 4.0 as recommended in the paper.
    #[must_use]
    pub fn recommended(num_experts: usize) -> Self {
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
    fn new(num_experts: usize) -> Self {
        Self::recommended(num_experts)
    }

    fn update_regret(&mut self, rewards: &[f32]) {
        let t = self.num_updates + 1;

        let regret_discount = if t > 1 {
            DiscountParams::discount_factor(t - 1, self.alpha)
        } else {
            0.0
        };
        let strategy_discount = if t > 1 {
            ((t - 1) as f32 / t as f32).powf(self.gamma)
        } else {
            0.0
        };

        let expected = regret_minimizer::dot(&self.p, rewards);

        // DCFR+ update: R^t = [R^{t-1} * d(t-1, alpha) + r^t]^+
        for (cr, &rw) in self.cumulative_regret.iter_mut().zip(rewards) {
            *cr = (*cr * regret_discount + rw - expected).max(0.0);
        }

        regret_minimizer::regret_match(&self.cumulative_regret, &mut self.p);

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
    fn test_dcfr_plus_new() {
        let rm = DcfrPlusRegretMatcher::new(3);
        assert!((rm.alpha() - 1.5).abs() < 1e-6);
        assert!((rm.gamma() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_dcfr_plus_custom_params() {
        let rm = DcfrPlusRegretMatcher::new_with_params(3, 2.0, 3.0);
        assert!((rm.alpha() - 2.0).abs() < 1e-6);
        assert!((rm.gamma() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_next_action() {
        let rm = DcfrPlusRegretMatcher::new(100);
        let mut rng = rng();
        for _ in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = DcfrPlusRegretMatcher::new(3);

        for _ in 0..10 {
            rm.update_regret(&[1.0, 0.0, -1.0]);
        }

        let weights = rm.best_weight();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
