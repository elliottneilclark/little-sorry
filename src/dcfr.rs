#![allow(clippy::cast_precision_loss)]
//! Discounted CFR (DCFR) implementation.
//!
//! This module provides [`DiscountedRegretMatcher`], a configurable
//! implementation of discounted counterfactual regret minimization.

use crate::discount::DiscountParams;
use crate::regret_minimizer::{self, RegretMinimizer};
use crate::{probability, vector_ops};

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
    p: Vec<f32>,
    sum_p: Vec<f32>,
    cumulative_regret: Vec<f32>,
    regret_weight: f32,
    num_updates: usize,
}

impl DiscountedRegretMatcher {
    /// Creates a new `DiscountedRegretMatcher` with custom discount parameters.
    ///
    /// # Panics
    ///
    /// Panics if `num_experts` is 0.
    #[must_use]
    pub fn new_with_params(num_experts: usize, params: DiscountParams) -> Self {
        let p = probability::uniform_weights(num_experts);
        Self {
            params,
            p,
            sum_p: vec![0.0; num_experts],
            cumulative_regret: vec![0.0; num_experts],
            regret_weight: 0.0,
            num_updates: 0,
        }
    }

    /// Creates a new `DiscountedRegretMatcher` using Linear CFR (LCFR).
    ///
    /// LCFR uses DCFR_{1,1,1}: all discounts are linear.
    #[must_use]
    pub fn lcfr(num_experts: usize) -> Self {
        Self::new_with_params(num_experts, DiscountParams::LCFR)
    }

    /// Creates a new `DiscountedRegretMatcher` using recommended parameters.
    ///
    /// Uses DCFR_{1.5,0,2} which has been shown to provide fast convergence.
    #[must_use]
    pub fn recommended(num_experts: usize) -> Self {
        Self::new_with_params(num_experts, DiscountParams::RECOMMENDED)
    }

    /// Creates a new `DiscountedRegretMatcher` with pruning-safe parameters.
    ///
    /// Uses DCFR_{1.5,0.5,2} which is safer for regret-based pruning.
    #[must_use]
    pub fn pruning_safe(num_experts: usize) -> Self {
        Self::new_with_params(num_experts, DiscountParams::PRUNING_SAFE)
    }

    /// Returns the discount parameters used by this matcher.
    #[must_use]
    pub fn params(&self) -> DiscountParams {
        self.params
    }
}

impl RegretMinimizer for DiscountedRegretMatcher {
    fn new(num_experts: usize) -> Self {
        Self::recommended(num_experts)
    }

    fn update_regret(&mut self, rewards: &[f32]) {
        let t = self.num_updates + 1;

        let positive_discount = DiscountParams::discount_factor(t, self.params.alpha);
        let negative_discount = DiscountParams::discount_factor(t, self.params.beta);
        let strategy_discount = (t as f32 / (t as f32 + 1.0)).powf(self.params.gamma);

        let expected = vector_ops::dot(&self.p, rewards);

        // Apply sign-based discounting and add instantaneous regret
        for (cr, &rw) in self.cumulative_regret.iter_mut().zip(rewards) {
            let discount = if *cr > 0.0 {
                positive_discount
            } else {
                negative_discount
            };
            *cr = *cr * discount + (rw - expected);
        }

        regret_minimizer::regret_match(&self.cumulative_regret, &mut self.p);

        // Track the discounted regret-weight sum `W^t = W^{t-1} * d + 1`,
        // mirroring the positive-regret discount applied to the accumulator.
        self.regret_weight = self.regret_weight * positive_discount + 1.0;

        // Discount and update cumulative strategy
        vector_ops::discounted_accumulate(&mut self.sum_p, strategy_discount, &self.p);
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

    fn cumulative_regret(&self) -> &[f32] {
        &self.cumulative_regret
    }

    /// DCFR discounts older regret, so the weight total is the tracked
    /// discounted sum `Σ_t ∏_{s>t} d(s, alpha)` rather than `T`.
    fn regret_weight_total(&self) -> f32 {
        self.regret_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    #[test]
    fn test_dcfr_new() {
        let _rm = DiscountedRegretMatcher::new(3);
    }

    #[test]
    fn test_dcfr_lcfr() {
        let rm = DiscountedRegretMatcher::lcfr(3);
        assert_eq!(rm.params(), DiscountParams::LCFR);
    }

    #[test]
    fn test_dcfr_recommended() {
        let rm = DiscountedRegretMatcher::recommended(3);
        assert_eq!(rm.params(), DiscountParams::RECOMMENDED);
    }

    #[test]
    fn test_dcfr_pruning_safe() {
        let rm = DiscountedRegretMatcher::pruning_safe(3);
        assert_eq!(rm.params(), DiscountParams::PRUNING_SAFE);
    }

    #[test]
    fn test_next_action() {
        let rm = DiscountedRegretMatcher::new(100);
        let mut rng = rng();
        for _i in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_num_updates_increments() {
        let mut rm = DiscountedRegretMatcher::new(3);
        assert_eq!(rm.num_updates(), 0);

        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert_eq!(rm.num_updates(), 1);
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = DiscountedRegretMatcher::new(3);

        for _ in 0..10 {
            rm.update_regret(&[1.0, 0.0, -1.0]);
        }

        let weights = rm.best_weight();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cumulative_regret_len() {
        let mut rm = DiscountedRegretMatcher::new(4);
        assert_eq!(rm.cumulative_regret().len(), 4);

        rm.update_regret(&[1.0, 0.0, -1.0, 0.5]);
        assert_eq!(rm.cumulative_regret().len(), 4);
    }

    #[test]
    fn test_average_regret_zero_before_updates() {
        let rm = DiscountedRegretMatcher::new(3);
        assert_eq!(rm.average_regret(), 0.0);
    }

    #[test]
    fn test_average_regret_positive_after_dominant_action() {
        let mut rm = DiscountedRegretMatcher::new(3);
        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert!(rm.average_regret() > 0.0);
    }
}
