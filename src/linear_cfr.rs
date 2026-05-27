#![allow(clippy::cast_precision_loss)]
//! Linear CFR implementation.
//!
//! This module provides [`LinearCfrRegretMatcher`], which applies
//! linear time-weighting to both regrets and strategies.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use crate::regret_minimizer::{self, RegretMinimizer};
use crate::{probability, vector_ops};

/// A regret matcher implementing Linear CFR.
///
/// Linear CFR applies linear time-weighting to regrets, giving more
/// importance to recent iterations. Unlike DCFR which discounts old
/// regrets, Linear CFR weights new regrets by the iteration number.
///
/// Update rules:
/// - Regret: `R^t = R^{t-1} + t * r^t`
/// - Strategy: `x^{t+1} = [R^t]^+ / ||[R^t]^+||_1`
/// - Cumulative: `X^t = X^{t-1} + t * x^t`
#[derive(Debug, Clone)]
pub struct LinearCfrRegretMatcher {
    p: Vec<f32>,
    sum_p: Vec<f32>,
    cumulative_regret: Vec<f32>,
    num_updates: usize,
}

impl RegretMinimizer for LinearCfrRegretMatcher {
    fn new(num_experts: usize) -> Self {
        let p = probability::uniform_weights(num_experts);
        Self {
            p,
            sum_p: vec![0.0; num_experts],
            cumulative_regret: vec![0.0; num_experts],
            num_updates: 0,
        }
    }

    fn update_regret(&mut self, rewards: &[f32]) {
        let t = (self.num_updates + 1) as f32;

        let expected = vector_ops::dot(&self.p, rewards);

        // Linear CFR: R^t = R^{t-1} + t * r^t
        for (cr, &rw) in self.cumulative_regret.iter_mut().zip(rewards) {
            *cr += t * (rw - expected);
        }

        regret_minimizer::regret_match(&self.cumulative_regret, &mut self.p);

        // Linear weighting: X^t = X^{t-1} + t * x^t
        vector_ops::scaled_add_assign(&mut self.sum_p, t, &self.p);
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

    /// Linear CFR weights iteration `t`'s regret by `t`, so the total weight
    /// after `T` updates is `Σ_{t=1}^{T} t = T(T+1)/2`.
    fn regret_weight_total(&self) -> f32 {
        let t = self.num_updates as f32;
        t * (t + 1.0) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    #[test]
    fn test_linear_cfr_new() {
        let _rm = LinearCfrRegretMatcher::new(3);
    }

    #[test]
    fn test_next_action() {
        let rm = LinearCfrRegretMatcher::new(100);
        let mut rng = rng();
        for _ in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = LinearCfrRegretMatcher::new(3);

        for _ in 0..10 {
            rm.update_regret(&[1.0, 0.0, -1.0]);
        }

        let weights = rm.best_weight();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_num_updates_increments() {
        let mut rm = LinearCfrRegretMatcher::new(3);
        assert_eq!(rm.num_updates(), 0);

        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert_eq!(rm.num_updates(), 1);
    }

    #[test]
    fn test_cumulative_regret_len() {
        let mut rm = LinearCfrRegretMatcher::new(4);
        assert_eq!(rm.cumulative_regret().len(), 4);

        rm.update_regret(&[1.0, 0.0, -1.0, 0.5]);
        assert_eq!(rm.cumulative_regret().len(), 4);
    }

    #[test]
    fn test_average_regret_zero_before_updates() {
        let rm = LinearCfrRegretMatcher::new(3);
        assert_eq!(rm.average_regret(), 0.0);
    }

    #[test]
    fn test_average_regret_positive_after_dominant_action() {
        let mut rm = LinearCfrRegretMatcher::new(3);
        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert!(rm.average_regret() > 0.0);
    }

    #[test]
    fn test_average_regret_normalizes_by_linear_weight() {
        let mut rm = LinearCfrRegretMatcher::new(3);

        // Update 1 (t=1) from uniform: expected = 0, inst = [1, 0, -1],
        // R = 1*inst = [1, 0, -1]. Weight total W = 1.
        // average_regret = max(0, max R) / W = 1.0 / 1.
        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert!((rm.average_regret() - 1.0).abs() < 1e-6);

        // After update 1, regret matching gives strategy [1, 0, 0].
        // Update 2 (t=2): expected = dot([1,0,0], [1,0,-1]) = 1.0,
        // inst = [0, -1, -2], R += 2*inst = [1, -2, -5]. max positive = 1.
        // Weight total W = 1 + 2 = 3, so average_regret = 1/3.
        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert!((rm.average_regret() - 1.0 / 3.0).abs() < 1e-6);
    }
}
