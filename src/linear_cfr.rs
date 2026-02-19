#![allow(clippy::cast_precision_loss)]
//! Linear CFR implementation.
//!
//! This module provides [`LinearCfrRegretMatcher`], which applies
//! linear time-weighting to both regrets and strategies.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use crate::regret_minimizer::{self, RegretMinimizer};

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
        let p = regret_minimizer::uniform_weights(num_experts);
        Self {
            p,
            sum_p: vec![0.0; num_experts],
            cumulative_regret: vec![0.0; num_experts],
            num_updates: 0,
        }
    }

    fn update_regret(&mut self, rewards: &[f32]) {
        let t = (self.num_updates + 1) as f32;

        let expected = regret_minimizer::dot(&self.p, rewards);

        // Linear CFR: R^t = R^{t-1} + t * r^t
        for (cr, &rw) in self.cumulative_regret.iter_mut().zip(rewards) {
            *cr += t * (rw - expected);
        }

        regret_minimizer::regret_match(&self.cumulative_regret, &mut self.p);

        // Linear weighting: X^t = X^{t-1} + t * x^t
        for (sp, &pi) in self.sum_p.iter_mut().zip(self.p.iter()) {
            *sp += t * pi;
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
}
