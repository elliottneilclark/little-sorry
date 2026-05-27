#![allow(clippy::cast_precision_loss)]
//! PCFR+ (Predictive CFR+) implementation.
//!
//! This module provides [`PcfrPlusRegretMatcher`], which uses predictions
//! of future regrets to compute the next strategy.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use crate::regret_minimizer::RegretMinimizer;
use crate::{probability, vector_ops};

/// A regret matcher implementing PCFR+ (Predictive CFR+).
///
/// PCFR+ extends CFR+ by using predictions of future regrets when
/// computing the next strategy. This can lead to faster convergence
/// by anticipating the effect of actions.
///
/// Update rules:
/// - Regret: `R^t = [R^{t-1} + r^t]^+`
/// - Predicted regret: `R_tilde^{t+1} = [R^t + v^{t+1}]^+`
/// - Strategy: `x^{t+1} = R_tilde^{t+1} / ||R_tilde^{t+1}||_1`
/// - Cumulative: `X^t = X^{t-1} + t^2 * x^t` (quadratic weighting)
#[derive(Debug, Clone)]
pub struct PcfrPlusRegretMatcher {
    p: Vec<f32>,
    sum_p: Vec<f32>,
    cumulative_regret: Vec<f32>,
    last_instantaneous_regret: Vec<f32>,
    num_updates: usize,
}

impl RegretMinimizer for PcfrPlusRegretMatcher {
    fn new(num_experts: usize) -> Self {
        let p = probability::uniform_weights(num_experts);
        Self {
            p,
            sum_p: vec![0.0; num_experts],
            cumulative_regret: vec![0.0; num_experts],
            last_instantaneous_regret: vec![0.0; num_experts],
            num_updates: 0,
        }
    }

    fn update_regret(&mut self, rewards: &[f32]) {
        let t = self.num_updates + 1;

        let expected = vector_ops::dot(&self.p, rewards);

        // CFR+ regret update + store instantaneous for prediction
        for ((cr, lr), &rw) in self
            .cumulative_regret
            .iter_mut()
            .zip(self.last_instantaneous_regret.iter_mut())
            .zip(rewards)
        {
            let inst = rw - expected;
            *cr = (*cr + inst).max(0.0);
            *lr = inst;
        }

        // Compute predicted regrets into p: R_tilde = [R^t + v]^+
        for ((pi, &cr), &lr) in self
            .p
            .iter_mut()
            .zip(self.cumulative_regret.iter())
            .zip(self.last_instantaneous_regret.iter())
        {
            *pi = (cr + lr).max(0.0);
        }
        probability::normalize_inplace(&mut self.p);

        // Quadratic weighting: X^t = X^{t-1} + t^2 * x^t
        let weight = (t * t) as f32;
        vector_ops::scaled_add_assign(&mut self.sum_p, weight, &self.p);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    #[test]
    fn test_pcfr_plus_new() {
        let _rm = PcfrPlusRegretMatcher::new(3);
    }

    #[test]
    fn test_next_action() {
        let rm = PcfrPlusRegretMatcher::new(100);
        let mut rng = rng();
        for _ in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = PcfrPlusRegretMatcher::new(3);

        for _ in 0..10 {
            rm.update_regret(&[1.0, 0.0, -1.0]);
        }

        let weights = rm.best_weight();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_num_updates_increments() {
        let mut rm = PcfrPlusRegretMatcher::new(3);
        assert_eq!(rm.num_updates(), 0);

        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert_eq!(rm.num_updates(), 1);
    }

    #[test]
    fn test_cumulative_regret_len() {
        let mut rm = PcfrPlusRegretMatcher::new(4);
        assert_eq!(rm.cumulative_regret().len(), 4);

        rm.update_regret(&[1.0, 0.0, -1.0, 0.5]);
        assert_eq!(rm.cumulative_regret().len(), 4);
    }

    #[test]
    fn test_average_regret_zero_before_updates() {
        let rm = PcfrPlusRegretMatcher::new(3);
        assert_eq!(rm.average_regret(), 0.0);
    }

    #[test]
    fn test_average_regret_positive_after_dominant_action() {
        let mut rm = PcfrPlusRegretMatcher::new(3);
        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert!(rm.average_regret() > 0.0);
    }

    /// CFR+ regret accumulation: `R^t = [R^{t-1} + r^t]^+`, where the
    /// instantaneous regret is `r^t = reward - dot(strategy, reward)`.
    #[test]
    fn test_cumulative_regret_matches_hand_computation() {
        let mut rm = PcfrPlusRegretMatcher::new(3);

        // Update 1 from uniform strategy [1/3, 1/3, 1/3]:
        //   expected = (1.0 + 0.5 + 0.0) / 3 = 0.5
        //   inst     = [0.5, 0.0, -0.5]
        //   R^1      = [0.5, 0.0, 0.0]   (negative floored)
        rm.update_regret(&[1.0, 0.5, 0.0]);
        let cr = rm.cumulative_regret();
        assert!((cr[0] - 0.5).abs() < 1e-6);
        assert!(cr[1].abs() < 1e-6);
        assert!(cr[2].abs() < 1e-6);
        assert!((rm.average_regret() - 0.5).abs() < 1e-6);

        // After update 1 the strategy saturates to [1, 0, 0].
        // Update 2 with reward favoring action 1:
        //   expected = dot([1,0,0], [0,1,0]) = 0.0
        //   inst     = [0.0, 1.0, 0.0]
        //   R^2      = [0.5, 1.0, 0.0]
        rm.update_regret(&[0.0, 1.0, 0.0]);
        let cr = rm.cumulative_regret();
        assert!((cr[0] - 0.5).abs() < 1e-6);
        assert!((cr[1] - 1.0).abs() < 1e-6);
        assert!(cr[2].abs() < 1e-6);
        // max positive regret = 1.0, W = T = 2 → 0.5
        assert!((rm.average_regret() - 0.5).abs() < 1e-6);
    }
}
