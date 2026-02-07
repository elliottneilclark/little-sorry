#![allow(clippy::cast_precision_loss)]
//! Linear CFR implementation.
//!
//! This module provides [`LinearCfrRegretMatcher`], which applies
//! linear time-weighting to both regrets and strategies.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use ndarray::prelude::*;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedAliasIndex;

use crate::errors::LittleError;
use crate::regret_minimizer::RegretMinimizer;

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
///
/// This gives more weight to later iterations, allowing faster
/// adaptation to the opponent's strategy.
#[derive(Debug, Clone)]
pub struct LinearCfrRegretMatcher {
    p: Array1<f32>,
    sum_p: Array1<f32>,
    cumulative_regret: Array1<f32>,
    dist: WeightedAliasIndex<f32>,
    num_updates: usize,
}

impl LinearCfrRegretMatcher {
    fn init_weights(num_experts: usize) -> Vec<f32> {
        vec![1.0 / num_experts as f32; num_experts]
    }
}

impl RegretMinimizer for LinearCfrRegretMatcher {
    fn new(num_experts: usize) -> Result<Self, LittleError> {
        let p = Self::init_weights(num_experts);
        let dist = WeightedAliasIndex::new(p.clone())?;
        Ok(Self {
            p: Array1::from(p),
            sum_p: Array1::zeros(num_experts),
            cumulative_regret: Array1::zeros(num_experts),
            dist,
            num_updates: 0,
        })
    }

    fn next_action<R: rand::Rng>(&self, rng: &mut R) -> usize {
        self.dist.sample(rng)
    }

    fn update_regret(&mut self, reward_array: ArrayView1<f32>) -> Result<(), LittleError> {
        let num_experts = self.p.len();
        let t = (self.num_updates + 1) as f32;

        // Compute instantaneous regret
        let expected_reward = self.p.dot(&reward_array);
        let instantaneous_regret = &reward_array - expected_reward;

        // Linear CFR: R^t = R^{t-1} + t * r^t
        self.cumulative_regret = &self.cumulative_regret + &(&instantaneous_regret * t);

        // Compute positive regrets for strategy
        let positive_regret: Array1<f32> = self
            .cumulative_regret
            .iter()
            .map(|&r| f32::max(0.0, r))
            .collect();

        let regret_sum = positive_regret.sum();

        if regret_sum <= 0.0 {
            self.p = Array1::from(Self::init_weights(num_experts));
        } else {
            self.p = positive_regret / regret_sum;
        }

        // Linear weighting for cumulative strategy: X^t = X^{t-1} + t * x^t
        self.sum_p = &self.sum_p + &(&self.p * t);
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
    fn test_linear_cfr_new() {
        let _rm = LinearCfrRegretMatcher::new(3).unwrap();
    }

    #[test]
    fn test_next_action() {
        let rm = LinearCfrRegretMatcher::new(100).unwrap();
        let mut rng = rng();
        for _ in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = LinearCfrRegretMatcher::new(3).unwrap();
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
        let mut rm = LinearCfrRegretMatcher::new(3).unwrap();
        assert_eq!(rm.num_updates(), 0);

        let rewards = array![1.0_f32, 0.0_f32, -1.0_f32];
        rm.update_regret(rewards.view()).unwrap();
        assert_eq!(rm.num_updates(), 1);
    }
}
