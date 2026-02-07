#![allow(clippy::cast_precision_loss)]
//! PCFR+ (Predictive CFR+) implementation.
//!
//! This module provides [`PcfrPlusRegretMatcher`], which uses predictions
//! of future regrets to compute the next strategy.
//!
//! Reference: "Equilibrium Finding with Weighted Regret Minimization"
//! (arXiv:2404.13891)

use ndarray::prelude::*;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedAliasIndex;

use crate::errors::LittleError;
use crate::regret_minimizer::RegretMinimizer;

/// A regret matcher implementing PCFR+ (Predictive CFR+).
///
/// PCFR+ extends CFR+ by using predictions of future regrets when
/// computing the next strategy. This can lead to faster convergence
/// by anticipating the effect of actions.
///
/// Update rules:
/// - Regret: `R^t = [R^{t-1} + r^t]^+`
/// - Predicted regret: `R̃^{t+1} = [R^t + v^{t+1}]^+`
/// - Strategy: `x^{t+1} = R̃^{t+1} / ||R̃^{t+1}||_1`
/// - Cumulative: `X^t = X^{t-1} + t^2 * x^t` (quadratic weighting)
///
/// The prediction `v^{t+1}` uses the current instantaneous regret
/// as a simple prediction for the next iteration.
#[derive(Debug, Clone)]
pub struct PcfrPlusRegretMatcher {
    p: Array1<f32>,
    sum_p: Array1<f32>,
    cumulative_regret: Array1<f32>,
    last_instantaneous_regret: Array1<f32>,
    dist: WeightedAliasIndex<f32>,
    num_updates: usize,
}

impl PcfrPlusRegretMatcher {
    fn init_weights(num_experts: usize) -> Vec<f32> {
        vec![1.0 / num_experts as f32; num_experts]
    }
}

impl RegretMinimizer for PcfrPlusRegretMatcher {
    fn new(num_experts: usize) -> Result<Self, LittleError> {
        let p = Self::init_weights(num_experts);
        let dist = WeightedAliasIndex::new(p.clone())?;
        Ok(Self {
            p: Array1::from(p),
            sum_p: Array1::zeros(num_experts),
            cumulative_regret: Array1::zeros(num_experts),
            last_instantaneous_regret: Array1::zeros(num_experts),
            dist,
            num_updates: 0,
        })
    }

    fn next_action<R: rand::Rng>(&self, rng: &mut R) -> usize {
        self.dist.sample(rng)
    }

    fn update_regret(&mut self, reward_array: ArrayView1<f32>) -> Result<(), LittleError> {
        let num_experts = self.p.len();
        let t = self.num_updates + 1;

        // Compute instantaneous regret
        let expected_reward = self.p.dot(&reward_array);
        let instantaneous_regret = &reward_array - expected_reward;

        // CFR+ regret update: R^t = [R^{t-1} + r^t]^+
        for i in 0..num_experts {
            self.cumulative_regret[i] =
                f32::max(0.0, self.cumulative_regret[i] + instantaneous_regret[i]);
        }

        // Store for prediction in next iteration
        self.last_instantaneous_regret = instantaneous_regret.to_owned();

        // Compute predicted regrets: R̃^{t+1} = [R^t + v^{t+1}]^+
        // where v^{t+1} = r^t (current regret as prediction)
        let predicted_regret: Array1<f32> = self
            .cumulative_regret
            .iter()
            .zip(self.last_instantaneous_regret.iter())
            .map(|(&r, &v)| f32::max(0.0, r + v))
            .collect();

        // Compute new strategy from predicted regrets
        let regret_sum = predicted_regret.sum();

        if regret_sum <= 0.0 {
            self.p = Array1::from(Self::init_weights(num_experts));
        } else {
            self.p = predicted_regret / regret_sum;
        }

        // Quadratic weighting for cumulative strategy: X^t = X^{t-1} + t^2 * x^t
        let weight = (t * t) as f32;
        self.sum_p = &self.sum_p + &(&self.p * weight);
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
    fn test_pcfr_plus_new() {
        let _rm = PcfrPlusRegretMatcher::new(3).unwrap();
    }

    #[test]
    fn test_next_action() {
        let rm = PcfrPlusRegretMatcher::new(100).unwrap();
        let mut rng = rng();
        for _ in 0..500 {
            let a = rm.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_best_weight_sums_to_one() {
        let mut rm = PcfrPlusRegretMatcher::new(3).unwrap();
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
        let mut rm = PcfrPlusRegretMatcher::new(3).unwrap();
        assert_eq!(rm.num_updates(), 0);

        let rewards = array![1.0_f32, 0.0_f32, -1.0_f32];
        rm.update_regret(rewards.view()).unwrap();
        assert_eq!(rm.num_updates(), 1);
    }
}
