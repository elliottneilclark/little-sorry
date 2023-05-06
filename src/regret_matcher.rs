#![allow(clippy::cast_precision_loss)]
//! This is a library for exploring regret minimization
//! with rust. Specifically this is mostly about poker.
use ndarray::prelude::*;
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::WeightedAliasIndex;

use std::vec::Vec;

use crate::errors::LittleError;

#[derive(Debug, Clone)]
pub struct RegretMatcher {
    // The chance each expert has of being chosen
    p: Array1<f64>,
    sum_p: Array1<f64>,
    // The amount the expert has accumulated
    expert_reward: Array1<f64>,
    // The cumulative reward earned
    cumulative_reward: f64,
    // The distribution that generates actions.
    dist: WeightedAliasIndex<f64>,
    num_updates: usize,
}

impl RegretMatcher {
    fn init_weights(num_experts: usize) -> Vec<f64> {
        vec![1.0 / num_experts as f64; num_experts]
    }
    pub fn new(num_experts: usize) -> Result<Self, LittleError> {
        // Every expert starts out with a weight.
        // Those weights all add to 1.0f
        let p = Self::init_weights(num_experts);
        Self::new_from_p(p)
    }
    pub fn new_from_p(p: Vec<f64>) -> Result<Self, LittleError> {
        // We're going to move p so capture it now
        let num_experts = p.len();
        // Create the distribution. This is a lot of
        // precompute
        let dist = WeightedAliasIndex::new(p.clone())?;
        Ok(Self {
            p: Array1::from(p),
            sum_p: Array1::zeros(num_experts),
            cumulative_reward: 0.0_f64,
            expert_reward: Array1::from(vec![0.0_f64; num_experts]),
            dist,
            num_updates: 0,
        })
    }
    pub fn next_action(&self) -> usize {
        self.dist.sample(&mut thread_rng())
    }

    pub fn update_regret(&mut self, reward_array: ArrayView1<f64>) -> Result<(), LittleError> {
        let num_experts = self.p.len();
        // Compute how much reward we could expect.
        // Any reward for an agent with a very low p will be very low.
        let r = self.p.dot(&reward_array);
        // Keep track of the total
        self.cumulative_reward += r;
        // Keep track of total un scaled amount each agent would win
        self.expert_reward += &reward_array;
        // The amount that each expert would be rewarded minus the expected value is the regret.
        let regret = &self.expert_reward - self.cumulative_reward;
        // Any regret that's negative is performing much worse than the
        // current suggestion. So just don't try and use it.
        let capped_regret: Array1<f64> = regret.iter().map(|v: &f64| f64::max(0.0, *v)).collect();
        let regret_sum = capped_regret.sum();
        if regret_sum <= 0.0 {
            // This shouldn't happen but if it does then don't count the previous tries.
            self.p = Array1::from(Self::init_weights(num_experts));
            self.cumulative_reward = 0.0;
            self.expert_reward = Array1::zeros(num_experts);
            self.num_updates = 0;
        } else {
            // The new probablities are the capped
            // regret over the total. This should always
            // give a number between 0 and 1. These
            // numbers should sum to 1
            self.p = capped_regret / regret_sum;
            // We'll use the sum_p to keep track of our best
            // guesses over all time. This will keep from
            // swinging wildly for any times that the more
            // than one agent has credibility.
            self.sum_p += &self.p;
            // Need to keep track of the number of times update_regret has been called.
            self.num_updates += 1;
        }
        self.dist = WeightedAliasIndex::new(self.p.to_vec())?;
        Ok(())
    }

    #[must_use]
    pub fn best_weight(&self) -> Vec<f64> {
        (self.sum_p.clone() / self.num_updates as f64).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_regret_gen_new() {
        let _rg = RegretMatcher::new(3);
    }

    #[test]
    fn test_next_action() {
        let rg = RegretMatcher::new(100).unwrap();
        for _i in 0..500 {
            let a = rg.next_action();
            assert!(a < 100);
        }
    }
}
