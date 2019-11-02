#![allow(clippy::cast_precision_loss)]
//! This is a library for exploring regret minimization
//! with rust. Specifically this is mostly about poker.
use ndarray::prelude::*;
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::weighted::WeightedIndex;

use std::vec::Vec;

#[derive(Debug, Clone)]
pub struct RegretMatcher {
    // How many experts we have.
    // That means this generator can emit an
    // action of 0..num_experts non inclusive.
    num_experts: usize,
    // The chance each expert has of being chosen
    p: Array1<f64>,
    sum_p: Array1<f64>,
    // The amount the expert has accumulated
    expert_reward: Array1<f64>,
    // The cumulative reward earned
    cumulative_reward: f64,
    // The distribution that generates actions.
    dist: WeightedIndex<f64>,

    num_games: usize,
}

impl RegretMatcher {
    fn rand_weights(num_experts: usize) -> Vec<f64> {
        vec![1.0 / num_experts as f64; num_experts]
    }
    #[must_use]
    pub fn new(num_experts: usize) -> Self {
        // Every expert starts out with a weight.
        // Those weights all add to 1.0f
        let p = Self::rand_weights(num_experts);
        Self::new_from_p(p)
    }

    #[must_use]
    pub fn new_from_p(p: Vec<f64>) -> Self {
        // We're going to move p so capture it now
        let num_experts = p.len();
        // Create the distribution. This is a lot of
        // precompute
        let dist = WeightedIndex::new(&p).unwrap();
        Self {
            num_experts,
            p: Array1::from(p),
            sum_p: Array1::zeros(num_experts),
            cumulative_reward: 0.0_f64,
            expert_reward: Array1::from(vec![0.0_f64; num_experts]),
            dist,
            num_games: 0,
        }
    }

    #[must_use]
    pub fn next_action(&self) -> usize {
        self.dist.sample(&mut thread_rng())
    }

    pub fn update_regret(&mut self, reward_array: ArrayView1<f64>) {
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
            self.p = Array1::from(Self::rand_weights(self.num_experts));
            self.cumulative_reward = 0.0;
            self.expert_reward = Array1::zeros(self.num_experts);
            self.num_games = 0;
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
            // Need to keep track of the number of games as well.
            self.num_games += 1;
        }
        self.dist = WeightedIndex::new(&self.p).unwrap();
    }

    #[must_use]
    pub fn best_weight(&self) -> Vec<f64> {
        (self.sum_p.clone() / self.num_games as f64).to_vec()
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
        assert!(true);
    }

    #[test]
    fn test_next_action() {
        let rg = RegretMatcher::new(100);
        for _i in 0..500 {
            let a = rg.next_action();
            assert!(a >= 0);
            assert!(a < 100);
        }
    }
}
