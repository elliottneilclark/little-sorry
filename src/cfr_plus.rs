#![allow(clippy::cast_precision_loss)]
//! CFR+ regret minimization implementation.
//!
//! This module provides [`CfrPlusRegretMatcher`], an implementation of the
//! CFR+ algorithm which floors negative regrets at zero.

use crate::regret_minimizer::{self, RegretMinimizer};

/// A regret matcher implementing CFR+ (regret matching plus).
///
/// CFR+ differs from vanilla CFR by flooring cumulative regrets at zero.
/// This can be viewed as DCFR_{inf, -inf, 2}: infinite positive discount
/// (keeping positive regrets), infinite negative discount (flooring at 0),
/// and linear average strategy weighting.
#[derive(Debug, Clone)]
pub struct CfrPlusRegretMatcher {
    p: Vec<f32>,
    sum_p: Vec<f32>,
    expert_reward: Vec<f32>,
    cumulative_reward: f32,
    num_updates: usize,
}

impl CfrPlusRegretMatcher {
    /// Creates a new `CfrPlusRegretMatcher` with custom initial probabilities.
    ///
    /// # Panics
    ///
    /// Panics if `p` is empty.
    #[must_use]
    pub fn new_from_p(p: Vec<f32>) -> Self {
        assert!(!p.is_empty(), "must have at least one expert");
        let n = p.len();
        Self {
            p,
            sum_p: vec![0.0; n],
            cumulative_reward: 0.0,
            expert_reward: vec![0.0; n],
            num_updates: 0,
        }
    }
}

impl RegretMinimizer for CfrPlusRegretMatcher {
    fn new(num_experts: usize) -> Self {
        Self::new_from_p(regret_minimizer::uniform_weights(num_experts))
    }

    fn update_regret(&mut self, rewards: &[f32]) {
        let n = self.p.len();
        let r = regret_minimizer::dot(&self.p, rewards);
        self.cumulative_reward += r;
        regret_minimizer::add_assign(&mut self.expert_reward, rewards);

        // Compute capped regrets into p and their sum
        let mut regret_sum = 0.0_f32;
        for i in 0..n {
            let regret = (self.expert_reward[i] - self.cumulative_reward).max(0.0);
            self.p[i] = regret;
            regret_sum += regret;
        }

        if regret_sum <= 0.0 {
            // Reset if all regrets are zero or negative
            regret_minimizer::uniform_fill(&mut self.p);
            self.cumulative_reward = 0.0;
            self.expert_reward.fill(0.0);
            self.num_updates = 0;
        } else {
            // Normalize to get new strategy
            let inv = 1.0 / regret_sum;
            for pi in self.p.iter_mut() {
                *pi *= inv;
            }
            // Accumulate for average strategy
            regret_minimizer::add_assign(&mut self.sum_p, &self.p);
            self.num_updates += 1;
        }
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
    fn test_cfr_plus_new() {
        let _rg = CfrPlusRegretMatcher::new(3);
    }

    #[test]
    fn test_next_action() {
        let rg = CfrPlusRegretMatcher::new(100);
        let mut rng = rng();
        for _i in 0..500 {
            let a = rg.next_action(&mut rng);
            assert!(a < 100);
        }
    }

    #[test]
    fn test_num_updates_increments() {
        let mut rm = CfrPlusRegretMatcher::new(3);
        assert_eq!(rm.num_updates(), 0);

        rm.update_regret(&[1.0, 0.0, -1.0]);
        assert_eq!(rm.num_updates(), 1);
    }
}
