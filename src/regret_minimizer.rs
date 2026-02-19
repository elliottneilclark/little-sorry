#![allow(clippy::cast_precision_loss)]
//! Trait definition and shared helpers for regret minimization algorithms.
//!
//! This module provides the [`RegretMinimizer`] trait which defines
//! the common interface for CFR variants like CFR+ and DCFR, along
//! with helper functions used across implementations.

/// A trait for regret minimization algorithms.
///
/// Implementors of this trait can be used interchangeably in game-solving
/// algorithms that rely on regret matching. The trait provides a common
/// interface for different CFR variants.
///
/// Only [`RegretMinimizer::new`], [`RegretMinimizer::update_regret`],
/// [`RegretMinimizer::num_updates`], [`RegretMinimizer::current_strategy`],
/// and [`RegretMinimizer::cumulative_strategy`] need to be implemented.
/// [`RegretMinimizer::next_action`] and [`RegretMinimizer::best_weight`]
/// have default implementations.
pub trait RegretMinimizer: Clone {
    /// Creates a new regret minimizer with the given number of actions/experts.
    ///
    /// # Panics
    ///
    /// Panics if `num_experts` is 0.
    fn new(num_experts: usize) -> Self
    where
        Self: Sized;

    /// Updates regrets based on the reward slice.
    ///
    /// This method should:
    /// 1. Compute instantaneous regret for each action
    /// 2. Update cumulative regrets (with algorithm-specific discounting)
    /// 3. Derive new strategy from regret matching
    /// 4. Update average strategy weights
    fn update_regret(&mut self, rewards: &[f32]);

    /// Returns the number of updates performed.
    #[must_use]
    fn num_updates(&self) -> usize;

    /// Returns the current strategy (probability distribution over actions).
    #[must_use]
    fn current_strategy(&self) -> &[f32];

    /// Returns the cumulative strategy weights.
    ///
    /// To get the average strategy (Nash equilibrium approximation),
    /// use [`RegretMinimizer::best_weight`] or normalize this by its sum.
    #[must_use]
    fn cumulative_strategy(&self) -> &[f32];

    /// Samples the next action according to the current strategy.
    fn next_action<R: rand::Rng>(&self, rng: &mut R) -> usize {
        sample_action(self.current_strategy(), rng)
    }

    /// Returns the average strategy weights (Nash equilibrium approximation).
    #[must_use]
    fn best_weight(&self) -> Vec<f32> {
        normalize_by_sum(self.cumulative_strategy())
    }
}

/// Create a uniform probability distribution of length `n`.
pub(crate) fn uniform_weights(n: usize) -> Vec<f32> {
    vec![1.0 / n as f32; n]
}

/// Fill a slice with uniform probabilities.
pub(crate) fn uniform_fill(p: &mut [f32]) {
    let uniform = 1.0 / p.len() as f32;
    p.fill(uniform);
}

/// Sample an action index from a probability distribution using cumulative sum.
pub(crate) fn sample_action<R: rand::Rng>(p: &[f32], rng: &mut R) -> usize {
    let r: f32 = rng.random();
    let mut cumsum = 0.0;
    for (i, &prob) in p.iter().enumerate() {
        cumsum += prob;
        if r < cumsum {
            return i;
        }
    }
    p.len() - 1
}

/// Dot product of two slices.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// dst[i] += src[i]
pub(crate) fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// Set `p[i] = max(0, regrets[i])` and normalize to sum to 1.
/// Falls back to uniform if no positive regrets exist.
pub(crate) fn regret_match(regrets: &[f32], p: &mut [f32]) {
    for (pi, &r) in p.iter_mut().zip(regrets) {
        *pi = r.max(0.0);
    }
    normalize_inplace(p);
}

/// Normalize non-negative values in `p` to sum to 1.
/// Falls back to uniform if sum is zero.
pub(crate) fn normalize_inplace(p: &mut [f32]) {
    let sum: f32 = p.iter().sum();
    if sum <= 0.0 {
        uniform_fill(p);
    } else {
        let inv = 1.0 / sum;
        for pi in p.iter_mut() {
            *pi *= inv;
        }
    }
}

/// Normalize a slice by its sum, returning a new Vec.
/// Falls back to uniform if sum is zero.
pub(crate) fn normalize_by_sum(sum_p: &[f32]) -> Vec<f32> {
    let sum: f32 = sum_p.iter().sum();
    if sum <= 0.0 {
        uniform_weights(sum_p.len())
    } else {
        let inv = 1.0 / sum;
        sum_p.iter().map(|&v| v * inv).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── uniform_weights ─────────────────────────────────────────────

    #[test]
    fn test_uniform_weights_sums_to_one() {
        for n in 1..=10 {
            let w = uniform_weights(n);
            assert_eq!(w.len(), n);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_uniform_weights_all_equal() {
        let w = uniform_weights(4);
        assert!(w.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    // ── uniform_fill ────────────────────────────────────────────────

    #[test]
    fn test_uniform_fill() {
        let mut p = vec![0.0; 5];
        uniform_fill(&mut p);
        assert!(p.iter().all(|&v| (v - 0.2).abs() < 1e-6));
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    // ── dot ─────────────────────────────────────────────────────────

    #[test]
    fn test_dot_basic() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_zeros() {
        assert!((dot(&[0.0, 0.0], &[1.0, 2.0])).abs() < 1e-6);
    }

    #[test]
    fn test_dot_single() {
        assert!((dot(&[3.0], &[7.0]) - 21.0).abs() < 1e-6);
    }

    // ── add_assign ──────────────────────────────────────────────────

    #[test]
    fn test_add_assign() {
        let mut dst = vec![1.0, 2.0, 3.0];
        add_assign(&mut dst, &[10.0, 20.0, 30.0]);
        assert!((dst[0] - 11.0).abs() < 1e-6);
        assert!((dst[1] - 22.0).abs() < 1e-6);
        assert!((dst[2] - 33.0).abs() < 1e-6);
    }

    // ── normalize_inplace ───────────────────────────────────────────

    #[test]
    fn test_normalize_inplace_positive() {
        let mut p = vec![2.0, 3.0, 5.0];
        normalize_inplace(&mut p);
        assert!((p[0] - 0.2).abs() < 1e-6);
        assert!((p[1] - 0.3).abs() < 1e-6);
        assert!((p[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_inplace_all_zeros_falls_back_to_uniform() {
        let mut p = vec![0.0, 0.0, 0.0];
        normalize_inplace(&mut p);
        let expected = 1.0 / 3.0;
        assert!(p.iter().all(|&v| (v - expected).abs() < 1e-6));
    }

    // ── normalize_by_sum ────────────────────────────────────────────

    #[test]
    fn test_normalize_by_sum_positive() {
        let result = normalize_by_sum(&[1.0, 3.0]);
        assert!((result[0] - 0.25).abs() < 1e-6);
        assert!((result[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_by_sum_zeros_falls_back_to_uniform() {
        let result = normalize_by_sum(&[0.0, 0.0, 0.0, 0.0]);
        assert!(result.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    // ── regret_match ────────────────────────────────────────────────

    #[test]
    fn test_regret_match_positive_regrets() {
        let regrets = [2.0, 0.0, 8.0];
        let mut p = vec![0.0; 3];
        regret_match(&regrets, &mut p);
        assert!((p[0] - 0.2).abs() < 1e-6);
        assert!((p[1]).abs() < 1e-6);
        assert!((p[2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_regret_match_all_negative_falls_back_to_uniform() {
        let regrets = [-1.0, -2.0, -3.0];
        let mut p = vec![0.0; 3];
        regret_match(&regrets, &mut p);
        let expected = 1.0 / 3.0;
        assert!(p.iter().all(|&v| (v - expected).abs() < 1e-6));
    }

    #[test]
    fn test_regret_match_mixed_regrets() {
        let regrets = [-5.0, 3.0, 7.0];
        let mut p = vec![0.0; 3];
        regret_match(&regrets, &mut p);
        assert!((p[0]).abs() < 1e-6); // negative clipped to 0
        assert!((p[1] - 0.3).abs() < 1e-6);
        assert!((p[2] - 0.7).abs() < 1e-6);
    }

    // ── sample_action ───────────────────────────────────────────────

    #[test]
    fn test_sample_action_deterministic_distribution() {
        let mut rng = rand::rng();
        // All weight on action 2
        let p = [0.0, 0.0, 1.0];
        for _ in 0..100 {
            assert_eq!(sample_action(&p, &mut rng), 2);
        }
    }

    #[test]
    fn test_sample_action_uniform_hits_all_actions() {
        let mut rng = rand::rng();
        let p = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let mut seen = [false; 3];
        for _ in 0..500 {
            seen[sample_action(&p, &mut rng)] = true;
        }
        assert!(seen.iter().all(|&s| s), "expected all actions sampled");
    }
}
