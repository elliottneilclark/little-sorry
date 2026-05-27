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
/// [`RegretMinimizer::new`], [`RegretMinimizer::update_regret`],
/// [`RegretMinimizer::num_updates`], [`RegretMinimizer::current_strategy`],
/// [`RegretMinimizer::cumulative_strategy`], and
/// [`RegretMinimizer::cumulative_regret`] need to be implemented.
/// [`RegretMinimizer::next_action`], [`RegretMinimizer::best_weight`],
/// [`RegretMinimizer::regret_weight_total`], and
/// [`RegretMinimizer::average_regret`] have default implementations.
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
        crate::probability::sample_action(self.current_strategy(), rng)
    }

    /// Returns the average strategy weights (Nash equilibrium approximation).
    #[must_use]
    fn best_weight(&self) -> Vec<f32> {
        crate::probability::normalize_by_sum(self.cumulative_strategy())
    }

    /// Per-action cumulative regret after the updates performed so far.
    ///
    /// Length equals the number of actions/experts. Values may be **signed**
    /// (vanilla / discounted / linear CFR) or **non-negative** (CFR+ variants,
    /// which floor at zero). Use [`RegretMinimizer::average_regret`]
    /// for a convergence scalar that is well-defined regardless of sign
    /// convention.
    #[must_use]
    fn cumulative_regret(&self) -> &[f32];

    /// Total regret weight `Wᵀ` accumulated so far — the sum of the
    /// per-iteration weights this matcher applies to instantaneous regret.
    ///
    /// This is the denominator used by [`RegretMinimizer::average_regret`] so
    /// that the result is a true (weight-aware) time average. The default `T`
    /// (= [`RegretMinimizer::num_updates`]) is correct for unweighted
    /// accumulation (CFR+, PCFR+). Matchers that weight regret over time
    /// override it: Linear CFR uses `T(T+1)/2`, and the discounted variants
    /// track the discounted weight sum. Returns `0.0` before any update.
    #[must_use]
    fn regret_weight_total(&self) -> f32 {
        self.num_updates() as f32
    }

    /// Average regret: `maxₐ [Rᵀ(a)]⁺ / Wᵀ`.
    ///
    /// The standard regret-minimization convergence diagnostic — the maximum
    /// over actions of the (weight-aware) time-averaged regret, clamped to be
    /// non-negative. It tends to `0` as the average strategy approaches a
    /// (local) equilibrium, so a small value means the node has nearly stopped
    /// gaining regret. The typical use is a stopping criterion: keep iterating
    /// until it drops below a tolerance (see the example below).
    ///
    /// # Behavior
    ///
    /// - Always `>= 0`.
    /// - Returns `0.0` before the first
    ///   [`update_regret`](RegretMinimizer::update_regret) — and again whenever
    ///   a matcher resets its accumulators (e.g. the CFR+ all-negative reset;
    ///   see [`crate::CfrPlusRegretMatcher`]). A `0.0` reading means "no
    ///   accumulated regret", which is **not** the same as "converged after
    ///   many iterations". Gate any stopping check on a minimum
    ///   [`num_updates`](RegretMinimizer::num_updates) so that a freshly
    ///   created or just-reset matcher is not mistaken for a converged one.
    /// - The averaging weight is
    ///   [`regret_weight_total`](RegretMinimizer::regret_weight_total), which
    ///   differs per matcher (`T` for CFR+/PCFR+, `T(T+1)/2` for Linear CFR, a
    ///   discounted sum for the DCFR variants). Absolute magnitudes are
    ///   therefore **not** comparable across matchers — pick the tolerance for
    ///   the matcher you are actually using. The per-element positive part
    ///   means both signed and clamped
    ///   [`cumulative_regret`](RegretMinimizer::cumulative_regret) storage are
    ///   handled correctly.
    ///
    /// # Examples
    ///
    /// Iterate until the average regret falls below a tolerance, but only after
    /// a minimum number of updates so the initial `0.0` is not treated as
    /// convergence:
    ///
    /// ```
    /// use little_sorry::{DcfrPlusRegretMatcher, RegretMinimizer};
    ///
    /// let mut matcher = DcfrPlusRegretMatcher::new(3);
    /// let tolerance = 0.01;
    /// let min_iterations = 100;
    ///
    /// for t in 0..10_000 {
    ///     matcher.update_regret(&[1.0, -0.5, 0.2]);
    ///     if t >= min_iterations && matcher.average_regret() < tolerance {
    ///         break;
    ///     }
    /// }
    /// assert!(matcher.average_regret() >= 0.0);
    /// ```
    #[must_use]
    fn average_regret(&self) -> f32 {
        let w = self.regret_weight_total();
        if w <= 0.0 {
            return 0.0;
        }
        let max_pos = self
            .cumulative_regret()
            .iter()
            .fold(0.0_f32, |m, &r| m.max(r.max(0.0)));
        max_pos / w
    }
}

/// Set `p[i] = max(0, regrets[i])` and normalize to sum to 1.
/// Falls back to uniform if no positive regrets exist.
pub(crate) fn regret_match(regrets: &[f32], p: &mut [f32]) {
    for (pi, &r) in p.iter_mut().zip(regrets) {
        *pi = r.max(0.0);
    }
    crate::probability::normalize_inplace(p);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ── average_regret default ──────────────────────────────────────

    /// The provided `average_regret` default must equal
    /// `max(0, max cumulative_regret) / regret_weight_total` for any
    /// implementor.
    fn assert_default_formula<M: RegretMinimizer>(m: &M) {
        let w = m.regret_weight_total();
        let expected = if w <= 0.0 {
            0.0
        } else {
            m.cumulative_regret()
                .iter()
                .fold(0.0_f32, |acc, &r| acc.max(r.max(0.0)))
                / w
        };
        assert!(
            (m.average_regret() - expected).abs() < 1e-6,
            "average_regret {} != formula {}",
            m.average_regret(),
            expected
        );
    }

    #[test]
    fn test_average_regret_matches_formula() {
        use crate::{DiscountedRegretMatcher, PcfrPlusRegretMatcher};

        // Before any update: the W == 0 guard yields 0.0.
        let mut m = PcfrPlusRegretMatcher::new(3);
        assert_eq!(m.average_regret(), 0.0);
        assert_default_formula(&m);

        // After a known update from uniform: regret = [2, 0, 0], W = T = 1.
        m.update_regret(&[3.0, 0.0, 0.0]);
        assert!((m.average_regret() - 2.0).abs() < 1e-6);
        assert_default_formula(&m);

        // A discounted matcher normalizes by its tracked discount weight, not
        // T; average_regret must still equal the documented formula over it.
        let mut d = DiscountedRegretMatcher::new(3);
        for _ in 0..5 {
            d.update_regret(&[1.0, 0.0, -1.0]);
        }
        assert_default_formula(&d);
    }
}
