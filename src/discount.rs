//! Discount parameter configurations for DCFR variants.
//!
//! This module provides the [`DiscountParams`] struct for configuring
//! discounted counterfactual regret minimization (DCFR) algorithms.

/// Configuration for discount factors in DCFR.
///
/// The three parameters control how regrets and average strategy weights
/// are discounted over time:
///
/// - `alpha`: Exponent for positive regret discount
/// - `beta`: Exponent for negative regret discount
/// - `gamma`: Exponent for average strategy weight discount
///
/// The discount factor at iteration `t` is computed as `t^exp / (t^exp + 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscountParams {
    /// Exponent for positive regret discount.
    pub alpha: f32,
    /// Exponent for negative regret discount.
    pub beta: f32,
    /// Exponent for average strategy discount.
    pub gamma: f32,
}

impl DiscountParams {
    /// Linear CFR: DCFR_{1,1,1}
    ///
    /// All regrets and strategy weights are discounted linearly.
    pub const LCFR: Self = Self {
        alpha: 1.0,
        beta: 1.0,
        gamma: 1.0,
    };

    /// Recommended DCFR: DCFR_{1.5,0,2}
    ///
    /// This configuration has been shown to provide fast convergence
    /// in practice. Positive regrets are discounted aggressively (α=1.5),
    /// negative regrets are not discounted (β=0), and strategy weights
    /// use quadratic discounting (γ=2).
    pub const RECOMMENDED: Self = Self {
        alpha: 1.5,
        beta: 0.0,
        gamma: 2.0,
    };

    /// Pruning-safe DCFR: DCFR_{1.5,0.5,2}
    ///
    /// A variant with moderate negative regret discounting that is
    /// safer to use with regret-based pruning techniques.
    pub const PRUNING_SAFE: Self = Self {
        alpha: 1.5,
        beta: 0.5,
        gamma: 2.0,
    };

    /// Creates a new `DiscountParams` with custom values.
    #[must_use]
    pub const fn new(alpha: f32, beta: f32, gamma: f32) -> Self {
        Self { alpha, beta, gamma }
    }

    /// Computes the discount factor: `t^exp / (t^exp + 1)`.
    ///
    /// For `exp = 0`, this returns `0.5` for all `t > 0`.
    /// As `exp` increases, the discount approaches 1 faster.
    /// As `exp` decreases toward negative values, the discount approaches 0.
    #[must_use]
    pub fn discount_factor(t: usize, exp: f32) -> f32 {
        let t_f = t as f32;
        let t_pow = t_f.powf(exp);
        t_pow / (t_pow + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discount_factor_zero_exp() {
        // t^0 / (t^0 + 1) = 1 / 2 = 0.5 for any t > 0
        assert!((DiscountParams::discount_factor(1, 0.0) - 0.5).abs() < 1e-6);
        assert!((DiscountParams::discount_factor(10, 0.0) - 0.5).abs() < 1e-6);
        assert!((DiscountParams::discount_factor(100, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_discount_factor_one_exp() {
        // t^1 / (t^1 + 1) = t / (t + 1)
        assert!((DiscountParams::discount_factor(1, 1.0) - 0.5).abs() < 1e-6);
        assert!((DiscountParams::discount_factor(2, 1.0) - 2.0 / 3.0).abs() < 1e-6);
        assert!((DiscountParams::discount_factor(9, 1.0) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_discount_factor_increases_with_t() {
        for exp in [0.5, 1.0, 1.5, 2.0] {
            let d1 = DiscountParams::discount_factor(1, exp);
            let d2 = DiscountParams::discount_factor(10, exp);
            let d3 = DiscountParams::discount_factor(100, exp);
            assert!(d1 < d2);
            assert!(d2 < d3);
        }
    }

    #[test]
    fn test_presets() {
        assert_eq!(DiscountParams::LCFR.alpha, 1.0);
        assert_eq!(DiscountParams::LCFR.beta, 1.0);
        assert_eq!(DiscountParams::LCFR.gamma, 1.0);

        assert_eq!(DiscountParams::RECOMMENDED.alpha, 1.5);
        assert_eq!(DiscountParams::RECOMMENDED.beta, 0.0);
        assert_eq!(DiscountParams::RECOMMENDED.gamma, 2.0);

        assert_eq!(DiscountParams::PRUNING_SAFE.alpha, 1.5);
        assert_eq!(DiscountParams::PRUNING_SAFE.beta, 0.5);
        assert_eq!(DiscountParams::PRUNING_SAFE.gamma, 2.0);
    }
}
