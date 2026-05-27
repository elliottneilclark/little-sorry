#![allow(clippy::cast_precision_loss)]
//! Operations on probability distributions over the simplex.
//!
//! Every routine here treats its argument as a categorical distribution: the
//! results sum to one, and degenerate inputs (empty, or non-positive total
//! mass) fall back to a uniform distribution. This is where the crate
//! centralizes uniform construction, normalization, and categorical sampling.

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
    use rand::RngExt;
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
