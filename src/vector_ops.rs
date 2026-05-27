//! Element-wise `f32`-slice arithmetic shared across the regret matchers.
//!
//! These are the crate's small numeric kernels. They are written to
//! auto-vectorize: each takes `&[f32]` / `&mut [f32]` slices and iterates with
//! `iter_mut().zip()` (no indexing, so no per-element bounds checks), and the
//! scaled variants (`scaled_add_assign`, `discounted_accumulate`) use the fused
//! `a * b + c` shape that LLVM lowers to vectorized multiply-add in release
//! builds. A `&mut` slice is guaranteed not to alias a `&` slice, which LLVM
//! relies on to vectorize the read-modify-write loops.
//!
//! [`dot`] is the exception: its `.sum()` reduction stays a sequential
//! add-chain because LLVM will not reassociate floating-point addition by
//! default (doing so would change the result). Action counts are small here,
//! so this is intentional.

/// Dot product `Σ a[i] · b[i]`.
///
/// Iterates over the overlap if the slices differ in length.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// `dst[i] += src[i]` for each overlapping element.
#[inline]
pub(crate) fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// `dst[i] += scale * src[i]` for each overlapping element.
///
/// The fused multiply-add shape vectorizes well in release builds.
#[inline]
pub(crate) fn scaled_add_assign(dst: &mut [f32], scale: f32, src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += scale * s;
    }
}

/// `dst[i] = dst[i] * factor + src[i]` for each overlapping element.
///
/// Discounted accumulation: scale the running total by `factor`, then add the
/// new contribution. The explicit `*d * factor + s` form (two roundings, not a
/// fused `mul_add`) preserves the exact arithmetic of the loops it replaces.
#[inline]
pub(crate) fn discounted_accumulate(dst: &mut [f32], factor: f32, src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = *d * factor + s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ── scaled_add_assign ───────────────────────────────────────────

    #[test]
    fn test_scaled_add_assign_basic() {
        let mut dst = vec![1.0, 2.0, 3.0];
        scaled_add_assign(&mut dst, 2.0, &[10.0, 20.0, 30.0]);
        assert!((dst[0] - 21.0).abs() < 1e-6); // 1 + 2*10
        assert!((dst[1] - 42.0).abs() < 1e-6); // 2 + 2*20
        assert!((dst[2] - 63.0).abs() < 1e-6); // 3 + 2*30
    }

    #[test]
    fn test_scaled_add_assign_zero_scale_is_noop() {
        let mut dst = vec![1.0, 2.0, 3.0];
        scaled_add_assign(&mut dst, 0.0, &[10.0, 20.0, 30.0]);
        assert!((dst[0] - 1.0).abs() < 1e-6);
        assert!((dst[1] - 2.0).abs() < 1e-6);
        assert!((dst[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_add_assign_negative_scale() {
        let mut dst = vec![5.0, 5.0];
        scaled_add_assign(&mut dst, -1.0, &[1.0, 2.0]);
        assert!((dst[0] - 4.0).abs() < 1e-6);
        assert!((dst[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_add_assign_empty() {
        let mut dst: Vec<f32> = vec![];
        scaled_add_assign(&mut dst, 2.0, &[]);
        assert!(dst.is_empty());
    }

    // ── discounted_accumulate ───────────────────────────────────────

    #[test]
    fn test_discounted_accumulate_basic() {
        let mut dst = vec![10.0, 20.0];
        discounted_accumulate(&mut dst, 0.5, &[1.0, 2.0]);
        assert!((dst[0] - 6.0).abs() < 1e-6); // 10*0.5 + 1
        assert!((dst[1] - 12.0).abs() < 1e-6); // 20*0.5 + 2
    }

    #[test]
    fn test_discounted_accumulate_factor_zero_becomes_src() {
        let mut dst = vec![10.0, 20.0, 30.0];
        discounted_accumulate(&mut dst, 0.0, &[1.0, 2.0, 3.0]);
        assert!((dst[0] - 1.0).abs() < 1e-6);
        assert!((dst[1] - 2.0).abs() < 1e-6);
        assert!((dst[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_discounted_accumulate_factor_one_adds_src() {
        // factor = 1.0 reduces to dst[i] += src[i]
        let mut dst = vec![1.0, 2.0, 3.0];
        discounted_accumulate(&mut dst, 1.0, &[10.0, 20.0, 30.0]);
        assert!((dst[0] - 11.0).abs() < 1e-6);
        assert!((dst[1] - 22.0).abs() < 1e-6);
        assert!((dst[2] - 33.0).abs() < 1e-6);
    }

    #[test]
    fn test_discounted_accumulate_empty() {
        let mut dst: Vec<f32> = vec![];
        discounted_accumulate(&mut dst, 0.5, &[]);
        assert!(dst.is_empty());
    }
}
