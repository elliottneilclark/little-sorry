//! Scalar unit-interval fixed-point: map `[0,1]` onto evenly spaced integer
//! codes `0..=max_code`. Shared by the export codec (`quantize`) and the u16
//! bounded-average strategy lane, so both round-trip identically.

/// Encode `x ∈ [0,1]` to an integer code in `0..=max_code` (round-to-nearest).
/// Out-of-range `x` is clamped so the integer cast cannot wrap.
#[inline]
pub(crate) fn encode(x: f32, max_code: u32) -> u32 {
    (x.clamp(0.0, 1.0) * max_code as f32).round() as u32
}

/// Decode an integer code in `0..=max_code` back to `[0,1]`.
#[inline]
pub(crate) fn decode(code: u32, max_code: u32) -> f32 {
    code as f32 / max_code as f32
}

/// Deterministic, stateless draw in `[0, 1)` from a cell+tick key. Same key ⇒
/// same value; no shared mutable state, so it is reproducible and race-free
/// under the `Atomic` backend. The three key components are mixed into one word
/// and run through the splitmix64 finalizer; the top 24 bits become the fraction
/// (24 bits is exactly representable in an `f32` mantissa, so the division is
/// exact and unbiased).
#[inline]
pub(crate) fn u01(row: usize, action: usize, update_count: usize) -> f32 {
    let mut z = (row as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((action as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((update_count as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 24 bits → [0, 1).
    #[allow(clippy::cast_precision_loss)]
    let num = (z >> 40) as f32;
    num / ((1u32 << 24) as f32)
}

/// Stochastic-rounding encode of `x ∈ [0,1]` to a code in `0..=max_code`, using a
/// caller-supplied draw `u01 ∈ [0,1)`. Rounds up to the next code with
/// probability equal to the fractional part, so `E[result] = clamp(x)·max_code`
/// (unbiased). Exact-integer scaled values (`frac == 0`) never round up, so the
/// endpoints round-trip exactly. Unlike round-to-nearest [`encode`], sub-quantum
/// increments survive in expectation instead of being discarded.
#[inline]
pub(crate) fn encode_stochastic(x: f32, max_code: u32, u01: f32) -> u32 {
    let scaled = x.clamp(0.0, 1.0) * max_code as f32;
    let floor = scaled.floor();
    let frac = scaled - floor;
    // Safety: `scaled` ∈ [0, max_code], so `floor` ∈ [0, max_code] and the cast
    // cannot wrap; the `+1` is clamped back to `max_code` below.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let base = floor as u32;
    (base + u32::from(u01 < frac)).min(max_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_midpoint_round_trip() {
        let max = u16::MAX as u32;
        assert_eq!(encode(0.0, max), 0);
        assert_eq!(encode(1.0, max), max);
        // round-to-nearest keeps within half a quantum
        for &x in &[0.1f32, 0.25, 0.5, 0.7777, 0.999] {
            let back = decode(encode(x, max), max);
            assert!((x - back).abs() <= 0.5 / max as f32 + 1e-7, "{x} vs {back}");
        }
    }

    #[test]
    fn out_of_range_is_clamped() {
        let max = u16::MAX as u32;
        assert_eq!(encode(2.0, max), max);
        assert_eq!(encode(-1.0, max), 0);
    }

    #[test]
    fn u01_is_deterministic_and_in_range() {
        // Same key ⇒ identical bits; draws stay in [0, 1).
        for &(r, a, t) in &[(0usize, 0usize, 1usize), (3, 1, 999), (7, 2, 67_000)] {
            assert_eq!(u01(r, a, t).to_bits(), u01(r, a, t).to_bits());
        }
        for t in 0..10_000usize {
            let u = u01(t % 5, t % 3, t);
            assert!((0.0..1.0).contains(&u), "u01 out of range: {u}");
        }
        // Distinct keys generally differ (guards against a constant generator).
        assert_ne!(u01(0, 0, 1), u01(0, 0, 2));
        assert_ne!(u01(0, 0, 1), u01(1, 0, 1));
        assert_ne!(u01(0, 0, 1), u01(0, 1, 1));
    }

    #[test]
    fn encode_stochastic_endpoints_and_clamp() {
        let max = u16::MAX as u32;
        for &u in &[0.0f32, 0.5, 0.999_999] {
            assert_eq!(encode_stochastic(0.0, max, u), 0);
            assert_eq!(encode_stochastic(1.0, max, u), max);
            assert_eq!(encode_stochastic(2.0, max, u), max); // clamped high
            assert_eq!(encode_stochastic(-1.0, max, u), 0); // clamped low
        }
    }

    #[test]
    fn encode_stochastic_brackets_and_stays_in_range() {
        let max = u16::MAX as u32;
        for i in 0..1000u32 {
            let x = i as f32 / 1000.0;
            let floor = (x.clamp(0.0, 1.0) * max as f32).floor();
            for &u in &[0.0f32, 0.3, 0.7, 0.999] {
                let c = encode_stochastic(x, max, u);
                assert!(c <= max, "code {c} exceeds max");
                assert!(
                    c as f32 >= floor && c as f32 <= floor + 1.0,
                    "code {c} not adjacent to {floor}"
                );
            }
        }
    }

    #[test]
    fn encode_stochastic_is_unbiased() {
        // A sub-quantum-resolution value (~100.3 codes) averaged over many draws
        // recovers the true scaled value within Monte-Carlo error.
        let max = u16::MAX as u32;
        let x = 0.001_530_5_f32;
        let scaled = (x.clamp(0.0, 1.0) * max as f32) as f64;
        let n = 200_000u32;
        let mut sum = 0u64;
        for t in 0..n {
            let u = u01(7, 2, t as usize);
            sum += u64::from(encode_stochastic(x, max, u));
        }
        let mean = sum as f64 / f64::from(n);
        assert!(
            (mean - scaled).abs() < 0.05,
            "biased: mean {mean} vs true {scaled}"
        );
    }
}
