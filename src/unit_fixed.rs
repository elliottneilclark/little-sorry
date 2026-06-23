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
}
