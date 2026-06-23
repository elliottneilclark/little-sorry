//! Fixed-point export codec for normalized strategies.
//!
//! A solved average strategy is a probability distribution: each entry sits in
//! `[0, 1]` and the row sums to one. Shipping it as 32-bit floats wastes space
//! when a read-only runtime only needs a few decimal digits of accuracy. A
//! *fixed-point* code maps the unit interval onto the evenly spaced integers
//! `0..=MAX`: a value `x` becomes the nearest integer `round(x · MAX)`, and a
//! code `c` decodes back to `c / MAX`. With `MAX = 2^b − 1` both endpoints stay
//! exact (`0 → 0`, `1 → MAX`), and the spacing between representable values —
//! the *quantum* — is `1 / MAX`. For 16-bit codes that quantum is
//! `1/65535 ≈ 1.526e-5`; round-to-nearest keeps each component within half a
//! quantum (`≈ 7.63e-6`) of its original value. (Standard uniform-quantizer
//! result; see e.g. the fixed-point quantization literature.)
//!
//! Rounding each component independently means the decoded values no longer sum
//! to exactly one, so [`dequantize_dist`] **renormalizes** — rescaling the row
//! back onto the simplex is the correct, proportion-preserving repair, and
//! baking it into the decoder keeps every caller honest.
//!
//! This codec is for **strategy export only** — a lossy snapshot for a runtime
//! that just reads the policy. It is *not* a solve checkpoint: quantizing raw
//! regret/strategy accumulators would destroy the precision and reproducibility
//! a resumed solve depends on, and a checkpoint would additionally need to
//! record the algorithm and its per-row lane count. Because the codec consumes
//! a finished distribution rather than algorithm-specific state, it is correct
//! for every matcher by construction.

/// A fixed-width unsigned code type for the unit-interval quantizer.
///
/// `MAX_CODE = 2^bits − 1` is the full-scale code that `1.0` maps to, so the
/// quantum is `1 / MAX_CODE` and both `0.0` and `1.0` stay exactly
/// representable.
pub trait FixedWidth: Copy {
    /// Largest code, representing `1.0` (i.e. `2^bits − 1`).
    const MAX_CODE: u32;
    /// Build the code type from a `u32` in `0..=MAX_CODE`.
    fn from_code(code: u32) -> Self;
    /// The numeric value of this code in `0..=MAX_CODE`.
    fn code(self) -> u32;
}

impl FixedWidth for u8 {
    const MAX_CODE: u32 = u8::MAX as u32;
    fn from_code(code: u32) -> Self {
        code as u8
    }
    fn code(self) -> u32 {
        u32::from(self)
    }
}

impl FixedWidth for u16 {
    const MAX_CODE: u32 = u16::MAX as u32;
    fn from_code(code: u32) -> Self {
        code as u16
    }
    fn code(self) -> u32 {
        u32::from(self)
    }
}

impl FixedWidth for u32 {
    const MAX_CODE: u32 = u32::MAX;
    fn from_code(code: u32) -> Self {
        code
    }
    fn code(self) -> u32 {
        self
    }
}

/// Encode a probability distribution to fixed-width codes.
///
/// Inputs are clamped to `[0, 1]` so an out-of-range component (e.g. a tiny
/// negative from float error) can never wrap the integer cast.
#[must_use]
pub fn quantize_dist<Q: FixedWidth>(probs: &[f32]) -> Vec<Q> {
    probs
        .iter()
        .map(|&x| Q::from_code(crate::unit_fixed::encode(x, Q::MAX_CODE)))
        .collect()
}

/// Decode fixed-width codes back to a probability distribution, renormalized to
/// sum to one. An all-zero (or empty) input falls back to the uniform
/// distribution, the only sensible distribution with no information.
#[must_use]
pub fn dequantize_dist<Q: FixedWidth>(codes: &[Q]) -> Vec<f32> {
    let mut out: Vec<f32> = codes
        .iter()
        .map(|&c| crate::unit_fixed::decode(c.code(), Q::MAX_CODE))
        .collect();
    crate::probability::normalize_inplace(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_roundtrip_within_one_quantum() {
        let probs = [0.1f32, 0.2, 0.3, 0.4];
        let back = dequantize_dist::<u16>(&quantize_dist::<u16>(&probs));
        for (a, b) in probs.iter().zip(&back) {
            assert!((a - b).abs() < 1.0 / 65535.0 + 1e-7, "{a} vs {b}");
        }
        assert!((back.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn one_hot_roundtrips() {
        let back = dequantize_dist::<u16>(&quantize_dist::<u16>(&[0.0f32, 1.0, 0.0]));
        assert!((back[1] - 1.0).abs() < 1e-6);
        assert!(back[0].abs() < 1e-6 && back[2].abs() < 1e-6);
    }

    #[test]
    fn all_zero_codes_decode_to_uniform() {
        let back = dequantize_dist::<u16>(&[0u16, 0, 0, 0]);
        assert!(back.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn u8_and_u32_widths_supported() {
        let probs = [0.25f32, 0.25, 0.5];
        let b8 = dequantize_dist::<u8>(&quantize_dist::<u8>(&probs));
        assert!((b8.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let b32 = dequantize_dist::<u32>(&quantize_dist::<u32>(&probs));
        for (a, b) in probs.iter().zip(&b32) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn out_of_range_inputs_are_clamped() {
        let codes = quantize_dist::<u16>(&[2.0, -1.0]);
        assert_eq!(codes[0], 65535);
        assert_eq!(codes[1], 0);
    }
}
