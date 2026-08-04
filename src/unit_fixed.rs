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

/// Splitmix64-finalized hash of a `(row, update_count, chunk)` key — the same
/// key-mix the old per-cell hash used, with the 4-action chunk index in the
/// per-action slot. One output word carries four independent 16-bit draws.
#[inline]
fn row_bits(row: usize, update_count: usize, chunk: usize) -> u64 {
    let mut z = (row as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((chunk as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((update_count as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic, stateless per-row draw stream for stochastic rounding: one
/// [`row_bits`] hash yields draws for four actions (16 bits each — exactly
/// the u16 lane quantum), so a row of ≤ 4 actions costs one hash per tick
/// where a per-cell hash would cost one per action. Same key ⇒ same stream;
/// no shared mutable state, so it is reproducible and race-free under the
/// `Atomic` backend. `bits / 2^16` is exact in f32, so draws are unbiased on
/// the 2⁻¹⁶ grid.
pub(crate) struct RowDraws {
    row: usize,
    update_count: usize,
    bits: u64,
    action: usize,
}

impl RowDraws {
    #[inline]
    pub(crate) fn new(row: usize, update_count: usize) -> Self {
        Self {
            row,
            update_count,
            bits: row_bits(row, update_count, 0),
            action: 0,
        }
    }

    /// The draw for the next action of this row/tick, in `[0, 1)`.
    #[inline]
    pub(crate) fn next_u01(&mut self) -> f32 {
        let (chunk, slot) = (self.action / 4, self.action % 4);
        if self.action > 0 && slot == 0 {
            self.bits = row_bits(self.row, self.update_count, chunk);
        }
        self.action += 1;
        // Truncating to the slot's low 16 bits is the point of the shift.
        #[allow(clippy::cast_possible_truncation)]
        let bits16 = (self.bits >> (16 * slot)) as u16;
        f32::from(bits16) / 65_536.0
    }
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
    fn row_draws_deterministic_and_in_range() {
        for &(r, t) in &[(0usize, 1usize), (3, 999), (168, 67_000)] {
            let mut a = RowDraws::new(r, t);
            let mut b = RowDraws::new(r, t);
            for _ in 0..8 {
                // 8 draws crosses a chunk boundary (two hashes).
                let (x, y) = (a.next_u01(), b.next_u01());
                assert_eq!(x.to_bits(), y.to_bits(), "same key ⇒ same stream");
                assert!((0.0..1.0).contains(&x), "draw out of range: {x}");
            }
        }
        // Distinct keys generally differ (guards against a constant generator).
        assert_ne!(
            RowDraws::new(0, 1).next_u01(),
            RowDraws::new(0, 2).next_u01()
        );
        assert_ne!(
            RowDraws::new(0, 1).next_u01(),
            RowDraws::new(1, 1).next_u01()
        );
    }

    #[test]
    fn row_draws_are_unbiased_per_action() {
        // E[encode_stochastic] must recover a sub-quantum value for EVERY
        // action slot of the row hash, not just slot 0.
        let max = u16::MAX as u32;
        let x = 0.001_530_5_f32;
        let scaled = f64::from(x.clamp(0.0, 1.0) * max as f32);
        let n = 200_000u32;
        for action in 0..4usize {
            let mut sum = 0u64;
            for t in 0..n {
                let mut draws = RowDraws::new(7, t as usize);
                let mut u = draws.next_u01();
                for _ in 0..action {
                    u = draws.next_u01();
                }
                sum += u64::from(encode_stochastic(x, max, u));
            }
            let mean = sum as f64 / f64::from(n);
            assert!(
                (mean - scaled).abs() < 0.05,
                "action {action} biased: mean {mean} vs true {scaled}"
            );
        }
    }

    #[test]
    fn row_draws_are_independent_across_actions() {
        // Draws for different actions of one row come from disjoint 16-bit
        // slices; empirically the joint sub-median event must hit ~1/4.
        let n = 100_000usize;
        let mut below = [0u32; 4];
        let mut joint = [[0u32; 4]; 4];
        for t in 0..n {
            let mut draws = RowDraws::new(3, t);
            let u: Vec<f32> = (0..4).map(|_| draws.next_u01()).collect();
            for a in 0..4 {
                if u[a] < 0.5 {
                    below[a] += 1;
                }
                for b in (a + 1)..4 {
                    if u[a] < 0.5 && u[b] < 0.5 {
                        joint[a][b] += 1;
                    }
                }
            }
        }
        for a in 0..4 {
            let p = f64::from(below[a]) / n as f64;
            assert!((p - 0.5).abs() < 0.01, "action {a} marginal {p}");
            for b in (a + 1)..4 {
                let p = f64::from(joint[a][b]) / n as f64;
                assert!((p - 0.25).abs() < 0.01, "pair ({a},{b}) joint {p}");
            }
        }
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
            let u = RowDraws::new(7, t as usize).next_u01();
            sum += u64::from(encode_stochastic(x, max, u));
        }
        let mean = sum as f64 / f64::from(n);
        assert!(
            (mean - scaled).abs() < 0.05,
            "biased: mean {mean} vs true {scaled}"
        );
    }
}
