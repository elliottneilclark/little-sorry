//! Signed per-row-scaled int16 regret codec. A whole row shares one f32 scale,
//! so 15-bit precision tracks the row's largest-magnitude regret — exactly what
//! regret-matching is most sensitive to. Solve state only; never exported.

const MAX_ABS: f32 = i16::MAX as f32; // 32767

/// Smallest positive scale such that every `round(r/scale)` fits in i16.
pub(crate) fn choose_scale(regret: &[f32]) -> f32 {
    let peak = regret.iter().fold(0.0f32, |m, &r| m.max(r.abs()));
    // Floor keeps scale finite/positive for an all-zero row.
    (peak / MAX_ABS).max(f32::MIN_POSITIVE)
}

pub(crate) fn encode(r: f32, scale: f32) -> i16 {
    let q = (r / scale).round();
    // Safety: the explicit >= MAX_ABS and <= i16::MIN bounds guarantee the
    // value is in range before casting, so no truncation can occur.
    #[allow(clippy::cast_possible_truncation)]
    if q >= MAX_ABS {
        i16::MAX
    } else if q <= i16::MIN as f32 {
        i16::MIN
    } else {
        q as i16
    }
}

pub(crate) fn decode(q: i16, scale: f32) -> f32 {
    q as f32 * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scale_keeps_codes_in_range_and_round_trips() {
        let regret = [0.0f32, 1234.5, -9999.0, 42.0];
        let s = choose_scale(&regret);
        for &r in &regret {
            let q = encode(r, s);
            let back = decode(q, s);
            // within one scale-quantum
            assert!((r - back).abs() <= s + 1e-3, "{r} vs {back} (s={s})");
        }
    }
    #[test]
    fn overflow_saturates_not_wraps() {
        let s = 1.0;
        assert_eq!(encode(1e9, s), i16::MAX);
        assert_eq!(encode(-1e9, s), i16::MIN);
    }
    #[test]
    fn saturation_boundaries() {
        let s = 1.0;
        assert_eq!(encode(32767.0, s), i16::MAX); // exactly MAX_ABS -> saturates
        assert_eq!(encode(32767.5, s), i16::MAX); // rounds to 32768 -> saturates
        assert_eq!(encode(32766.5, s), i16::MAX); // rounds to 32767 (== MAX_ABS) -> saturates
        assert_eq!(encode(32765.0, s), 32765i16); // in-range, not saturated
        assert_eq!(encode(-32768.0, s), i16::MIN); // exactly i16::MIN -> saturates
        assert_eq!(encode(-32767.0, s), -32767i16); // in-range, not saturated
    }
    #[test]
    fn all_zero_regret_has_finite_positive_scale() {
        let s = choose_scale(&[0.0, 0.0, 0.0]);
        assert!(s > 0.0 && s.is_finite());
    }
}
