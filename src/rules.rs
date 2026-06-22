//! The concrete regret-update rules.
//!
//! Each type here is a zero-sized marker implementing [`UpdateRule`]; the
//! algorithm lives entirely in its method bodies, which reproduce the exact
//! floating-point arithmetic of the matching scalar matcher so a batched
//! matcher at batch size 1 agrees bit-for-bit. The five rules span the shape
//! space the abstraction must cover: signed vs. floored regret, weighted vs.
//! discounted accumulation, and non-predictive (2 lanes) vs. predictive (3).
//!
//! Two recurring ideas, stated once:
//!
//! - **Discounting old regret.** Regret accumulated under early, poorly-informed
//!   strategies is downweighted so the running totals track improving play. The
//!   factor `d(t, e) = tᵉ / (tᵉ + 1)` rises toward 1 with `t`, so recent regret
//!   is discounted less than old regret. (Brown & Sandholm 2019,
//!   *Solving Imperfect-Information Games via Discounted Regret Minimization*,
//!   arXiv:1809.04040.)
//! - **Prediction.** If play changes smoothly, the next instantaneous regret
//!   resembles the last one, so a predictive rule forms its strategy from the
//!   cumulative regret *plus* that most-recent regret — reacting one step early.
//!   The prediction is transient: it shapes the strategy but is never stored
//!   into cumulative regret. (Farina, Kroer & Sandholm 2021, arXiv:2007.14358;
//!   Xu et al. 2024, *Minimizing Weighted Counterfactual Regret with Optimistic
//!   Online Mirror Descent*, arXiv:2404.13891.)

use crate::discount::DiscountParams;
use crate::probability::normalize_inplace;
use crate::regret_minimizer::regret_match;
use crate::update_rule::UpdateRule;

/// Discount parameters for the `+`/predictive discounted rules: `alpha`
/// discounts cumulative regret, `gamma` discounts the average-strategy
/// contribution. (Unlike full DCFR there is no separate negative-regret
/// exponent — flooring at zero removes negative regret outright.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlusDiscount {
    /// Regret-discount exponent.
    pub alpha: f32,
    /// Average-strategy-discount exponent.
    pub gamma: f32,
}

/// `((t-1)/t)^gamma`, the average-strategy discount the `+` rules apply once
/// they have a previous iterate. Returns `0.0` at `t == 1`, where there is no
/// prior strategy to carry forward (matching the scalar matchers).
fn plus_strategy_discount(t: usize, gamma: f32) -> f32 {
    if t > 1 {
        ((t - 1) as f32 / t as f32).powf(gamma)
    } else {
        0.0
    }
}

/// Normalize `[regret · regret_discount + last_inst]^+` into `out`. With
/// `regret_discount = 1` and zero prediction this reduces to plain regret
/// matching, so it serves predictive and non-predictive derivation alike.
fn predicted_strategy(regret: &[f32], last_inst: &[f32], regret_discount: f32, out: &mut [f32]) {
    for ((o, &r), &m) in out.iter_mut().zip(regret).zip(last_inst) {
        *o = (r * regret_discount + m).max(0.0);
    }
    normalize_inplace(out);
}

// ── DCFR ─────────────────────────────────────────────────────────────────────

/// Discounted CFR. Signed cumulative regret, each iteration's old positive
/// regret scaled by `d(t, α)` and old negative regret by `d(t, β)` before the
/// new regret is added; the average strategy is discounted by `(t/(t+1))^γ`.
pub struct Dcfr;

/// Shared per-iteration factors for [`Dcfr`].
pub struct DcfrStep {
    positive: f32,
    negative: f32,
    strategy: f32,
}

impl UpdateRule for Dcfr {
    type Params = DiscountParams;
    type Step = DcfrStep;
    const LANES: usize = 2;

    fn step(p: &Self::Params, t: usize) -> Self::Step {
        DcfrStep {
            positive: DiscountParams::discount_factor(t, p.alpha),
            negative: DiscountParams::discount_factor(t, p.beta),
            strategy: (t as f32 / (t as f32 + 1.0)).powf(p.gamma),
        }
    }

    fn strategy_from_lanes(_: &Self::Params, regret: &[f32], _: &[f32], _: f32, out: &mut [f32]) {
        regret_match(regret, out);
    }

    fn pre_discount(_: &Self::Step) -> f32 {
        0.0
    }
    fn post_discount(_: &Self::Step) -> f32 {
        0.0
    }

    fn accumulate_regret(s: &Self::Step, old: f32, reward: f32, expected: f32) -> f32 {
        let d = if old > 0.0 { s.positive } else { s.negative };
        old * d + (reward - expected)
    }

    fn strategy_accumulation(s: &Self::Step) -> (f32, f32) {
        (s.strategy, 1.0)
    }

    fn regret_weight_step(s: &Self::Step, old_w: f32) -> f32 {
        old_w * s.positive + 1.0
    }
    fn regret_weight_total(_: &Self::Params, _t: usize, accum_w: f32) -> f32 {
        accum_w
    }
}

// ── DCFR+ ─────────────────────────────────────────────────────────────────────

/// Discounted CFR+. Like DCFR but regret is floored at zero *after* the discount
/// and add, so a recovering action need not first pay back accumulated negative
/// regret. A single discount `d(t-1, α)` applies (no negative branch, since
/// there is no negative regret to treat).
pub struct DcfrPlus;

/// Shared per-iteration factors for [`DcfrPlus`].
pub struct PlusStep {
    regret: f32,
    strategy: f32,
}

impl DcfrPlus {
    /// Grid-searched defaults from the source paper: `α = 1.5`, `γ = 4`.
    pub const RECOMMENDED: PlusDiscount = PlusDiscount {
        alpha: 1.5,
        gamma: 4.0,
    };
}

impl UpdateRule for DcfrPlus {
    type Params = PlusDiscount;
    type Step = PlusStep;
    const LANES: usize = 2;

    fn step(p: &Self::Params, t: usize) -> Self::Step {
        PlusStep {
            // The accumulator being discounted is the *previous* iterate, so its
            // index is t-1; nothing to discount on the first iteration.
            regret: if t > 1 {
                DiscountParams::discount_factor(t - 1, p.alpha)
            } else {
                0.0
            },
            strategy: plus_strategy_discount(t, p.gamma),
        }
    }

    fn strategy_from_lanes(_: &Self::Params, regret: &[f32], _: &[f32], _: f32, out: &mut [f32]) {
        regret_match(regret, out);
    }

    fn pre_discount(_: &Self::Step) -> f32 {
        0.0
    }
    fn post_discount(_: &Self::Step) -> f32 {
        0.0
    }

    fn accumulate_regret(s: &Self::Step, old: f32, reward: f32, expected: f32) -> f32 {
        // Left-associative `old*d + reward - expected`, mirroring dcfr_plus.rs.
        (old * s.regret + reward - expected).max(0.0)
    }

    fn strategy_accumulation(s: &Self::Step) -> (f32, f32) {
        (s.strategy, 1.0)
    }

    fn regret_weight_step(s: &Self::Step, old_w: f32) -> f32 {
        old_w * s.regret + 1.0
    }
    fn regret_weight_total(_: &Self::Params, _t: usize, accum_w: f32) -> f32 {
        accum_w
    }
}

// ── Linear CFR ────────────────────────────────────────────────────────────────

/// Linear CFR. The cheapest discounting: weight iteration `t`'s regret and
/// strategy contribution by `t`, so early iterates fade linearly. Equivalent to
/// DCFR with α=β=γ=1, but expressed (like the scalar) in the increasing-weight
/// form, so the stored totals grow rather than staying bounded.
pub struct LinearCfr;

/// Shared per-iteration factor for [`LinearCfr`]: the iteration weight `t`.
pub struct LinearStep {
    t: f32,
}

impl UpdateRule for LinearCfr {
    type Params = ();
    type Step = LinearStep;
    const LANES: usize = 2;

    fn step(_: &Self::Params, t: usize) -> Self::Step {
        LinearStep { t: t as f32 }
    }

    fn strategy_from_lanes(_: &Self::Params, regret: &[f32], _: &[f32], _: f32, out: &mut [f32]) {
        regret_match(regret, out);
    }

    fn pre_discount(_: &Self::Step) -> f32 {
        0.0
    }
    fn post_discount(_: &Self::Step) -> f32 {
        0.0
    }

    fn accumulate_regret(s: &Self::Step, old: f32, reward: f32, expected: f32) -> f32 {
        old + s.t * (reward - expected)
    }

    fn strategy_accumulation(s: &Self::Step) -> (f32, f32) {
        (1.0, s.t)
    }

    fn regret_weight_step(_: &Self::Step, old_w: f32) -> f32 {
        old_w // unused; the total is a closed form of t
    }
    fn regret_weight_total(_: &Self::Params, t: usize, _accum_w: f32) -> f32 {
        // Σ_{i=1}^{t} i = t(t+1)/2, the total weight applied to regret.
        let t = t as f32;
        t * (t + 1.0) / 2.0
    }
}

// ── PCFR+ ─────────────────────────────────────────────────────────────────────

/// Predictive CFR+. CFR+'s floored regret, but the strategy is formed from the
/// cumulative regret plus the most-recent instantaneous regret (the
/// prediction), and the average strategy uses quadratic (`t²`) weighting.
pub struct PcfrPlus;

/// Shared per-iteration factor for [`PcfrPlus`]: the quadratic averaging weight
/// `t²`.
pub struct PcfrPlusStep {
    quadratic: f32,
}

impl UpdateRule for PcfrPlus {
    type Params = ();
    type Step = PcfrPlusStep;
    const LANES: usize = 3;

    fn step(_: &Self::Params, t: usize) -> Self::Step {
        PcfrPlusStep {
            quadratic: (t * t) as f32,
        }
    }

    fn strategy_from_lanes(
        _: &Self::Params,
        regret: &[f32],
        last: &[f32],
        d: f32,
        out: &mut [f32],
    ) {
        predicted_strategy(regret, last, d, out);
    }

    fn pre_discount(_: &Self::Step) -> f32 {
        1.0 // undiscounted prediction: regret enters the strategy at full weight
    }
    fn post_discount(_: &Self::Step) -> f32 {
        1.0
    }

    fn accumulate_regret(_: &Self::Step, old: f32, reward: f32, expected: f32) -> f32 {
        (old + (reward - expected)).max(0.0)
    }

    fn strategy_accumulation(s: &Self::Step) -> (f32, f32) {
        (1.0, s.quadratic)
    }

    fn regret_weight_step(_: &Self::Step, old_w: f32) -> f32 {
        old_w // unused; total is T
    }
    fn regret_weight_total(_: &Self::Params, t: usize, _accum_w: f32) -> f32 {
        t as f32
    }
}

// ── PDCFR+ ────────────────────────────────────────────────────────────────────

/// Predictive Discounted CFR+. Stores regret like DCFR+ (discounted, floored)
/// but forms the strategy like PCFR+ (regret plus prediction) — additionally
/// discounting the regret term by `d(t, α)` inside the prediction. The widest
/// shape: predictive (3 lanes) *and* discounted.
pub struct PdcfrPlus;

/// Shared per-iteration factors for [`PdcfrPlus`].
pub struct PdcfrPlusStep {
    /// `d(t-1, α)` — discount on the previous accumulator and the pre-update
    /// strategy's regret term.
    previous: f32,
    /// `d(t, α)` — discount on the post-update strategy's regret term.
    current: f32,
    strategy: f32,
}

impl PdcfrPlus {
    /// Grid-searched defaults from the source paper: `α = 2.3`, `γ = 5`.
    pub const RECOMMENDED: PlusDiscount = PlusDiscount {
        alpha: 2.3,
        gamma: 5.0,
    };
}

impl UpdateRule for PdcfrPlus {
    type Params = PlusDiscount;
    type Step = PdcfrPlusStep;
    const LANES: usize = 3;

    fn step(p: &Self::Params, t: usize) -> Self::Step {
        PdcfrPlusStep {
            previous: if t > 1 {
                DiscountParams::discount_factor(t - 1, p.alpha)
            } else {
                0.0
            },
            current: DiscountParams::discount_factor(t, p.alpha),
            strategy: plus_strategy_discount(t, p.gamma),
        }
    }

    fn strategy_from_lanes(
        _: &Self::Params,
        regret: &[f32],
        last: &[f32],
        d: f32,
        out: &mut [f32],
    ) {
        predicted_strategy(regret, last, d, out);
    }

    fn pre_discount(s: &Self::Step) -> f32 {
        s.previous
    }
    fn post_discount(s: &Self::Step) -> f32 {
        s.current
    }

    fn accumulate_regret(s: &Self::Step, old: f32, reward: f32, expected: f32) -> f32 {
        // Parenthesized inst, mirroring pdcfr_plus.rs.
        (old * s.previous + (reward - expected)).max(0.0)
    }

    fn strategy_accumulation(s: &Self::Step) -> (f32, f32) {
        (s.strategy, 1.0)
    }

    fn regret_weight_step(s: &Self::Step, old_w: f32) -> f32 {
        old_w * s.previous + 1.0
    }
    fn regret_weight_total(_: &Self::Params, _t: usize, accum_w: f32) -> f32 {
        accum_w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discount::DiscountParams;
    use crate::update_rule::UpdateRule;

    // ── DCFR (signed, non-predictive) ───────────────────────────────────────

    #[test]
    fn dcfr_mirrors_scalar() {
        let p = DiscountParams::RECOMMENDED;
        let s = Dcfr::step(&p, 3);
        let pos = DiscountParams::discount_factor(3, p.alpha);
        let neg = DiscountParams::discount_factor(3, p.beta);
        let strat = (3.0f32 / 4.0).powf(p.gamma);

        // Sign of the OLD regret picks the discount; instantaneous regret is
        // added parenthesized, exactly as dcfr.rs writes it.
        assert_eq!(
            Dcfr::accumulate_regret(&s, 2.0, 5.0, 1.0),
            2.0 * pos + (5.0 - 1.0)
        );
        assert_eq!(
            Dcfr::accumulate_regret(&s, -2.0, 5.0, 1.0),
            -2.0 * neg + (5.0 - 1.0)
        );
        assert_eq!(Dcfr::strategy_accumulation(&s), (strat, 1.0));
        assert_eq!(Dcfr::regret_weight_step(&s, 4.0), 4.0 * pos + 1.0);
        assert_eq!(Dcfr::regret_weight_total(&p, 7, 9.5), 9.5);
        assert_eq!(Dcfr::pre_discount(&s), 0.0);
        assert_eq!(Dcfr::LANES, 2);
    }

    // ── DCFR+ (floored, non-predictive) ─────────────────────────────────────

    #[test]
    fn dcfr_plus_mirrors_scalar() {
        let p = DcfrPlus::RECOMMENDED;
        let s = DcfrPlus::step(&p, 3);
        let prev = DiscountParams::discount_factor(2, p.alpha);
        let strat = (2.0f32 / 3.0).powf(p.gamma);

        // Left-associative `old*prev + rw - exp`, floored — exactly dcfr_plus.rs.
        assert_eq!(
            DcfrPlus::accumulate_regret(&s, 2.0, 5.0, 1.0),
            (2.0 * prev + 5.0 - 1.0).max(0.0)
        );
        assert_eq!(DcfrPlus::accumulate_regret(&s, -10.0, 0.0, 1.0), 0.0); // floored
        assert_eq!(DcfrPlus::strategy_accumulation(&s), (strat, 1.0));
        assert_eq!(DcfrPlus::regret_weight_step(&s, 4.0), 4.0 * prev + 1.0);
        assert_eq!(DcfrPlus::pre_discount(&s), 0.0); // non-predictive
        assert_eq!(DcfrPlus::LANES, 2);
    }

    #[test]
    fn dcfr_plus_first_iteration_has_no_history() {
        let s = DcfrPlus::step(&DcfrPlus::RECOMMENDED, 1);
        // t == 1: nothing to discount yet, so old regret is dropped entirely.
        assert_eq!(
            DcfrPlus::accumulate_regret(&s, 9.0, 2.0, 0.5),
            (2.0f32 - 0.5).max(0.0)
        );
        assert_eq!(DcfrPlus::strategy_accumulation(&s).0, 0.0);
    }

    // ── Linear CFR (weighted, non-predictive) ───────────────────────────────

    #[test]
    fn linear_cfr_mirrors_scalar() {
        let s = LinearCfr::step(&(), 4);
        // R += t * (rw - exp); X += t * x; weight total = t(t+1)/2.
        assert_eq!(
            LinearCfr::accumulate_regret(&s, 3.0, 5.0, 1.0),
            3.0 + 4.0 * (5.0 - 1.0)
        );
        assert_eq!(LinearCfr::strategy_accumulation(&s), (1.0, 4.0));
        assert_eq!(LinearCfr::regret_weight_total(&(), 4, 0.0), 4.0 * 5.0 / 2.0);
        assert_eq!(LinearCfr::pre_discount(&s), 0.0);
        assert_eq!(LinearCfr::LANES, 2);
    }

    // ── PCFR+ (predictive, no discount) ─────────────────────────────────────

    #[test]
    fn pcfr_plus_mirrors_scalar() {
        let s = PcfrPlus::step(&(), 3);
        // Floored CFR+ regret; quadratic averaging weight t².
        assert_eq!(
            PcfrPlus::accumulate_regret(&s, 1.0, 4.0, 1.0),
            (1.0f32 + (4.0 - 1.0)).max(0.0)
        );
        assert_eq!(PcfrPlus::accumulate_regret(&s, 0.0, 0.0, 5.0), 0.0); // floored
        assert_eq!(PcfrPlus::strategy_accumulation(&s), (1.0, 9.0)); // t² = 9
        assert_eq!(PcfrPlus::regret_weight_total(&(), 3, 0.0), 3.0); // T
        // Predictive but undiscounted: regret enters the strategy at weight 1.
        assert_eq!(PcfrPlus::pre_discount(&s), 1.0);
        assert_eq!(PcfrPlus::post_discount(&s), 1.0);
        assert_eq!(PcfrPlus::LANES, 3);

        // strategy = normalize([regret + last_inst]^+)
        let mut out = [0.0f32; 2];
        PcfrPlus::strategy_from_lanes(&(), &[1.0, 0.0], &[0.0, 1.0], 1.0, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6 && (out[1] - 0.5).abs() < 1e-6);
    }

    // ── PDCFR+ (predictive, discounted; 3 lanes) ────────────────────────────

    #[test]
    fn pdcfr_plus_mirrors_scalar() {
        let p = PdcfrPlus::RECOMMENDED; // alpha 2.3, gamma 5
        let s = PdcfrPlus::step(&p, 3);
        let prev = DiscountParams::discount_factor(2, p.alpha);
        let curr = DiscountParams::discount_factor(3, p.alpha);
        let strat = (2.0f32 / 3.0).powf(p.gamma);

        // Stored regret: DCFR+-style with parenthesized inst, floored.
        assert_eq!(
            PdcfrPlus::accumulate_regret(&s, 2.0, 5.0, 1.0),
            (2.0 * prev + (5.0 - 1.0)).max(0.0)
        );
        assert_eq!(PdcfrPlus::strategy_accumulation(&s), (strat, 1.0));
        assert_eq!(PdcfrPlus::regret_weight_step(&s, 4.0), 4.0 * prev + 1.0);
        // Pre-update strategy uses last iteration's discount d(t-1); post uses d(t).
        assert_eq!(PdcfrPlus::pre_discount(&s), prev);
        assert_eq!(PdcfrPlus::post_discount(&s), curr);
        assert_eq!(PdcfrPlus::LANES, 3);

        // strategy = normalize([regret*disc + last_inst]^+)
        let mut out = [0.0f32; 2];
        PdcfrPlus::strategy_from_lanes(&p, &[1.0, 1.0], &[0.0, 0.0], curr, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6 && (out[1] - 0.5).abs() < 1e-6);
    }
}
