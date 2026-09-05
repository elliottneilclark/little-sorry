//! The algorithm abstraction shared by the batched matcher.
//!
//! Every regret-matching variant keeps, per information set, a few `f32` arrays
//! of length `num_actions` — call each a *lane*. Lane 0 is the cumulative
//! regret (how much, in hindsight, we wish we had played each action more);
//! lane 1 is the cumulative strategy (the running sum of strategies actually
//! played, whose normalization is the equilibrium-approximating average). A
//! predictive variant adds a third lane: the most recent one-step regret,
//! reused as a cheap prediction of the next one. The *number* of lanes is the
//! only algorithm-specific fact the storage and batching layers ever need; this
//! trait carries it as [`UpdateRule::LANES`].
//!
//! The trait separates work that depends only on the iteration count from work
//! that depends on the rewards. Time-dependent factors — discount exponents
//! raised to powers, ratios — are computed once per batch into a
//! [`UpdateRule::Step`] and shared across every row, so a decision point owning
//! hundreds of information sets pays for each transcendental once, not hundreds
//! of times. Even the strategy derivation, which a predictive rule discounts by
//! such a factor, receives it precomputed (see [`UpdateRule::pre_discount`] /
//! [`UpdateRule::post_discount`]) rather than recomputing a `powf` per row. The
//! reward-dependent arithmetic is expressed per cell so it runs unchanged over
//! either storage backend.
//!
//! Sources for the per-rule arithmetic are cited on each implementation in
//! `crate::rules`.

/// An abstract regret-update rule.
///
/// Implementors supply only the algorithm-specific arithmetic; storage,
/// batching, readout, and serialization are written against this trait and make
/// no assumption about which algorithm is in use.
pub trait UpdateRule {
    /// Per-instance parameters (e.g. discount exponents). Held per matcher, never
    /// global, so two matchers of the same rule with different parameters
    /// coexist.
    type Params: Clone;

    /// Constants derived once per iteration from `(params, t)` and shared across
    /// all rows of a batch. This is where the expensive `powf`/division results
    /// live so they are computed once, not once per row.
    type Step;

    /// Number of per-row state lanes: 2 for non-predictive rules (regret,
    /// cumulative strategy), 3 for predictive ones (plus last-instantaneous
    /// regret in lane 2).
    const LANES: usize;

    /// Compute the shared per-iteration constants for iteration `t` (1-based).
    fn step(params: &Self::Params, t: usize) -> Self::Step;

    /// Write the strategy a row plays given its stored lanes, normalized to a
    /// distribution. `regret_discount` is the precomputed factor a predictive
    /// rule applies to its regret before adding the prediction in `last_inst`;
    /// non-predictive rules ignore both. Re-deriving the strategy from the lanes
    /// — rather than storing it — lets the matcher drop a whole per-row array at
    /// no precision cost, and is how a batched matcher reproduces a scalar
    /// matcher's carried strategy exactly.
    fn strategy_from_lanes(
        params: &Self::Params,
        regret: &[f32],
        last_inst: &[f32],
        regret_discount: f32,
        out: &mut [f32],
    );

    /// Regret discount for deriving the *pre-update* strategy of an iteration
    /// (the strategy carried in from the previous step). `0.0` for
    /// non-predictive rules, which ignore it.
    fn pre_discount(step: &Self::Step) -> f32;

    /// Regret discount for deriving the *post-update* strategy of an iteration
    /// (the strategy this step will play and accumulate). `0.0` for
    /// non-predictive rules.
    fn post_discount(step: &Self::Step) -> f32;

    /// New cumulative regret for one action from its old value, this action's
    /// `reward`, and the strategy's `expected` value. The instantaneous regret
    /// is `reward − expected`; rules take the two terms separately rather than
    /// pre-combined so each can reproduce its scalar counterpart's exact
    /// floating-point expression (the associativity of `… + reward − expected`
    /// is load-bearing for bit-for-bit equivalence).
    fn accumulate_regret(step: &Self::Step, old_regret: f32, reward: f32, expected: f32) -> f32;

    /// Whether [`accumulate_regret`](Self::accumulate_regret) clamps its result
    /// at zero (the `+` family). An integer-accumulating regret lane applies the
    /// same clamp in code space after folding the discount and increment.
    const FLOORS_REGRET: bool;

    /// The factor [`accumulate_regret`](Self::accumulate_regret) multiplies
    /// `old_regret` by this iteration — sign-dependent for DCFR, `1.0` for the
    /// undiscounted rules. Together with
    /// [`regret_increment`](Self::regret_increment) this is the update split
    /// into the two terms an integer lane needs separately: the accumulator is
    /// discounted in its own (wide) precision and the small increment is added
    /// to it, so a sub-f32-ulp increment at large magnitude is not rounded away
    /// before the lane ever sees it. Every rule satisfies
    /// `accumulate_regret(s, o, r, e) ≈ [o · regret_discount(s, o) +
    /// regret_increment(s, r, e)]⁺` to within f32 rounding; the f32 method
    /// remains the bit-exact form the scalar matchers are compared against.
    fn regret_discount(step: &Self::Step, old_regret: f32) -> f32;

    /// The term [`accumulate_regret`](Self::accumulate_regret) adds to the
    /// discounted old regret: `reward − expected` scaled by the rule's
    /// iteration weight (see [`regret_discount`](Self::regret_discount)).
    fn regret_increment(step: &Self::Step, reward: f32, expected: f32) -> f32;

    /// How the cumulative-strategy lane advances: `X ← X · discount + weight · x`
    /// for the returned `(discount, weight)`. This single shape covers
    /// discounted accumulation (`(d, 1)`) and increasing-weight accumulation
    /// (`(1, t)` or `(1, t²)`) alike.
    fn strategy_accumulation(step: &Self::Step) -> (f32, f32);

    /// Advance the shared regret-weight accumulator. Discounted rules mirror the
    /// regret discount here (`W ← W · d + 1`); closed-form rules may treat it as
    /// a plain counter, since [`UpdateRule::regret_weight_total`] ignores it.
    fn regret_weight_step(step: &Self::Step, old_w: f32) -> f32;

    /// The total regret weight after `t` updates — the denominator that turns
    /// accumulated regret into a true time average for the convergence
    /// diagnostic. Discounted rules return the accumulated `accum_w`;
    /// closed-form rules return a function of `t` (e.g. `t(t+1)/2`).
    fn regret_weight_total(params: &Self::Params, t: usize, accum_w: f32) -> f32;
}

/// Simulate one action's cumulative regret under `R` when its instantaneous
/// regret is `−loss` on every iteration `1..=steps` — a dominated action that
/// always earns `loss` less than the strategy's expected value. Returns the
/// final cumulative regret.
///
/// Callers derive a *pruning floor* from this rather than reusing a constant
/// from another solver: e.g. set an [`Int32Config`](crate::lane::Int32Config)
/// floor to `k ×` this value at the warm-up length, so a dominated action is
/// prunable after warm-up and an action pruned early can still un-prune. The
/// shape depends on the rule, which is why the floor must be derived per rule:
/// under DCFR with `β < 1` negative regret is discounted every tick and the
/// trajectory approaches a finite fixed point, while under linear CFR it grows
/// without bound (see `dominated_regret_after_matches_scalar_loop` in this
/// module's tests, which pins both numerically). For the `+` family, whose
/// regret is floored at zero, the result is always `0.0`.
///
/// ```
/// use little_sorry::{Dcfr, DiscountParams, LinearCfr, dominated_regret_after};
///
/// let dcfr = dominated_regret_after::<Dcfr>(&DiscountParams::RECOMMENDED, 200, 1.0);
/// let linear = dominated_regret_after::<LinearCfr>(&(), 200, 1.0);
/// assert!(dcfr < 0.0 && dcfr > -3.0, "DCFR settles near a fixed point: {dcfr}");
/// assert!(linear < -10_000.0, "linear CFR keeps growing: {linear}");
/// ```
pub fn dominated_regret_after<R: UpdateRule>(params: &R::Params, steps: usize, loss: f32) -> f32 {
    let mut regret = 0.0f32;
    for t in 1..=steps {
        let step = R::step(params, t);
        // reward 0, expected `loss` ⇒ instantaneous regret −loss.
        regret = R::accumulate_regret(&step, regret, 0.0, loss);
    }
    regret
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal non-predictive rule used only to exercise the trait surface.
    struct MockRule;
    impl UpdateRule for MockRule {
        type Params = ();
        type Step = ();
        const LANES: usize = 2;
        fn step(_: &Self::Params, _t: usize) -> Self::Step {}
        fn strategy_from_lanes(
            _: &Self::Params,
            regret: &[f32],
            _last_inst: &[f32],
            _regret_discount: f32,
            out: &mut [f32],
        ) {
            crate::regret_minimizer::regret_match(regret, out);
        }
        fn pre_discount(_: &Self::Step) -> f32 {
            0.0
        }
        fn post_discount(_: &Self::Step) -> f32 {
            0.0
        }
        fn accumulate_regret(_: &Self::Step, old_r: f32, reward: f32, expected: f32) -> f32 {
            old_r + (reward - expected)
        }
        const FLOORS_REGRET: bool = false;
        fn regret_discount(_: &Self::Step, _old_r: f32) -> f32 {
            1.0
        }
        fn regret_increment(_: &Self::Step, reward: f32, expected: f32) -> f32 {
            reward - expected
        }
        fn strategy_accumulation(_: &Self::Step) -> (f32, f32) {
            (1.0, 1.0)
        }
        fn regret_weight_step(_: &Self::Step, old_w: f32) -> f32 {
            old_w + 1.0
        }
        fn regret_weight_total(_: &Self::Params, _t: usize, accum_w: f32) -> f32 {
            accum_w
        }
    }

    #[test]
    fn strategy_from_zero_regret_is_uniform() {
        let mut out = [0.0f32; 3];
        MockRule::strategy_from_lanes(&(), &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 0.0, &mut out);
        assert!(out.iter().all(|&v| (v - 1.0 / 3.0).abs() < 1e-6));
    }

    #[test]
    fn non_predictive_is_two_lane() {
        assert_eq!(MockRule::LANES, 2);
    }

    #[test]
    fn dominated_regret_after_matches_scalar_loop() {
        use crate::discount::DiscountParams;
        use crate::rules::{Dcfr, DcfrPlus, LinearCfr};

        // DCFR: hand-written recurrence with the sign-dependent discount.
        for params in [
            DiscountParams::RECOMMENDED,
            DiscountParams::new(3.0, 0.0, 20.0),
        ] {
            let mut r = 0.0f32;
            for t in 1..=500usize {
                let d = if r > 0.0 {
                    DiscountParams::discount_factor(t, params.alpha)
                } else {
                    DiscountParams::discount_factor(t, params.beta)
                };
                r = r * d + (0.0 - 1.0);
            }
            let got = dominated_regret_after::<Dcfr>(&params, 500, 1.0);
            assert_eq!(got.to_bits(), r.to_bits(), "DCFR {params:?}: {got} vs {r}");
            // β = 0 ⇒ negative discount is the constant 1/2, so the trajectory
            // r ← r/2 − loss converges to the fixed point −2·loss.
            assert!((got + 2.0).abs() < 1e-4, "DCFR fixed point: {got}");
            let longer = dominated_regret_after::<Dcfr>(&params, 5_000, 1.0);
            assert!(
                (longer - got).abs() < 1e-4,
                "DCFR trajectory must be bounded: {got} → {longer}"
            );
        }

        // Linear CFR: r ← r − t·loss, i.e. −loss·t(t+1)/2 — unbounded.
        let mut r = 0.0f32;
        for t in 1..=200usize {
            r += t as f32 * (0.0 - 1.0);
        }
        let got = dominated_regret_after::<LinearCfr>(&(), 200, 1.0);
        assert_eq!(got.to_bits(), r.to_bits(), "linear: {got} vs {r}");
        assert!(
            (got + 200.0 * 201.0 / 2.0).abs() < 1.0,
            "linear closed form: {got}"
        );
        assert!(
            dominated_regret_after::<LinearCfr>(&(), 400, 1.0) < 3.0 * got,
            "linear CFR grows super-linearly"
        );

        // Floored rules never go negative.
        assert_eq!(
            dominated_regret_after::<DcfrPlus>(&DcfrPlus::RECOMMENDED, 200, 1.0),
            0.0
        );
    }
}
