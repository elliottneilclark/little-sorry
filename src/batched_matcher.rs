//! A matcher owning many information sets over a pluggable cell backend.
//!
//! One instance holds `num_rows` information sets ("rows"), each over the same
//! `num_actions`, with cumulative regret and cumulative strategy held in
//! pluggable lane stores (a [`crate::lane::Layout`]) rather than a single flat
//! cell array. Predictive rules additionally own a matcher-side
//! last-instantaneous-regret lane (always f32). All rows share one
//! iteration clock: a single batched update advances the clock once and touches
//! every row, so the time-dependent factors a rule needs are computed once for
//! the whole batch rather than once per row — the win that makes a decision
//! point owning hundreds of rows cheap.
//!
//! Updates run entirely through `&self` via the backend's interior-mutable
//! cells, and touch each cell with an independent load/store, so the same code
//! drives the single-threaded [`crate::storage::Local`] backend and the
//! lock-free [`crate::storage::Atomic`] one. With `Local` the matcher is `!Sync`
//! and fully deterministic; with `Atomic` it is `Sync` and may be updated
//! concurrently, with the benign-race semantics documented on the backend.
//!
//! The current strategy is never stored — it is re-derived from the lanes on
//! demand. That drops a whole per-row array, and re-deriving it from unchanged
//! regret reproduces exactly the strategy a stored-strategy matcher would carry,
//! which is what makes a batch-size-1 matcher match its scalar counterpart
//! bit-for-bit.

use crate::lane::{F32Full, Layout, RegretLane, StrategyLane};
use crate::storage::{AccumCell, CounterCell, StorageBackend};
use crate::update_rule::UpdateRule;
use std::marker::PhantomData;

/// Per-call working buffers, sized to one row. Allocated once per update call
/// and reused across every row in a batch, so the per-row hot path allocates
/// nothing.
struct Scratch {
    regret: Vec<f32>,
    last_inst: Vec<f32>,
    strategy: Vec<f32>,
    reward: Vec<f32>,
}

impl Scratch {
    fn new(num_actions: usize) -> Self {
        Self {
            regret: vec![0.0; num_actions],
            last_inst: vec![0.0; num_actions],
            strategy: vec![0.0; num_actions],
            reward: vec![0.0; num_actions],
        }
    }
}

/// A batched regret matcher generic over the update rule `R`, the storage
/// backend `B`, and the memory layout `L` (which lane stores hold cumulative
/// regret and cumulative strategy). `L` defaults to [`F32Full`], reproducing
/// the previous all-f32 behavior, so `BatchedMatcher::<R, B>` keeps working.
pub struct BatchedMatcher<R: UpdateRule, B: StorageBackend, L: Layout<R, B> = F32Full> {
    params: R::Params,
    num_rows: usize,
    num_actions: usize,
    regret: L::Regret,
    strategy: L::Strategy,
    /// Last-instantaneous-regret lane for predictive rules, stored as f32 bits.
    /// Empty for non-predictive rules (`R::LANES <= 2`).
    last_inst: Vec<B::Cell<u32>>,
    counter: B::Counter,
    /// Shared regret-weight accumulator for the average-regret diagnostic; one
    /// scalar (f32 bits) for the whole batch, advanced once per tick.
    regret_weight: B::Cell<u32>,
    _rule: PhantomData<R>,
}

impl<R: UpdateRule, B: StorageBackend, L: Layout<R, B>> BatchedMatcher<R, B, L> {
    /// Create a matcher of `num_rows` information sets over `num_actions`
    /// actions. All accumulators start at zero, so every row reads as the
    /// uniform strategy until updated.
    ///
    /// # Panics
    ///
    /// Panics if `num_rows` or `num_actions` is zero.
    #[must_use]
    pub fn new(num_rows: usize, num_actions: usize, params: R::Params) -> Self {
        assert!(num_rows > 0, "num_rows must be > 0");
        assert!(num_actions > 0, "num_actions must be > 0");
        let last_inst = if R::LANES > 2 {
            (0..num_rows * num_actions)
                .map(|_| B::Cell::<u32>::default())
                .collect()
        } else {
            Vec::new()
        };
        Self {
            params,
            num_rows,
            num_actions,
            regret: L::Regret::new(num_rows, num_actions),
            strategy: L::Strategy::new(num_rows, num_actions),
            last_inst,
            counter: B::Counter::default(),
            regret_weight: B::Cell::<u32>::default(),
            _rule: PhantomData,
        }
    }

    #[inline]
    fn li_load(&self, idx: usize) -> f32 {
        f32::from_bits(self.last_inst[idx].load())
    }
    #[inline]
    fn li_store(&self, idx: usize, v: f32) {
        self.last_inst[idx].store(v.to_bits());
    }
    #[inline]
    fn rw_load(&self) -> f32 {
        f32::from_bits(self.regret_weight.load())
    }
    #[inline]
    fn rw_store(&self, v: f32) {
        self.regret_weight.store(v.to_bits());
    }

    /// Number of information sets.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of actions per information set.
    #[must_use]
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Number of updates (clock ticks) applied so far.
    #[must_use]
    pub fn num_updates(&self) -> usize {
        self.counter.load()
    }

    /// Advance the shared clock by one tick and compute the rule's per-iteration
    /// constants once for the resulting iteration.
    fn tick(&self) -> R::Step {
        let t = self.counter.fetch_incr() + 1;
        R::step(&self.params, t)
    }

    /// Apply this tick's regret-weight recurrence to the shared accumulator.
    fn advance_weight(&self, step: &R::Step) {
        self.rw_store(R::regret_weight_step(step, self.rw_load()));
    }

    /// Update one row against a precomputed step, returning the row's expected
    /// value under its pre-update strategy. This is the whole per-cell dance;
    /// `tick`/`advance_weight` handle the shared clock and weight around it.
    fn update_one(
        &self,
        row: usize,
        step: &R::Step,
        value: impl Fn(usize) -> f32,
        s: &mut Scratch,
    ) -> f32 {
        let a = self.num_actions;
        let predictive = R::LANES > 2;

        // Snapshot the lanes we read, and cache the rewards so the value
        // accessor is called exactly once per action.
        self.regret.read_row(row, a, &mut s.regret);
        for i in 0..a {
            if predictive {
                s.last_inst[i] = self.li_load(row * a + i);
            }
            s.reward[i] = value(i);
        }

        // Expected value uses the strategy carried in from the previous tick,
        // re-derived from the unchanged lanes (predictive rules discount the
        // regret term by the previous iterate's factor).
        R::strategy_from_lanes(
            &self.params,
            &s.regret,
            &s.last_inst,
            R::pre_discount(step),
            &mut s.strategy,
        );
        let expected = crate::vector_ops::dot(&s.strategy, &s.reward);

        // Per-cell regret update; predictive rules also store the fresh
        // instantaneous regret for next tick's prediction.
        for i in 0..a {
            s.regret[i] = R::accumulate_regret(step, s.regret[i], s.reward[i], expected);
            if predictive {
                let inst = s.reward[i] - expected;
                self.li_store(row * a + i, inst);
                s.last_inst[i] = inst;
            }
        }
        self.regret.write_row(row, a, &s.regret);

        // The strategy this tick plays (and accumulates) is derived from the
        // updated lanes, then folded into the cumulative-strategy lane.
        R::strategy_from_lanes(
            &self.params,
            &s.regret,
            &s.last_inst,
            R::post_discount(step),
            &mut s.strategy,
        );
        self.strategy.accumulate(row, a, step, &s.strategy);

        expected
    }

    /// Advance every row by one shared tick. `value(action, row)` supplies the
    /// reward for an action at a row from whatever layout the caller holds;
    /// `expected_out[row]` receives that row's expected value under its
    /// pre-update strategy (for propagating values up a game tree).
    ///
    /// # Panics
    ///
    /// Panics if `expected_out.len() < num_rows`.
    pub fn update_batch(&self, value: impl Fn(usize, usize) -> f32, expected_out: &mut [f32]) {
        assert!(
            expected_out.len() >= self.num_rows,
            "expected_out too short"
        );
        let step = self.tick();
        let mut scratch = Scratch::new(self.num_actions);
        for (row, ev) in expected_out.iter_mut().enumerate().take(self.num_rows) {
            *ev = self.update_one(row, &step, |a| value(a, row), &mut scratch);
        }
        self.advance_weight(&step);
    }

    /// Advance a single row by one shared tick, returning its expected value.
    /// For solvers that do not visit every row each iteration; the batching win
    /// only applies when many rows share a tick.
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows`.
    pub fn update_row(&self, row: usize, value: impl Fn(usize) -> f32) -> f32 {
        assert!(row < self.num_rows, "row out of range");
        let step = self.tick();
        let mut scratch = Scratch::new(self.num_actions);
        let ev = self.update_one(row, &step, value, &mut scratch);
        self.advance_weight(&step);
        ev
    }

    /// Zeros the average (cumulative-strategy) lane and its per-row weights;
    /// leaves cumulative regret, current strategy, and the clock untouched.
    /// Pair with `seed` for a clean warm start where the target drives only the
    /// current strategy and the exported average is rebuilt from re-equilibrated
    /// play. The regret-based diagnostics (`average_regret`) are intentionally
    /// preserved: `reset_average` zeros the average *strategy* lane only, not the
    /// cumulative regret or its weight accumulator.
    pub fn reset_average(&self) {
        self.strategy.reset();
    }

    /// Overwrite every row's cumulative-regret lane from `regret(action, row)` and
    /// set the shared iteration clock to `t0`, so the regret-matched `current_into`
    /// starts at a warm-started target and subsequent discounting behaves as if
    /// `t0` iterations had run. The strategy and last-instantaneous lanes are left
    /// untouched; the average builds from later iterations.
    ///
    /// Diagnostic note: for discounted rules the `average_regret` baseline is not
    /// reconstructed across a seed — treat it as a fresh diagnostic afterwards.
    pub fn seed(&self, regret: impl Fn(usize, usize) -> f32, t0: usize) {
        let a = self.num_actions;
        let mut row_buf = vec![0.0f32; a];
        for row in 0..self.num_rows {
            for (i, slot) in row_buf.iter_mut().enumerate() {
                *slot = regret(i, row);
            }
            self.regret.write_row(row, a, &row_buf);
        }
        self.counter.store(t0);
    }

    /// Write a row's current strategy (the distribution it would play next) into
    /// `out`. Equal to what a stored-strategy matcher would hold.
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows` or `out.len() < num_actions`.
    pub fn current_into(&self, row: usize, out: &mut [f32]) {
        assert!(row < self.num_rows, "row out of range");
        assert!(out.len() >= self.num_actions, "out too short");
        let out = &mut out[..self.num_actions];
        let t = self.num_updates();
        let step = R::step(&self.params, t);
        let predictive = R::LANES > 2;
        let mut regret = vec![0.0; self.num_actions];
        let mut last_inst = vec![0.0; self.num_actions];
        self.regret.read_row(row, self.num_actions, &mut regret);
        if predictive {
            for (i, slot) in last_inst.iter_mut().enumerate() {
                *slot = self.li_load(row * self.num_actions + i);
            }
        }
        R::strategy_from_lanes(
            &self.params,
            &regret,
            &last_inst,
            R::post_discount(&step),
            out,
        );
    }

    /// Write a row's average strategy — the normalized cumulative strategy, i.e.
    /// the equilibrium approximation — into `out`. This is the single,
    /// algorithm-independent readout the export codec consumes.
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows` or `out.len() < num_actions`.
    pub fn average_into(&self, row: usize, out: &mut [f32]) {
        assert!(row < self.num_rows, "row out of range");
        assert!(out.len() >= self.num_actions, "out too short");
        self.strategy.average_into(row, self.num_actions, out);
    }

    /// Write row `r`'s cumulative regret — the **signed accumulator** that drives
    /// regret matching — into `out`. This is the primitive CFR state, *not* the
    /// positive-part-normalized strategy `current_into` returns: the strategy is a
    /// read-only projection (take each action's positive regret, normalize), so
    /// the accumulator carries information (including negative regret) the strategy
    /// discards.
    ///
    /// It is exactly the quantity [`seed`](Self::seed) writes, read through the
    /// same [`RegretLane`](crate::lane::RegretLane), so `read → modify → seed`
    /// round-trips: bit-exact for the `F32Regret` store, and within one row-scaled
    /// quantum for the `Int16Regret` store (which decodes per-row-scaled i16).
    ///
    /// This is what makes an annealed warm-restart perturbation expressible:
    /// `regret_into(row)` → add decaying noise → `seed(.., num_updates())` re-aims
    /// the current strategy while keeping the iteration clock.
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows` or `out.len() < num_actions`.
    pub fn regret_into(&self, row: usize, out: &mut [f32]) {
        assert!(row < self.num_rows, "row out of range");
        assert!(out.len() >= self.num_actions, "out too short");
        self.regret
            .read_row(row, self.num_actions, &mut out[..self.num_actions]);
    }

    /// The average-regret convergence diagnostic for a row: the maximum positive
    /// cumulative regret over the shared regret-weight total. Tends to zero as
    /// the row approaches equilibrium; `0.0` before any update.
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows`.
    #[must_use]
    pub fn average_regret(&self, row: usize) -> f32 {
        assert!(row < self.num_rows, "row out of range");
        let w = R::regret_weight_total(&self.params, self.num_updates(), self.rw_load());
        if w <= 0.0 {
            return 0.0;
        }
        let mut regret = vec![0.0; self.num_actions];
        self.regret.read_row(row, self.num_actions, &mut regret);
        let max_pos = regret.iter().fold(0.0_f32, |m, &r| m.max(r.max(0.0)));
        max_pos / w
    }
}

#[cfg(test)]
impl<R: UpdateRule, B: StorageBackend> BatchedMatcher<R, B> {
    /// Raw cumulative-regret lane for a row (test-only; the public surface
    /// exposes derived strategies, not raw accumulators).
    fn raw_regret(&self, row: usize) -> Vec<f32> {
        let mut v = vec![0.0; self.num_actions];
        self.regret.read_row(row, self.num_actions, &mut v);
        v
    }

    /// Raw cumulative-strategy lane for a row (test-only). `F32SumStrategy`
    /// stores the un-normalized sum; expose it for the golden bit-test, which
    /// compares against the scalar matcher's `cumulative_strategy()`.
    fn raw_strategy(&self, row: usize) -> Vec<f32> {
        (0..self.num_actions)
            .map(|i| self.strategy.strategy_raw_cell(row, i, self.num_actions))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discount::DiscountParams;
    use crate::lane::HalfRegret;
    use crate::rules::{Dcfr, PdcfrPlus};
    use crate::storage::Local;

    #[test]
    fn fresh_matcher_reads_uniform() {
        let m = BatchedMatcher::<Dcfr, Local>::new(2, 3, DiscountParams::RECOMMENDED);
        let mut out = [0.0f32; 3];
        m.average_into(0, &mut out);
        assert!(
            out.iter().all(|&v| (v - 1.0 / 3.0).abs() < 1e-6),
            "avg {out:?}"
        );
        m.current_into(1, &mut out);
        assert!(
            out.iter().all(|&v| (v - 1.0 / 3.0).abs() < 1e-6),
            "cur {out:?}"
        );
        assert_eq!(m.num_updates(), 0);
    }

    #[test]
    fn symmetric_reward_from_uniform_has_zero_expected_value() {
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 3, DiscountParams::RECOMMENDED);
        let ev = m.update_row(0, |a| [1.0, -1.0, 0.0][a]);
        assert!(ev.abs() < 1e-6, "ev {ev}");
        assert_eq!(m.num_updates(), 1);
    }

    #[test]
    fn update_batch_fills_expected_value_per_row() {
        let m = BatchedMatcher::<Dcfr, Local>::new(2, 2, DiscountParams::RECOMMENDED);
        let mut ev = [0.0f32; 2];
        // Uniform strategy: row 0 EV = (2+0)/2 = 1; row 1 EV = (0+4)/2 = 2.
        m.update_batch(
            |a, row| {
                if row == 0 {
                    [2.0, 0.0][a]
                } else {
                    [0.0, 4.0][a]
                }
            },
            &mut ev,
        );
        assert!((ev[0] - 1.0).abs() < 1e-6, "{ev:?}");
        assert!((ev[1] - 2.0).abs() < 1e-6, "{ev:?}");
        assert_eq!(m.num_updates(), 1); // one shared tick for the whole batch
    }

    #[test]
    fn predictive_rule_uses_three_lanes() {
        // A 3-lane rule must allocate and drive its extra last-instantaneous
        // lane without panicking on the out-of-2-lane index.
        let m = BatchedMatcher::<PdcfrPlus, Local>::new(1, 3, PdcfrPlus::RECOMMENDED);
        let mut out = [0.0f32; 3];
        for _ in 0..5 {
            m.update_row(0, |a| [1.0, 0.0, -1.0][a]);
        }
        m.average_into(0, &mut out);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    // ── Golden bit-for-bit equivalence with the scalar matchers ─────────────
    //
    // A batch-size-1 matcher on the Local backend must reproduce its scalar
    // counterpart exactly — same cumulative regret, cumulative strategy, current
    // strategy, and average strategy — over an identical reward sequence. This
    // is the deterministic-reproducibility contract downstream solvers rely on,
    // so the comparison is on raw bit patterns, not an approximate tolerance.

    use crate::regret_minimizer::RegretMinimizer;
    use crate::rules::{DcfrPlus, LinearCfr, PcfrPlus};

    /// Deterministic reward stream in roughly `[-1, 1]`, identical for both
    /// matchers. A plain LCG keeps the sequence reproducible without pulling in
    /// `rand` or floating-point seeding.
    fn next_reward(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = (*state >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
        2.0 * unit - 1.0
    }

    fn assert_bits(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{label}[{i}]: {g} != {w}");
        }
    }

    /// Drive a batched rule and its scalar twin over the same rewards, asserting
    /// bit-equality of every quantity after every update.
    fn golden<R, M>(params: R::Params, mut scalar: M, num_actions: usize, iters: usize)
    where
        R: UpdateRule,
        M: RegretMinimizer,
    {
        let batched = BatchedMatcher::<R, Local>::new(1, num_actions, params);
        let mut state = 0x1234_5678_9abc_def0;
        let mut current = vec![0.0f32; num_actions];
        for _ in 0..iters {
            let rewards: Vec<f32> = (0..num_actions).map(|_| next_reward(&mut state)).collect();
            scalar.update_regret(&rewards);
            batched.update_row(0, |a| rewards[a]);

            assert_bits("regret", &batched.raw_regret(0), scalar.cumulative_regret());
            assert_bits(
                "strategy",
                &batched.raw_strategy(0),
                scalar.cumulative_strategy(),
            );
            batched.current_into(0, &mut current);
            assert_bits("current", &current, scalar.current_strategy());
        }
        let mut average = vec![0.0f32; num_actions];
        batched.average_into(0, &mut average);
        assert_bits("average", &average, &scalar.best_weight());
    }

    #[test]
    fn golden_dcfr() {
        golden::<Dcfr, _>(
            DiscountParams::RECOMMENDED,
            DiscountedRegretMatcher::recommended(4),
            4,
            200,
        );
    }

    #[test]
    fn golden_dcfr_plus() {
        golden::<DcfrPlus, _>(
            DcfrPlus::RECOMMENDED,
            DcfrPlusRegretMatcher::recommended(4),
            4,
            200,
        );
    }

    #[test]
    fn golden_linear_cfr() {
        golden::<LinearCfr, _>((), LinearCfrRegretMatcher::new(4), 4, 200);
    }

    #[test]
    fn golden_pcfr_plus() {
        golden::<PcfrPlus, _>((), PcfrPlusRegretMatcher::new(4), 4, 200);
    }

    #[test]
    fn golden_pdcfr_plus() {
        golden::<PdcfrPlus, _>(
            PdcfrPlus::RECOMMENDED,
            PdcfrPlusRegretMatcher::recommended(4),
            4,
            200,
        );
    }

    use crate::{
        DcfrPlusRegretMatcher, DiscountedRegretMatcher, LinearCfrRegretMatcher,
        PcfrPlusRegretMatcher, PdcfrPlusRegretMatcher,
    };

    #[test]
    fn seed_from_own_regret_is_noop_on_current() {
        use crate::storage::Local;
        let m = BatchedMatcher::<Dcfr, Local>::new(2, 3, DiscountParams::RECOMMENDED);
        let mut ev = [0.0f32; 2];
        for _ in 0..10 {
            m.update_batch(|a, _| [1.0, -0.5, 0.2][a], &mut ev);
        }
        let mut before = [0.0f32; 3];
        m.current_into(1, &mut before);
        let t = m.num_updates();
        let snaps: Vec<Vec<f32>> = (0..2).map(|r| m.raw_regret(r)).collect();
        m.seed(|a, row| snaps[row][a], t);
        let mut after = [0.0f32; 3];
        m.current_into(1, &mut after);
        for (x, y) in before.iter().zip(&after) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "seed from own regret must be a no-op"
            );
        }
        assert_eq!(m.num_updates(), t);
    }

    #[test]
    fn seed_positive_regret_reproduces_target_current_strategy() {
        use crate::storage::Local;
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 3, DiscountParams::RECOMMENDED);
        let target = [0.2f32, 0.3, 0.5];
        // Positive regret proportional to the target → regret-matching normalizes to it.
        m.seed(|a, _row| 100.0 * target[a], 50);
        let mut out = [0.0f32; 3];
        m.current_into(0, &mut out);
        for (a, b) in target.iter().zip(&out) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        assert_eq!(m.num_updates(), 50);
    }

    #[test]
    fn reset_average_makes_average_uniform() {
        let m = BatchedMatcher::<Dcfr, Local>::new(2, 3, DiscountParams::RECOMMENDED);
        let mut ev = [0.0f32; 2];
        // Run ~20 updates with asymmetric rewards to push the average away from uniform.
        for _ in 0..20 {
            m.update_batch(|a, _| [1.0, -0.5, 0.2][a], &mut ev);
        }
        // Snapshot current strategy and clock for row 1.
        let mut current_before = [0.0f32; 3];
        m.current_into(1, &mut current_before);
        let updates_before = m.num_updates();

        // Reset only the average lane.
        m.reset_average();

        // Average should now be uniform (zeroed sum → normalize → 1/3 each).
        let mut avg = [0.0f32; 3];
        for row in 0..2 {
            m.average_into(row, &mut avg);
            for &v in &avg {
                assert!(
                    (v - 1.0 / 3.0).abs() < 1e-6,
                    "row {row}: expected uniform after reset, got {avg:?}"
                );
            }
        }

        // current_into and num_updates must be unchanged.
        let mut current_after = [0.0f32; 3];
        m.current_into(1, &mut current_after);
        assert_eq!(m.num_updates(), updates_before, "clock must not change");
        for (b, a) in current_before.iter().zip(&current_after) {
            assert_eq!(
                b.to_bits(),
                a.to_bits(),
                "current strategy must be unchanged after reset_average"
            );
        }
    }

    #[test]
    fn seed_under_i16_reproduces_target_within_tolerance() {
        use crate::lane::HalfBoth;
        use crate::storage::Local;
        let m = BatchedMatcher::<Dcfr, Local, HalfBoth>::new(1, 3, DiscountParams::RECOMMENDED);
        let target = [0.2f32, 0.3, 0.5];
        m.seed(|a, _| 100.0 * target[a], 50);
        let mut out = [0.0f32; 3];
        m.current_into(0, &mut out);
        for (a, b) in target.iter().zip(&out) {
            assert!(
                (a - b).abs() < 2e-3,
                "i16 seed within tolerance: {a} vs {b}"
            );
        }
        assert_eq!(m.num_updates(), 50);
    }

    #[test]
    fn regret_into_matches_raw_regret_helper() {
        // Build regret via a deterministic update sequence, then the public
        // reader must equal the test-only raw_regret helper element-for-element.
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 3, DiscountParams::RECOMMENDED);
        for _ in 0..7 {
            m.update_row(0, |a| [1.0, -0.5, 0.2][a]);
        }
        let mut out = [0.0f32; 3];
        m.regret_into(0, &mut out);
        let raw = m.raw_regret(0);
        for (i, (&g, &w)) in out.iter().zip(&raw).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "regret_into[{i}] {g} != raw {w}");
        }
    }

    #[test]
    fn regret_into_then_seed_is_noop_on_current() {
        // read → seed(read values, t0 = num_updates()) leaves next current strategy
        // bit-identical for the exact F32 regret store.
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 3, DiscountParams::RECOMMENDED);
        for _ in 0..10 {
            m.update_row(0, |a| [0.8, -0.3, 0.1][a]);
        }
        let mut before = [0.0f32; 3];
        m.current_into(0, &mut before);

        let mut r = [0.0f32; 3];
        m.regret_into(0, &mut r);
        m.seed(|a, _row| r[a], m.num_updates());

        let mut after = [0.0f32; 3];
        m.current_into(0, &mut after);
        for (i, (&b, &a)) in before.iter().zip(&after).enumerate() {
            assert_eq!(b.to_bits(), a.to_bits(), "current[{i}] changed: {b} != {a}");
        }
    }

    #[test]
    fn regret_into_decodes_int16_layout_within_quantum() {
        // On the i16 regret layout, seed a known vector and read it back: decode
        // must land within one row-scaled quantum of the seeded values.
        let m = BatchedMatcher::<Dcfr, Local, HalfRegret>::new(1, 3, DiscountParams::RECOMMENDED);
        let seeded = [1000.0f32, -250.0, 30.0];
        m.seed(|a, _row| seeded[a], 5);
        let mut out = [0.0f32; 3];
        m.regret_into(0, &mut out);
        // Int16Regret scale ≈ peak/i16::MAX = 1000/32767.
        let quantum = 1000.0 / i16::MAX as f32;
        for (i, (&g, &w)) in out.iter().zip(&seeded).enumerate() {
            assert!(
                (g - w).abs() <= quantum + 1e-2,
                "regret_into[{i}] {g} vs seeded {w}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "row out of range")]
    fn regret_into_panics_on_bad_row() {
        let m = BatchedMatcher::<Dcfr, Local>::new(2, 3, DiscountParams::RECOMMENDED);
        let mut out = [0.0f32; 3];
        m.regret_into(2, &mut out);
    }

    #[test]
    #[should_panic(expected = "out too short")]
    fn regret_into_panics_on_short_out() {
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 3, DiscountParams::RECOMMENDED);
        let mut out = [0.0f32; 2];
        m.regret_into(0, &mut out);
    }
}
