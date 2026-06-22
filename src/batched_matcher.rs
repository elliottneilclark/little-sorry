//! A matcher owning many information sets over a pluggable cell backend.
//!
//! One instance holds `num_rows` information sets ("rows"), each over the same
//! `num_actions`, laid out as a single flat array of cells addressed
//! `[row][lane][action]` (lanes per [`crate::update_rule`]). All rows share one
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

use crate::storage::{CounterCell, FloatCell, StorageBackend};
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

/// A batched regret matcher generic over the update rule `R` and the storage
/// backend `B`.
pub struct BatchedMatcher<R: UpdateRule, B: StorageBackend> {
    params: R::Params,
    num_rows: usize,
    num_actions: usize,
    row_stride: usize,
    cells: Vec<B::Float>,
    counter: B::Counter,
    /// Shared regret-weight accumulator for the average-regret diagnostic; one
    /// scalar for the whole batch, advanced once per tick.
    regret_weight: B::Float,
    _rule: PhantomData<R>,
}

impl<R: UpdateRule, B: StorageBackend> BatchedMatcher<R, B> {
    /// Lane indices. Lane 0 holds cumulative regret, lane 1 cumulative strategy,
    /// and (predictive rules) lane 2 the last instantaneous regret.
    const REGRET: usize = 0;
    const STRATEGY: usize = 1;
    const LAST_INST: usize = 2;

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
        let row_stride = R::LANES * num_actions;
        let cells = (0..num_rows * row_stride)
            .map(|_| B::Float::default())
            .collect();
        Self {
            params,
            num_rows,
            num_actions,
            row_stride,
            cells,
            counter: B::Counter::default(),
            regret_weight: B::Float::default(),
            _rule: PhantomData,
        }
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

    #[inline]
    fn cell(&self, row: usize, lane: usize, action: usize) -> &B::Float {
        &self.cells[row * self.row_stride + lane * self.num_actions + action]
    }

    /// Advance the shared clock by one tick and compute the rule's per-iteration
    /// constants once for the resulting iteration.
    fn tick(&self) -> R::Step {
        let t = self.counter.fetch_incr() + 1;
        R::step(&self.params, t)
    }

    /// Apply this tick's regret-weight recurrence to the shared accumulator.
    fn advance_weight(&self, step: &R::Step) {
        self.regret_weight
            .store(R::regret_weight_step(step, self.regret_weight.load()));
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
        let predictive = R::LANES > Self::LAST_INST;

        // Snapshot the lanes we read, and cache the rewards so the value
        // accessor is called exactly once per action.
        for i in 0..a {
            s.regret[i] = self.cell(row, Self::REGRET, i).load();
            if predictive {
                s.last_inst[i] = self.cell(row, Self::LAST_INST, i).load();
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
            let new_r = R::accumulate_regret(step, s.regret[i], s.reward[i], expected);
            self.cell(row, Self::REGRET, i).store(new_r);
            s.regret[i] = new_r;
            if predictive {
                let inst = s.reward[i] - expected;
                self.cell(row, Self::LAST_INST, i).store(inst);
                s.last_inst[i] = inst;
            }
        }

        // The strategy this tick plays (and accumulates) is derived from the
        // updated lanes, then folded into the cumulative-strategy lane.
        R::strategy_from_lanes(
            &self.params,
            &s.regret,
            &s.last_inst,
            R::post_discount(step),
            &mut s.strategy,
        );
        let (discount, weight) = R::strategy_accumulation(step);
        for i in 0..a {
            let cell = self.cell(row, Self::STRATEGY, i);
            cell.store(cell.load() * discount + weight * s.strategy[i]);
        }

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
        let predictive = R::LANES > Self::LAST_INST;
        let mut regret = vec![0.0; self.num_actions];
        let mut last_inst = vec![0.0; self.num_actions];
        for i in 0..self.num_actions {
            regret[i] = self.cell(row, Self::REGRET, i).load();
            if predictive {
                last_inst[i] = self.cell(row, Self::LAST_INST, i).load();
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
        let out = &mut out[..self.num_actions];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.cell(row, Self::STRATEGY, i).load();
        }
        crate::probability::normalize_inplace(out);
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
        let w = R::regret_weight_total(&self.params, self.num_updates(), self.regret_weight.load());
        if w <= 0.0 {
            return 0.0;
        }
        let max_pos = (0..self.num_actions).fold(0.0_f32, |m, i| {
            m.max(self.cell(row, Self::REGRET, i).load().max(0.0))
        });
        max_pos / w
    }
}

#[cfg(test)]
impl<R: UpdateRule, B: StorageBackend> BatchedMatcher<R, B> {
    /// Raw cumulative-regret lane for a row (test-only; the public surface
    /// exposes derived strategies, not raw accumulators).
    fn raw_regret(&self, row: usize) -> Vec<f32> {
        (0..self.num_actions)
            .map(|i| self.cell(row, Self::REGRET, i).load())
            .collect()
    }

    /// Raw cumulative-strategy lane for a row (test-only).
    fn raw_strategy(&self, row: usize) -> Vec<f32> {
        (0..self.num_actions)
            .map(|i| self.cell(row, Self::STRATEGY, i).load())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discount::DiscountParams;
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
}
