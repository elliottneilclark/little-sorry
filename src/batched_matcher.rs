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

/// Reusable working buffers for one row's update. Constructed once (e.g. one
/// per worker thread) and reused across calls via
/// [`BatchedMatcher::update_batch_with`], so the per-visit hot path performs
/// no heap allocation. A scratch built for `num_actions` serves any matcher
/// with that many actions or fewer.
pub struct Scratch {
    regret: Vec<f32>,
    last_inst: Vec<f32>,
    strategy: Vec<f32>,
    reward: Vec<f32>,
    /// Per-action traversal mask for the masked update paths.
    active: Vec<bool>,
}

impl Scratch {
    /// Buffers sized for matchers with up to `num_actions` actions.
    #[must_use]
    pub fn new(num_actions: usize) -> Self {
        Self {
            regret: vec![0.0; num_actions],
            last_inst: vec![0.0; num_actions],
            strategy: vec![0.0; num_actions],
            reward: vec![0.0; num_actions],
            active: vec![true; num_actions],
        }
    }

    /// The action capacity this scratch was built for.
    #[must_use]
    pub fn num_actions(&self) -> usize {
        self.regret.len()
    }
}

/// Actions covered by the stack-allocated row buffer; larger rows fall back
/// to the heap. Downstream consumers run ≤ 4 actions.
const INLINE_ACTIONS: usize = 8;

/// A row-sized f32 buffer: inline array up to [`INLINE_ACTIONS`], heap `Vec`
/// beyond, so read paths stay allocation-free at practical action counts.
enum RowBuf {
    Inline([f32; INLINE_ACTIONS]),
    Heap(Vec<f32>),
}

impl RowBuf {
    fn new(len: usize) -> Self {
        if len <= INLINE_ACTIONS {
            RowBuf::Inline([0.0; INLINE_ACTIONS])
        } else {
            RowBuf::Heap(vec![0.0; len])
        }
    }
    fn slice_mut(&mut self, len: usize) -> &mut [f32] {
        match self {
            RowBuf::Inline(a) => &mut a[..len],
            RowBuf::Heap(v) => &mut v[..len],
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
    /// uniform strategy until updated. The regret lane is built with its
    /// default [`RegretLane::Config`]; see
    /// [`with_regret_config`](Self::with_regret_config) to choose one.
    ///
    /// # Panics
    ///
    /// Panics if `num_rows` or `num_actions` is zero.
    #[must_use]
    pub fn new(num_rows: usize, num_actions: usize, params: R::Params) -> Self {
        Self::with_regret_config(num_rows, num_actions, params, Default::default())
    }

    /// [`new`](Self::new) with an explicit regret-lane configuration — the
    /// int32 lanes' scale and floor ([`Int32Config`](crate::lane::Int32Config));
    /// `()` for the f32 and i16 lanes.
    ///
    /// # Panics
    ///
    /// Panics if `num_rows` or `num_actions` is zero, or if the lane rejects
    /// `config`.
    #[must_use]
    pub fn with_regret_config(
        num_rows: usize,
        num_actions: usize,
        params: R::Params,
        config: <L::Regret as RegretLane<B>>::Config,
    ) -> Self {
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
            regret: L::Regret::new(num_rows, num_actions, config),
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

    /// The floor the regret lane clamps stored regret to, if it has one
    /// (`None` for the unbounded f32 and i16 lanes). A pruning caller compares
    /// against this to know how far below zero a skipped action can sit.
    #[must_use]
    pub fn regret_floor(&self) -> Option<f32> {
        self.regret.floor()
    }

    /// The regret lane itself, for value-exact checkpoint export — e.g.
    /// [`Int32Regret::codes_row`](crate::lane::Int32Regret::codes_row), which
    /// hands out raw codes where [`regret_into`](Self::regret_into) would round
    /// through f32.
    #[must_use]
    pub fn regret_lane(&self) -> &L::Regret {
        &self.regret
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
        // accessor is called exactly once per action. The scratch may be
        // over-sized (built for a wider matcher), so every buffer is sliced
        // to this matcher's action count.
        self.regret.read_row(row, a, &mut s.regret[..a]);
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
            &s.regret[..a],
            &s.last_inst[..a],
            R::pre_discount(step),
            &mut s.strategy[..a],
        );
        let expected = crate::vector_ops::dot(&s.strategy[..a], &s.reward[..a]);

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
        // The f32 lanes store the row as computed; wider lanes re-apply the
        // rule's (discount, increment) split in their own precision.
        let (regret, reward) = (&mut s.regret[..a], &s.reward[..a]);
        self.regret.accumulate_row(
            row,
            a,
            regret,
            |i, old| {
                Some((
                    R::regret_discount(step, old),
                    R::regret_increment(step, reward[i], expected),
                ))
            },
            R::FLOORS_REGRET,
            self.num_updates(),
        );

        // The strategy this tick plays (and accumulates) is derived from the
        // updated lanes, then folded into the cumulative-strategy lane.
        R::strategy_from_lanes(
            &self.params,
            &s.regret[..a],
            &s.last_inst[..a],
            R::post_discount(step),
            &mut s.strategy[..a],
        );
        self.strategy
            .accumulate(row, a, step, &s.strategy[..a], self.num_updates());

        expected
    }

    /// [`update_one`](Self::update_one) where only the actions `active` admits
    /// were traversed this tick. Inactive actions have no reward: their regret
    /// and `last_inst` cells are left exactly as stored, and the expected value
    /// is taken over the active set with the pre-update strategy renormalised
    /// over it. When every action is active this is `update_one` step for
    /// step, including the exact `dot` for the expected value.
    fn update_one_masked(
        &self,
        row: usize,
        step: &R::Step,
        value: impl Fn(usize) -> f32,
        active: impl Fn(usize) -> bool,
        s: &mut Scratch,
    ) -> f32 {
        debug_assert!(
            L::Regret::MASK_EXACT,
            "masked updates are unsupported on this layout: its regret lane \
             cannot round-trip an inactive action's regret unchanged"
        );
        let a = self.num_actions;
        let predictive = R::LANES > 2;

        self.regret.read_row(row, a, &mut s.regret[..a]);
        let mut all_active = true;
        for i in 0..a {
            if predictive {
                s.last_inst[i] = self.li_load(row * a + i);
            }
            let on = active(i);
            s.active[i] = on;
            all_active &= on;
            // The value accessor is only consulted for traversed actions.
            s.reward[i] = if on { value(i) } else { 0.0 };
        }

        // Pre-update strategy over *all* actions: a pruned action with negative
        // regret already carries zero mass; a positive-regret action the caller
        // chose not to traverse keeps its mass and is renormalised out of
        // `expected` only.
        R::strategy_from_lanes(
            &self.params,
            &s.regret[..a],
            &s.last_inst[..a],
            R::pre_discount(step),
            &mut s.strategy[..a],
        );
        let expected = if all_active {
            crate::vector_ops::dot(&s.strategy[..a], &s.reward[..a])
        } else {
            masked_expected(&s.strategy[..a], &s.reward[..a], &s.active[..a])
        };

        for i in 0..a {
            if !s.active[i] {
                continue;
            }
            s.regret[i] = R::accumulate_regret(step, s.regret[i], s.reward[i], expected);
            if predictive {
                let inst = s.reward[i] - expected;
                self.li_store(row * a + i, inst);
                s.last_inst[i] = inst;
            }
        }
        // `None` leaves an inactive cell untouched: the f32 lane skips the
        // store, code-space lanes skip the cell but keep their rounding draw
        // aligned.
        let (regret, reward, mask) = (&mut s.regret[..a], &s.reward[..a], &s.active[..a]);
        self.regret.accumulate_row(
            row,
            a,
            regret,
            |i, old| {
                mask[i].then(|| {
                    (
                        R::regret_discount(step, old),
                        R::regret_increment(step, reward[i], expected),
                    )
                })
            },
            R::FLOORS_REGRET,
            self.num_updates(),
        );

        R::strategy_from_lanes(
            &self.params,
            &s.regret[..a],
            &s.last_inst[..a],
            R::post_discount(step),
            &mut s.strategy[..a],
        );
        self.strategy
            .accumulate(row, a, step, &s.strategy[..a], self.num_updates());

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
        self.update_batch_with(&mut Scratch::new(self.num_actions), value, expected_out);
    }

    /// [`update_batch`](Self::update_batch) with caller-owned scratch buffers —
    /// the allocation-free hot path. Keep one [`Scratch`] per worker thread and
    /// reuse it across calls and matchers.
    ///
    /// # Panics
    ///
    /// Panics if `expected_out.len() < num_rows` or
    /// `scratch.num_actions() < num_actions`.
    pub fn update_batch_with(
        &self,
        scratch: &mut Scratch,
        value: impl Fn(usize, usize) -> f32,
        expected_out: &mut [f32],
    ) {
        assert!(
            expected_out.len() >= self.num_rows,
            "expected_out too short"
        );
        assert!(
            scratch.num_actions() >= self.num_actions,
            "scratch too small"
        );
        let step = self.tick();
        for (row, ev) in expected_out.iter_mut().enumerate().take(self.num_rows) {
            *ev = self.update_one(row, &step, |a| value(a, row), scratch);
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
        self.update_row_with(&mut Scratch::new(self.num_actions), row, value)
    }

    /// [`update_row`](Self::update_row) with caller-owned scratch buffers — the
    /// allocation-free single-row path.
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows` or `scratch.num_actions() < num_actions`.
    pub fn update_row_with(
        &self,
        scratch: &mut Scratch,
        row: usize,
        value: impl Fn(usize) -> f32,
    ) -> f32 {
        assert!(row < self.num_rows, "row out of range");
        assert!(
            scratch.num_actions() >= self.num_actions,
            "scratch too small"
        );
        let step = self.tick();
        let ev = self.update_one(row, &step, value, scratch);
        self.advance_weight(&step);
        ev
    }

    /// [`update_row_with`](Self::update_row_with) where `active(action)` says
    /// whether that action was traversed this tick — the partial update a
    /// regret-pruning solver needs when it skips an action's subtree and so has
    /// no reward for it.
    ///
    /// Inactive actions keep their stored regret exactly (no discount, no add)
    /// and, for predictive rules, their last-instantaneous regret; `value` is
    /// not called for them. The pre-update strategy is still derived from the
    /// full lanes, and the returned expected value is
    /// `Σ_active σ(a)·value(a) / Σ_active σ(a)` — the plain mean of the active
    /// rewards if no active action carries mass, `0.0` if nothing is active.
    /// The post-update strategy is derived from the full updated lanes and
    /// accumulated as usual, and the clock advances once. With every action
    /// active this is [`update_row_with`](Self::update_row_with) bit for bit.
    ///
    /// Unsupported on the i16 regret layouts, whose per-row rescale cannot
    /// leave an inactive cell untouched ([`RegretLane::MASK_EXACT`]); debug
    /// builds assert this.
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows` or `scratch.num_actions() < num_actions`.
    pub fn update_row_masked_with(
        &self,
        scratch: &mut Scratch,
        row: usize,
        value: impl Fn(usize) -> f32,
        active: impl Fn(usize) -> bool,
    ) -> f32 {
        assert!(row < self.num_rows, "row out of range");
        assert!(
            scratch.num_actions() >= self.num_actions,
            "scratch too small"
        );
        let step = self.tick();
        let ev = self.update_one_masked(row, &step, value, active, scratch);
        self.advance_weight(&step);
        ev
    }

    /// [`update_batch_with`](Self::update_batch_with) with a per-row action
    /// mask `active(action, row)`; see
    /// [`update_row_masked_with`](Self::update_row_masked_with) for the
    /// semantics of an inactive action.
    ///
    /// # Panics
    ///
    /// Panics if `expected_out.len() < num_rows` or
    /// `scratch.num_actions() < num_actions`.
    pub fn update_batch_masked_with(
        &self,
        scratch: &mut Scratch,
        value: impl Fn(usize, usize) -> f32,
        active: impl Fn(usize, usize) -> bool,
        expected_out: &mut [f32],
    ) {
        assert!(
            expected_out.len() >= self.num_rows,
            "expected_out too short"
        );
        assert!(
            scratch.num_actions() >= self.num_actions,
            "scratch too small"
        );
        let step = self.tick();
        for (row, ev) in expected_out.iter_mut().enumerate().take(self.num_rows) {
            *ev =
                self.update_one_masked(row, &step, |a| value(a, row), |a| active(a, row), scratch);
        }
        self.advance_weight(&step);
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
        self.counter.store(t0);
        let update_count = self.num_updates();
        for row in 0..self.num_rows {
            for (i, slot) in row_buf.iter_mut().enumerate() {
                *slot = regret(i, row);
            }
            self.regret.write_row(row, a, &row_buf, update_count);
        }
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
        let n = self.num_actions;
        let out = &mut out[..n];
        let t = self.num_updates();
        let step = R::step(&self.params, t);
        let predictive = R::LANES > 2;

        let mut regret_buf = RowBuf::new(n);
        let regret = regret_buf.slice_mut(n);
        self.regret.read_row(row, n, regret);

        // 2-lane rules never read the last-instantaneous lane; hand them an
        // empty slice instead of materializing a dead buffer.
        let last_n = if predictive { n } else { 0 };
        let mut last_buf = RowBuf::new(last_n);
        let last_inst = last_buf.slice_mut(last_n);
        for (i, slot) in last_inst.iter_mut().enumerate() {
            *slot = self.li_load(row * n + i);
        }

        R::strategy_from_lanes(
            &self.params,
            regret,
            last_inst,
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
    /// Panics if `row >= num_rows`, `out.len() < num_actions`, or the layout
    /// keeps no average lane ([`NoStrategy`](crate::lane::NoStrategy)) — snapshot
    /// [`current_into`](Self::current_into) instead.
    pub fn average_into(&self, row: usize, out: &mut [f32]) {
        assert!(
            L::Strategy::HAS_AVERAGE,
            "{} keeps no average lane; snapshot `current_into` instead",
            std::any::type_name::<L>()
        );
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
    /// same [`RegretLane`], so `read → modify → seed`
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

/// Expected value of `reward` under `strategy` restricted to the `active`
/// actions: `Σ_active σ·r / Σ_active σ`, the plain mean of the active rewards
/// when the active set carries no mass, and `0.0` when nothing is active.
fn masked_expected(strategy: &[f32], reward: &[f32], active: &[bool]) -> f32 {
    let mut weighted = 0.0f32;
    let mut mass = 0.0f32;
    let mut plain = 0.0f32;
    let mut count = 0usize;
    for ((&s, &r), &on) in strategy.iter().zip(reward).zip(active) {
        if on {
            weighted += s * r;
            mass += s;
            plain += r;
            count += 1;
        }
    }
    if mass > 0.0 {
        weighted / mass
    } else if count > 0 {
        plain / count as f32
    } else {
        0.0
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
    use crate::unit_fixed::RowDraws;

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

    #[test]
    fn update_batch_with_matches_update_batch_bit_for_bit() {
        // Twin matchers over an identical deterministic reward stream; one uses
        // the allocating entry point, the other a reused, over-sized scratch.
        let params = DiscountParams::RECOMMENDED;
        let a = BatchedMatcher::<Dcfr, Local>::new(3, 3, params);
        let b = BatchedMatcher::<Dcfr, Local>::new(3, 3, params);
        let mut scratch = Scratch::new(5); // over-sized on purpose
        let mut ev_a = [0.0f32; 3];
        let mut ev_b = [0.0f32; 3];
        let mut state = 0x9876_5432_10ab_cdefu64;
        for _ in 0..100 {
            let rewards: Vec<f32> = (0..9).map(|_| next_reward(&mut state)).collect();
            a.update_batch(|act, row| rewards[row * 3 + act], &mut ev_a);
            b.update_batch_with(&mut scratch, |act, row| rewards[row * 3 + act], &mut ev_b);
            for (x, y) in ev_a.iter().zip(&ev_b) {
                assert_eq!(x.to_bits(), y.to_bits(), "expected values diverged");
            }
            for row in 0..3 {
                assert_bits("regret", &b.raw_regret(row), &a.raw_regret(row));
                assert_bits("strategy", &b.raw_strategy(row), &a.raw_strategy(row));
            }
        }
    }

    #[test]
    fn update_row_with_matches_update_row_bit_for_bit() {
        let params = DiscountParams::RECOMMENDED;
        let a = BatchedMatcher::<Dcfr, Local>::new(1, 3, params);
        let b = BatchedMatcher::<Dcfr, Local>::new(1, 3, params);
        let mut scratch = Scratch::new(3);
        let mut state = 0x0f0f_1234_dead_beefu64;
        for _ in 0..100 {
            let rewards: Vec<f32> = (0..3).map(|_| next_reward(&mut state)).collect();
            let ev_a = a.update_row(0, |act| rewards[act]);
            let ev_b = b.update_row_with(&mut scratch, 0, |act| rewards[act]);
            assert_eq!(ev_a.to_bits(), ev_b.to_bits(), "expected values diverged");
            assert_bits("regret", &b.raw_regret(0), &a.raw_regret(0));
            assert_bits("strategy", &b.raw_strategy(0), &a.raw_strategy(0));
        }
    }

    #[test]
    #[should_panic(expected = "scratch too small")]
    fn update_batch_with_panics_on_small_scratch() {
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 3, DiscountParams::RECOMMENDED);
        let mut scratch = Scratch::new(2);
        let mut ev = [0.0f32; 1];
        m.update_batch_with(&mut scratch, |a, _| a as f32, &mut ev);
    }

    // ── int32 config, no-average layout, masked updates ─────────────────────

    use crate::lane::{Int32Config, Int32Full, Int32NoAverage};

    #[test]
    #[should_panic(expected = "no average lane")]
    fn average_into_panics_on_no_average_layout() {
        let m =
            BatchedMatcher::<Dcfr, Local, Int32NoAverage>::new(1, 3, DiscountParams::RECOMMENDED);
        let mut out = [0.0f32; 3];
        m.average_into(0, &mut out);
    }

    #[test]
    fn no_average_layout_updates_and_reads_current() {
        let m =
            BatchedMatcher::<Dcfr, Local, Int32NoAverage>::new(2, 3, DiscountParams::RECOMMENDED);
        let mut ev = [0.0f32; 2];
        for _ in 0..50 {
            m.update_batch(|a, _| [1.0, -0.5, 0.2][a], &mut ev);
        }
        m.reset_average(); // no-op, must not panic
        let mut cur = [0.0f32; 3];
        m.current_into(1, &mut cur);
        assert!(cur[0] > 0.9, "action 0 dominates: {cur:?}");
        assert!((cur.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(m.num_updates(), 50);
    }

    #[test]
    fn with_regret_config_threads_floor() {
        let cfg = Int32Config {
            scale: 100.0,
            floor: -7.5,
        };
        let m = BatchedMatcher::<Dcfr, Local, Int32Full>::with_regret_config(
            1,
            3,
            DiscountParams::RECOMMENDED,
            cfg,
        );
        assert_eq!(m.regret_floor(), Some(-7.5));
        assert_eq!(m.regret_lane().scale(), 100.0);
        // Seeding below the floor clamps; the lane reports it through regret_into.
        m.seed(|a, _| [-100.0, 0.0, 3.0][a], 0);
        let mut r = [0.0f32; 3];
        m.regret_into(0, &mut r);
        assert_eq!(r, [-7.5, 0.0, 3.0]);

        let f = BatchedMatcher::<Dcfr, Local>::new(1, 3, DiscountParams::RECOMMENDED);
        assert_eq!(f.regret_floor(), None);
        let d = BatchedMatcher::<Dcfr, Local, Int32Full>::new(1, 3, DiscountParams::RECOMMENDED);
        assert_eq!(d.regret_floor(), Some(i32::MIN as f32 / 100.0));
    }

    #[test]
    fn int32_average_uses_stored_regret() {
        let t = (1..)
            .find(|&t| RowDraws::new(0, t).next_u01() > 0.5)
            .expect("row draws include a value above one half");
        let increment = RowDraws::new(0, t).next_u01() * 0.5;
        let m = BatchedMatcher::<Dcfr, Local, Int32Full>::with_regret_config(
            1,
            2,
            DiscountParams::RECOMMENDED,
            Int32Config {
                scale: 1.0,
                ..Int32Config::default()
            },
        );
        m.seed(|_, _| 0.0, t - 1);
        m.update_row(0, |a| if a == 0 { 2.0 * increment } else { 0.0 });

        let mut current = [0.0f32; 2];
        m.current_into(0, &mut current);
        assert_eq!(current, [0.5, 0.5], "stored regrets have no positive code");

        let mut average = [0.0f32; 2];
        m.average_into(0, &mut average);
        assert_eq!(
            average, current,
            "average must use the stored regret strategy"
        );
    }

    #[test]
    fn int32_full_tracks_f32_full_on_the_golden_stream() {
        // Same reward stream through F32Full and Int32Full: cumulative regret
        // agrees to within the accumulated rounding (a random walk of ≤ 1
        // quantum per tick) and the average strategy agrees closely.
        let params = DiscountParams::RECOMMENDED;
        let f = BatchedMatcher::<Dcfr, Local>::new(1, 4, params);
        let q = BatchedMatcher::<Dcfr, Local, Int32Full>::new(1, 4, params);
        let mut state = 0x5555_aaaa_1234_5678u64;
        let ticks = 300;
        for _ in 0..ticks {
            let rewards: Vec<f32> = (0..4).map(|_| next_reward(&mut state)).collect();
            f.update_row(0, |a| rewards[a]);
            q.update_row(0, |a| rewards[a]);
        }
        let (mut rf, mut rq) = ([0.0f32; 4], [0.0f32; 4]);
        f.regret_into(0, &mut rf);
        q.regret_into(0, &mut rq);
        for a in 0..4 {
            assert!(
                (rf[a] - rq[a]).abs() < 0.01 * (ticks as f32).sqrt() * 3.0,
                "regret[{a}] {} vs {}",
                rf[a],
                rq[a]
            );
        }
        let (mut af, mut aq) = ([0.0f32; 4], [0.0f32; 4]);
        f.average_into(0, &mut af);
        q.average_into(0, &mut aq);
        for a in 0..4 {
            assert!(
                (af[a] - aq[a]).abs() < 1e-2,
                "average[{a}] {} vs {}",
                af[a],
                aq[a]
            );
        }
    }

    #[test]
    fn masked_update_leaves_inactive_regret_untouched() {
        fn run<L: Layout<Dcfr, Local>>(
            masked: &BatchedMatcher<Dcfr, Local, L>,
            ev_tol: f32,
        ) -> (BatchedMatcher<Dcfr, Local, L>, Vec<f32>) {
            // Action 2 is masked out for the whole run; its seed is negative so
            // it carries no mass, making the other three a pure 3-action game.
            let seed = [0.5f32, -0.25, -5.0, 0.75];
            masked.seed(|a, _| seed[a], 0);
            let sub = BatchedMatcher::<Dcfr, Local, L>::new(1, 3, DiscountParams::RECOMMENDED);
            sub.seed(|a, _| [seed[0], seed[1], seed[3]][a], 0);
            let mut scratch = Scratch::new(4);
            let mut state = 0xdead_beef_0000_1111u64;
            for _ in 0..100 {
                let r: Vec<f32> = (0..4).map(|_| next_reward(&mut state)).collect();
                let ev_m = masked.update_row_masked_with(&mut scratch, 0, |a| r[a], |a| a != 2);
                let ev_s = sub.update_row_with(&mut scratch, 0, |a| [r[0], r[1], r[3]][a]);
                assert!((ev_m - ev_s).abs() < ev_tol, "expected {ev_m} vs {ev_s}");
            }
            let mut out = vec![0.0f32; 4];
            masked.regret_into(0, &mut out);
            (sub, out)
        }

        // F32Full: inactive cell bit-identical to its seed; others match the
        // subgame to the tolerance of the renormalised expected value.
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 4, DiscountParams::RECOMMENDED);
        let (sub, out) = run(&m, 1e-5);
        assert_eq!(out[2].to_bits(), (-5.0f32).to_bits());
        let mut sub_r = [0.0f32; 3];
        sub.regret_into(0, &mut sub_r);
        for (i, sub_i) in [0usize, 1, 3].into_iter().enumerate() {
            assert!(
                (out[sub_i] - sub_r[i]).abs() < 1e-4 * sub_r[i].abs().max(1.0),
                "action {sub_i}: {} vs subgame {}",
                out[sub_i],
                sub_r[i]
            );
        }

        // Int32Full: inactive cell code-identical to its seed code; the others
        // track the subgame within the rounding random walk (the two matchers
        // draw different stochastic-rounding slots for the same action, so
        // per-tick expected values agree only to the 0.01 quantum's effect).
        let m = BatchedMatcher::<Dcfr, Local, Int32Full>::new(1, 4, DiscountParams::RECOMMENDED);
        let mut seed_code = [0i32; 4];
        m.seed(|a, _| [0.5, -0.25, -5.0, 0.75][a], 0);
        m.regret_lane().codes_row(0, 4, &mut seed_code);
        let (sub, out) = run(&m, 5e-2);
        let mut codes = [0i32; 4];
        m.regret_lane().codes_row(0, 4, &mut codes);
        assert_eq!(codes[2], seed_code[2]);
        assert_eq!(codes[2], -500);
        sub.regret_into(0, &mut sub_r);
        for (i, sub_i) in [0usize, 1, 3].into_iter().enumerate() {
            assert!(
                (out[sub_i] - sub_r[i]).abs() < 0.5,
                "action {sub_i}: {} vs subgame {}",
                out[sub_i],
                sub_r[i]
            );
        }
    }

    #[test]
    fn masked_expected_renormalises_over_active() {
        let mut scratch = Scratch::new(4);
        // σ = [0.75, 0.25, 0, 0]; action 1 masked ⇒ expected = 0.75·2 / 0.75.
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 4, DiscountParams::RECOMMENDED);
        m.seed(|a, _| [3.0, 1.0, 0.0, 0.0][a], 0);
        let ev =
            m.update_row_masked_with(&mut scratch, 0, |a| [2.0, 100.0, 4.0, 6.0][a], |a| a != 1);
        assert!((ev - 2.0).abs() < 1e-6, "{ev}");

        // No active action carries mass ⇒ plain mean of the active rewards.
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 4, DiscountParams::RECOMMENDED);
        m.seed(|a, _| [3.0, 1.0, -1.0, -1.0][a], 0);
        let ev =
            m.update_row_masked_with(&mut scratch, 0, |a| [2.0, 100.0, 4.0, 6.0][a], |a| a >= 2);
        assert!((ev - 5.0).abs() < 1e-6, "{ev}");

        // Nothing active: 0.0, the clock still advances, nothing is written.
        let ev = m.update_row_masked_with(&mut scratch, 0, |_| unreachable!(), |_| false);
        assert_eq!(ev, 0.0);
        assert_eq!(m.num_updates(), 2);
        let mut r = [0.0f32; 4];
        m.regret_into(0, &mut r);
        assert_eq!(r[0], 3.0, "never active: seed kept ({r:?})");
        assert_eq!(r[1], 1.0, "never active: seed kept ({r:?})");
        assert!(
            r[2] < -1.0 && r[3] > 0.0,
            "active once, then untouched: {r:?}"
        );

        // The value accessor is never consulted for an inactive action.
        let m = BatchedMatcher::<Dcfr, Local>::new(1, 4, DiscountParams::RECOMMENDED);
        m.update_row_masked_with(
            &mut scratch,
            0,
            |a| {
                assert_ne!(a, 3, "value called for masked action");
                1.0
            },
            |a| a != 3,
        );

        // Batch form: per-row masks and rewards.
        let m = BatchedMatcher::<Dcfr, Local>::new(2, 4, DiscountParams::RECOMMENDED);
        m.seed(|a, _| [3.0, 1.0, 0.0, 0.0][a], 0);
        let mut ev = [0.0f32; 2];
        m.update_batch_masked_with(
            &mut scratch,
            |a, _| [2.0, 100.0, 4.0, 6.0][a],
            |a, row| if row == 0 { a != 1 } else { a != 0 },
            &mut ev,
        );
        assert!((ev[0] - 2.0).abs() < 1e-6, "{ev:?}");
        assert!((ev[1] - 100.0).abs() < 1e-6, "{ev:?}");
        assert_eq!(m.num_updates(), 1);
    }

    #[test]
    fn masked_update_is_identity_when_all_active() {
        fn check<R: UpdateRule>(params: R::Params) {
            let a = BatchedMatcher::<R, Local>::new(1, 4, params.clone());
            let b = BatchedMatcher::<R, Local>::new(1, 4, params);
            a.seed(|i, _| [0.5, 0.0, 0.25, 0.1][i], 3);
            b.seed(|i, _| [0.5, 0.0, 0.25, 0.1][i], 3);
            let mut scratch = Scratch::new(4);
            let mut state = 0x0bad_cafe_f00d_1234u64;
            for _ in 0..100 {
                let r: Vec<f32> = (0..4).map(|_| next_reward(&mut state)).collect();
                let ev_a = a.update_row_with(&mut scratch, 0, |i| r[i]);
                let ev_b = b.update_row_masked_with(&mut scratch, 0, |i| r[i], |_| true);
                assert_eq!(ev_a.to_bits(), ev_b.to_bits(), "expected values diverged");
                assert_bits("regret", &b.raw_regret(0), &a.raw_regret(0));
                assert_bits("strategy", &b.raw_strategy(0), &a.raw_strategy(0));
            }
            let (mut ca, mut cb) = ([0.0f32; 4], [0.0f32; 4]);
            a.current_into(0, &mut ca);
            b.current_into(0, &mut cb);
            assert_bits("current", &cb, &ca);
        }
        check::<Dcfr>(DiscountParams::RECOMMENDED);
        check::<PdcfrPlus>(PdcfrPlus::RECOMMENDED); // exercises the last_inst lane
    }

    #[test]
    fn masked_update_keeps_last_inst_for_inactive() {
        let m = BatchedMatcher::<PdcfrPlus, Local>::new(1, 3, PdcfrPlus::RECOMMENDED);
        let mut scratch = Scratch::new(3);
        m.update_row_with(&mut scratch, 0, |a| [1.0, -1.0, 0.5][a]);
        let before = m.li_load(2);
        m.update_row_masked_with(&mut scratch, 0, |a| [0.3, 0.9, 0.0][a], |a| a != 2);
        assert_eq!(m.li_load(2).to_bits(), before.to_bits());
        assert_ne!(m.li_load(0).to_bits(), before.to_bits());
    }
}

/// Thread-filtered allocation counting: the global allocator counts only
/// while the current thread has opted in, so unrelated tests running in the
/// same process (plain `cargo test`) cannot perturb the count.
#[cfg(test)]
mod alloc_tests {
    use super::*;
    use crate::discount::DiscountParams;
    use crate::lane::HalfStrategyShared;
    use crate::rules::Dcfr;
    use crate::storage::Atomic;
    use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
    use std::cell::Cell;

    thread_local! {
        // const-initialized TLS: no lazy allocation inside the allocator.
        // `Some(n)` means this thread is counting; keeping the count itself
        // thread-local stops two concurrent `allocations_in` calls from
        // adding to each other's total.
        static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
    }

    struct CountingAlloc;

    fn note_alloc() {
        // try_with: TLS may be unavailable during thread teardown.
        let _ = COUNT.try_with(|c| {
            if let Some(n) = c.get() {
                c.set(Some(n + 1));
            }
        });
    }

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
            note_alloc();
            unsafe { System.alloc(l) }
        }
        unsafe fn alloc_zeroed(&self, l: AllocLayout) -> *mut u8 {
            note_alloc();
            unsafe { System.alloc_zeroed(l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: AllocLayout, n: usize) -> *mut u8 {
            note_alloc();
            unsafe { System.realloc(p, l, n) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
            unsafe { System.dealloc(p, l) }
        }
    }

    #[global_allocator]
    static COUNTING_ALLOC: CountingAlloc = CountingAlloc;

    fn allocations_in(f: impl FnOnce()) -> usize {
        COUNT.with(|c| c.set(Some(0)));
        f();
        COUNT.with(|c| c.replace(None)).unwrap_or(0)
    }

    #[test]
    fn counting_harness_detects_allocations() {
        // Negative control: a plain Vec allocation must register, otherwise
        // the zero-alloc assertion below would pass vacuously.
        let n = allocations_in(|| {
            let v = vec![0u8; 128];
            std::hint::black_box(&v);
        });
        assert!(n > 0, "counting allocator failed to observe an allocation");
    }

    #[test]
    fn hot_path_is_allocation_free() {
        // The exact consumer profile: Dcfr, Atomic, HalfStrategyShared.
        let m = BatchedMatcher::<Dcfr, Atomic, HalfStrategyShared>::new(
            169,
            3,
            DiscountParams::new(3.0, 0.0, 20.0),
        );
        let mut scratch = Scratch::new(4);
        let mut expected = vec![0.0f32; 169];
        let mut out = [0.0f32; 3];
        // Warm-up outside the counted section.
        m.update_batch_with(&mut scratch, |a, _| [1.0, -0.5, 0.25][a], &mut expected);
        m.current_into(0, &mut out);
        let n = allocations_in(|| {
            for _ in 0..10 {
                m.update_batch_with(&mut scratch, |a, _| [1.0, -0.5, 0.25][a], &mut expected);
                for row in 0..169 {
                    m.current_into(row, &mut out);
                }
            }
        });
        assert_eq!(n, 0, "hot path performed {n} heap allocations");
    }
}
