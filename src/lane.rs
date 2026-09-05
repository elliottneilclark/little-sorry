//! Pluggable lane stores for regret and strategy accumulation.
//!
//! A *lane* is a flat array of per-action values owned by one information set
//! (a "row"). The two traits here separate *what* is stored from *how* the
//! caller will use it:
//!
//! - [`RegretLane`] — stores cumulative regret; the matcher computes the new
//!   regret via its [`crate::update_rule::UpdateRule`] and hands the row to
//!   `write_row`. Readout reverses the encoding into plain `f32`.
//! - [`StrategyLane`] — owns the accumulation recurrence
//!   `X ← X·discount + weight·x`  and the normalization-on-read that turns the
//!   accumulated sum into an average strategy.
//!
//! Each trait is parameterized over a [`crate::storage::StorageBackend`], so the
//! same arithmetic runs over both [`crate::storage::Local`] (single-threaded,
//! zero-overhead) and [`crate::storage::Atomic`] (lock-free, `Sync`).
//!
//! The [`Layout`] combinator pairs a regret lane store with a strategy lane
//! store and is the single type parameter `BatchedMatcher` accepts for
//! pluggable memory layouts. [`F32Full`] is the default layout (f32 regret +
//! f32 strategy), reproducing the original all-f32 behavior. [`Int32Regret`]
//! is the supported 4-byte regret lane for large solves (fixed-point with a
//! caller-chosen quantum and an optional pruning floor), and [`NoStrategy`]
//! drops the average lane entirely for layouts whose average is a mean of
//! periodic `current_into` snapshots.

use crate::storage::{AccumCell, StorageBackend};
use crate::update_rule::UpdateRule;

// ── Traits ───────────────────────────────────────────────────────────────────

/// Cumulative-regret lane. The matcher computes new regret via `UpdateRule`
/// (unchanged math) and hands the row to `write_row`; the store owns only the
/// representation. `read_row` returns the stored regret as f32.
///
/// A lane whose representation is wider than f32 (the int32 lane) cannot get
/// its precision benefit through `read_row → f32 arithmetic → write_row`: once
/// a row's magnitude passes ~2²⁴ quanta the f32 add has already rounded a small
/// increment away before `write_row` sees it. Such lanes override
/// [`accumulate_row`](Self::accumulate_row), which receives the update as a
/// `(discount, increment)` pair per action and applies it in the lane's own
/// precision.
pub trait RegretLane<B: StorageBackend>: Sized {
    /// Construction parameters; `()` for lanes that need none.
    type Config: Clone + Default;

    /// Whether a cell whose value is written back unchanged round-trips to the
    /// identical stored code. Required by the matcher's masked updates, which
    /// leave inactive actions' regret untouched by re-writing what they read;
    /// `false` for [`Int16Regret`], whose per-row rescale moves every code.
    const MASK_EXACT: bool = true;

    fn new(num_rows: usize, num_actions: usize, config: Self::Config) -> Self;
    fn read_row(&self, row: usize, num_actions: usize, out: &mut [f32]);
    /// Store a whole row. `update_count` is the matcher's shared clock at this
    /// write; lanes that round stochastically key their draw on
    /// `(row, update_count)` exactly as [`U16AvgStrategy`] does.
    fn write_row(&self, row: usize, num_actions: usize, regret: &[f32], update_count: usize);

    /// Apply one tick's regret update to a row. `regret` is the new row the
    /// rule computed in f32 (what `write_row` would store); a wider lane updates
    /// it to the exact values it stored. `term(action, old)` returns the same
    /// update split as `(discount, increment)` with
    /// `new = [old · discount + increment]`, clamped at zero when
    /// `floor_at_zero`, or `None` to leave that action's cell exactly as
    /// stored. The default stores `regret`; lanes wider than f32 apply the
    /// split in their own precision instead.
    ///
    fn accumulate_row(
        &self,
        row: usize,
        num_actions: usize,
        regret: &mut [f32],
        term: impl Fn(usize, f32) -> Option<(f32, f32)>,
        floor_at_zero: bool,
        update_count: usize,
    ) {
        let _ = (term, floor_at_zero);
        self.write_row(row, num_actions, regret, update_count);
    }

    /// The floor applied to stored regret, if the representation has one;
    /// `None` for unbounded stores. Callers use it to reason about pruning.
    fn floor(&self) -> Option<f32> {
        None
    }
}

/// Cumulative/average-strategy lane. Owns its accumulation recurrence and its
/// average readout; `accumulate` consumes the rule's `(discount, weight)` via
/// `R::strategy_accumulation`.
pub trait StrategyLane<R: UpdateRule, B: StorageBackend>: Sized {
    /// Whether `average_into` is meaningful. `false` only for [`NoStrategy`],
    /// which stores nothing; `BatchedMatcher::average_into` refuses such a
    /// layout instead of returning a made-up distribution.
    const HAS_AVERAGE: bool = true;

    fn new(num_rows: usize, num_actions: usize) -> Self;
    fn accumulate(
        &self,
        row: usize,
        num_actions: usize,
        step: &R::Step,
        strategy: &[f32],
        update_count: usize,
    );
    fn average_into(&self, row: usize, num_actions: usize, out: &mut [f32]);
    /// Zero every cell that stores the accumulated average/strategy — every σ̄
    /// cell and, for `U16AvgStrategy`, every per-row weight `W`. Leaves
    /// cumulative regret, the current strategy, and the clock untouched.
    fn reset(&self);
}

// ── f32 stores ───────────────────────────────────────────────────────────────

/// f32 regret stored as the bit pattern in a u32 word (exact round-trip).
pub struct F32Regret<B: StorageBackend> {
    cells: Vec<B::Cell<u32>>,
}

impl<B: StorageBackend> RegretLane<B> for F32Regret<B> {
    type Config = ();
    fn new(num_rows: usize, num_actions: usize, (): ()) -> Self {
        Self {
            cells: (0..num_rows * num_actions)
                .map(|_| B::Cell::<u32>::default())
                .collect(),
        }
    }
    fn read_row(&self, row: usize, n: usize, out: &mut [f32]) {
        for (i, slot) in out[..n].iter_mut().enumerate() {
            *slot = f32::from_bits(self.cells[row * n + i].load());
        }
    }
    fn write_row(&self, row: usize, n: usize, regret: &[f32], _update_count: usize) {
        for (i, &v) in regret[..n].iter().enumerate() {
            self.cells[row * n + i].store(v.to_bits());
        }
    }
    /// Stores every cell `term` admits and leaves the rest untouched. The
    /// unmasked path's `term` is unconditionally `Some`, so this reduces to
    /// `write_row` once inlined; only a real mask pays for the check.
    fn accumulate_row(
        &self,
        row: usize,
        n: usize,
        regret: &mut [f32],
        term: impl Fn(usize, f32) -> Option<(f32, f32)>,
        _floor_at_zero: bool,
        _update_count: usize,
    ) {
        for (i, &value) in regret[..n].iter().enumerate() {
            let cell = &self.cells[row * n + i];
            if term(i, f32::from_bits(cell.load())).is_some() {
                cell.store(value.to_bits());
            }
        }
    }
}

/// f32 cumulative strategy: `X ← X·discount + weight·x`, normalize on read.
pub struct F32SumStrategy<B: StorageBackend> {
    cells: Vec<B::Cell<u32>>,
}

impl<B: StorageBackend> F32SumStrategy<B> {
    /// Construct a new lane without needing to name the rule type.
    pub(crate) fn new(num_rows: usize, num_actions: usize) -> Self {
        Self {
            cells: (0..num_rows * num_actions)
                .map(|_| B::Cell::<u32>::default())
                .collect(),
        }
    }
    #[inline]
    fn load(&self, idx: usize) -> f32 {
        f32::from_bits(self.cells[idx].load())
    }
    #[inline]
    fn store(&self, idx: usize, v: f32) {
        self.cells[idx].store(v.to_bits());
    }
}

impl<R: UpdateRule, B: StorageBackend> StrategyLane<R, B> for F32SumStrategy<B> {
    fn new(num_rows: usize, num_actions: usize) -> Self {
        F32SumStrategy::new(num_rows, num_actions)
    }
    fn accumulate(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        strategy: &[f32],
        _update_count: usize,
    ) {
        let (discount, weight) = R::strategy_accumulation(step);
        for (i, &s) in strategy[..n].iter().enumerate() {
            let idx = row * n + i;
            self.store(idx, self.load(idx) * discount + weight * s);
        }
    }
    fn average_into(&self, row: usize, n: usize, out: &mut [f32]) {
        for (i, slot) in out[..n].iter_mut().enumerate() {
            *slot = self.load(row * n + i);
        }
        crate::probability::normalize_inplace(&mut out[..n]);
    }
    fn reset(&self) {
        self.reset_cells();
    }
}

/// Inherent forwarders so callers can write `lane.accumulate::<Rule>(...)` with
/// a turbofish rather than spelling out the fully-qualified trait path.
#[allow(dead_code)] // test-only: the matcher drives the `StrategyLane` trait directly
impl<B: StorageBackend> F32SumStrategy<B> {
    /// Zero every accumulated-sum cell. Rule-independent; used by
    /// `StrategyLane::reset` and `BatchedMatcher::reset_average`.
    pub(crate) fn reset_cells(&self) {
        for cell in &self.cells {
            cell.store(0u32); // 0u32 == 0.0f32 bits
        }
    }

    pub(crate) fn accumulate<R: UpdateRule>(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        s: &[f32],
        update_count: usize,
    ) {
        <Self as StrategyLane<R, B>>::accumulate(self, row, n, step, s, update_count);
    }
    pub(crate) fn average_into<R: UpdateRule>(&self, row: usize, n: usize, out: &mut [f32]) {
        <Self as StrategyLane<R, B>>::average_into(self, row, n, out);
    }
}

#[cfg(test)]
impl<B: StorageBackend> F32SumStrategy<B> {
    /// Return the raw (un-normalized) accumulated value for cell `(row, i)`.
    /// Used by Task 4's golden bit-tests.
    pub(crate) fn strategy_raw_cell(&self, row: usize, i: usize, num_actions: usize) -> f32 {
        self.load(row * num_actions + i)
    }
}

// ── u16 bounded-average store ────────────────────────────────────────────────

/// Strategy lane storing the bounded running average σ̄ ∈ \[0,1\] in u16, with one
/// f32 weight `W` per row driving the incremental update — derived from the f32
/// `sum_p/Σsum_p` recurrence: `W ← discount·W + weight`,
/// `σ̄ += (weight/W)·(σ − σ̄)`. Always in \[0,1\] (convex), so u16 fixed-point is
/// safe. `average_into` returns σ̄ directly (one normalize repairs rounding drift).
/// Encoding uses **stochastic rounding** keyed by `(row, update_count)` — one
/// hash per row per tick yields an independent 16-bit draw per action (see
/// `unit_fixed::RowDraws`) — so sub-quantum increments survive in expectation
/// and the average does not freeze under high-γ long-horizon DCFR (see
/// `unit_fixed::encode_stochastic`).
pub struct U16AvgStrategy<B: StorageBackend> {
    cells: Vec<B::Cell<u16>>,
    weight: Vec<B::Cell<u32>>, // per-row W, f32 bits
}

impl<B: StorageBackend> U16AvgStrategy<B> {
    const MAX: u32 = u16::MAX as u32;

    /// Construct a new lane without needing to name the rule type.
    pub(crate) fn new(num_rows: usize, num_actions: usize) -> Self {
        Self {
            cells: (0..num_rows * num_actions)
                .map(|_| B::Cell::<u16>::default())
                .collect(),
            weight: (0..num_rows).map(|_| B::Cell::<u32>::default()).collect(),
        }
    }

    #[inline]
    fn sigma(&self, idx: usize) -> f32 {
        crate::unit_fixed::decode(self.cells[idx].load() as u32, Self::MAX)
    }

    #[inline]
    fn set_sigma_stochastic(&self, idx: usize, v: f32, u01: f32) {
        // Safety: encode_stochastic clamps v to [0,1] and clamps the code to
        // 0..=MAX, so the result always fits a u16.
        #[allow(clippy::cast_possible_truncation)]
        self.cells[idx].store(crate::unit_fixed::encode_stochastic(v, Self::MAX, u01) as u16);
    }

    #[inline]
    fn w_load(&self, row: usize) -> f32 {
        f32::from_bits(self.weight[row].load())
    }

    #[inline]
    fn w_store(&self, row: usize, v: f32) {
        self.weight[row].store(v.to_bits());
    }
}

impl<R: UpdateRule, B: StorageBackend> StrategyLane<R, B> for U16AvgStrategy<B> {
    fn new(num_rows: usize, num_actions: usize) -> Self {
        U16AvgStrategy::new(num_rows, num_actions)
    }

    fn accumulate(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        strategy: &[f32],
        update_count: usize,
    ) {
        let (discount, weight) = R::strategy_accumulation(step);
        let w_new = discount * self.w_load(row) + weight;
        self.w_store(row, w_new);
        let frac = if w_new > 0.0 { weight / w_new } else { 0.0 };
        let mut draws = crate::unit_fixed::RowDraws::new(row, update_count);
        for (i, &s) in strategy[..n].iter().enumerate() {
            let idx = row * n + i;
            let cur = self.sigma(idx);
            let v = cur + frac * (s - cur);
            self.set_sigma_stochastic(idx, v, draws.next_u01());
        }
    }

    fn average_into(&self, row: usize, n: usize, out: &mut [f32]) {
        for (i, slot) in out[..n].iter_mut().enumerate() {
            *slot = self.sigma(row * n + i);
        }
        crate::probability::normalize_inplace(&mut out[..n]);
    }

    fn reset(&self) {
        self.reset_cells();
    }
}

/// Inherent forwarders so callers can write `lane.accumulate::<Rule>(...)` with
/// a turbofish rather than spelling out the fully-qualified trait path.
#[allow(dead_code)] // test-only: the matcher drives the `StrategyLane` trait directly
impl<B: StorageBackend> U16AvgStrategy<B> {
    /// Zero every σ̄ cell and every per-row `W` cell. Rule-independent; used by
    /// `StrategyLane::reset` and `BatchedMatcher::reset_average`. Zeroing `W`
    /// is essential: it ensures `frac = weight/(0 + weight) = 1` on the next
    /// `accumulate`, so σ̄ is set to σ₁ exactly rather than being dragged toward
    /// zero by a near-zero fraction.
    pub(crate) fn reset_cells(&self) {
        for cell in &self.cells {
            cell.store(0u16); // 0u16 == 0.0 in fixed-point σ̄ encoding
        }
        for w in &self.weight {
            w.store(0u32); // 0u32 == 0.0f32 bits; zeroing W ensures frac=1 on next accumulate
        }
    }

    pub(crate) fn accumulate<R: UpdateRule>(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        s: &[f32],
        update_count: usize,
    ) {
        <Self as StrategyLane<R, B>>::accumulate(self, row, n, step, s, update_count);
    }

    pub(crate) fn average_into<R: UpdateRule>(&self, row: usize, n: usize, out: &mut [f32]) {
        <Self as StrategyLane<R, B>>::average_into(self, row, n, out);
    }
}

// ── u16 bounded-average store with a single shared weight ────────────────────

/// Strategy lane storing the bounded running average σ̄ ∈ \[0,1\] in u16, with a
/// **single** shared f32 weight `W` driving the incremental update.
///
/// This is a drop-in replacement for [`U16AvgStrategy`] that removes the
/// `num_rows × 4 B` per-row weight vector and replaces it with one shared
/// cell. The saving restores the full ~25% total footprint cut at small action
/// counts (e.g. rs-poker's ~3-action information sets).
///
/// # Contract: `update_batch`-only driving
///
/// The single shared `W` is valid **only** when every row advances on every
/// tick — i.e. the lane is driven exclusively by `update_batch`, which always
/// visits row 0 first, advancing `W` exactly once per tick; rows `1..num_rows`
/// then fold against that same already-advanced `W`. Calling `update_row(r)`
/// on a shared-weight matrix with `r ≠ 0` leaves the per-row average undefined
/// because `W` is only advanced at row 0. If you need independent per-row
/// updates, use [`U16AvgStrategy`] instead.
pub struct U16AvgStrategyShared<B: StorageBackend> {
    cells: Vec<B::Cell<u16>>,
    weight: B::Cell<u32>, // single shared W, f32 bits
}

impl<B: StorageBackend> U16AvgStrategyShared<B> {
    const MAX: u32 = u16::MAX as u32;

    /// Construct a new lane without needing to name the rule type.
    pub(crate) fn new(num_rows: usize, num_actions: usize) -> Self {
        Self {
            cells: (0..num_rows * num_actions)
                .map(|_| B::Cell::<u16>::default())
                .collect(),
            weight: B::Cell::<u32>::default(),
        }
    }

    #[inline]
    fn sigma(&self, idx: usize) -> f32 {
        crate::unit_fixed::decode(self.cells[idx].load() as u32, Self::MAX)
    }

    #[inline]
    fn set_sigma_stochastic(&self, idx: usize, v: f32, u01: f32) {
        // Safety: encode_stochastic clamps v to [0,1] and clamps the code to
        // 0..=MAX, so the result always fits a u16.
        #[allow(clippy::cast_possible_truncation)]
        self.cells[idx].store(crate::unit_fixed::encode_stochastic(v, Self::MAX, u01) as u16);
    }

    #[inline]
    fn w_load(&self) -> f32 {
        f32::from_bits(self.weight.load())
    }

    #[inline]
    fn w_store(&self, v: f32) {
        self.weight.store(v.to_bits());
    }
}

impl<R: UpdateRule, B: StorageBackend> StrategyLane<R, B> for U16AvgStrategyShared<B> {
    fn new(num_rows: usize, num_actions: usize) -> Self {
        U16AvgStrategyShared::new(num_rows, num_actions)
    }

    fn accumulate(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        strategy: &[f32],
        update_count: usize,
    ) {
        let (discount, weight) = R::strategy_accumulation(step);
        // Row 0 advances the shared W once per tick; rows ≠ 0 reuse it.
        if row == 0 {
            let w_new = discount * self.w_load() + weight;
            self.w_store(w_new);
        }
        let w = self.w_load();
        let frac = if w > 0.0 { weight / w } else { 0.0 };
        let mut draws = crate::unit_fixed::RowDraws::new(row, update_count);
        for (i, &s) in strategy[..n].iter().enumerate() {
            let idx = row * n + i;
            let cur = self.sigma(idx);
            let v = cur + frac * (s - cur);
            self.set_sigma_stochastic(idx, v, draws.next_u01());
        }
    }

    fn average_into(&self, row: usize, n: usize, out: &mut [f32]) {
        for (i, slot) in out[..n].iter_mut().enumerate() {
            *slot = self.sigma(row * n + i);
        }
        crate::probability::normalize_inplace(&mut out[..n]);
    }

    fn reset(&self) {
        self.reset_cells();
    }
}

/// Inherent forwarders so callers can write `lane.accumulate::<Rule>(...)` with
/// a turbofish rather than spelling out the fully-qualified trait path.
#[allow(dead_code)] // test-only: the matcher drives the `StrategyLane` trait directly
impl<B: StorageBackend> U16AvgStrategyShared<B> {
    /// Zero every σ̄ cell and the single shared `W` cell. Rule-independent; used
    /// by `StrategyLane::reset` and `BatchedMatcher::reset_average`. Zeroing `W`
    /// ensures `frac = weight/(0 + weight) = 1` on the next `accumulate`, so σ̄
    /// is set to σ₁ exactly rather than being dragged toward zero by a near-zero
    /// fraction.
    pub(crate) fn reset_cells(&self) {
        for cell in &self.cells {
            cell.store(0u16); // 0u16 == 0.0 in fixed-point σ̄ encoding
        }
        self.weight.store(0u32); // 0u32 == 0.0f32 bits; zeroing W ensures frac=1 on next accumulate
    }

    pub(crate) fn accumulate<R: UpdateRule>(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        s: &[f32],
        update_count: usize,
    ) {
        <Self as StrategyLane<R, B>>::accumulate(self, row, n, step, s, update_count);
    }

    pub(crate) fn average_into<R: UpdateRule>(&self, row: usize, n: usize, out: &mut [f32]) {
        <Self as StrategyLane<R, B>>::average_into(self, row, n, out);
    }
}

// ── Layout combinator ────────────────────────────────────────────────────────

/// A memory layout: an independent choice of regret and strategy lane store.
pub trait Layout<R: UpdateRule, B: StorageBackend> {
    type Regret: RegretLane<B>;
    type Strategy: StrategyLane<R, B>;
}

/// Default layout — f32 regret + f32 strategy == today's behavior.
pub struct F32Full;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for F32Full {
    type Regret = F32Regret<B>;
    type Strategy = F32SumStrategy<B>;
}

/// f32 regret + u16 bounded-average strategy. ~25% total footprint cut.
pub struct HalfStrategy;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for HalfStrategy {
    type Regret = F32Regret<B>;
    type Strategy = U16AvgStrategy<B>;
}

// ── i16 scaled regret store ───────────────────────────────────────────────────

/// Cumulative regret as per-row-scaled i16. One f32 scale per row; `write_row`
/// recomputes the scale from the row's current regret, so growth that would
/// overflow simply enlarges the scale (Cepheus-style, adapted to per-step
/// requantize — lossy for DCFR's split α/β discount; gated behind the
/// exploitability-equivalence test, not assumed).
///
/// ## β = 0 and the int16 lossiness regime
///
/// DCFR's negative-regret discount factor is `t^β / (t^β + 1)`. At β = 0 this
/// collapses to the constant `1/(1+1) = 0.5`, independent of step `t`. This
/// does **not** restore integer-exact accumulation: the positive lane still
/// carries the per-step α schedule, and negatives still scale by 0.5 each step,
/// so the lane stays in the requantize-each-step regime (lossy by per-step
/// rounding, not by swamping). β = 0 is favorable, though: negatives decay by
/// half each step, staying small relative to the positive peak that sets the
/// per-row scale, and regret-matching clamps negatives at 0 — so precision
/// concentrates exactly where strategy quality is decided. Both
/// `DiscountParams::RECOMMENDED = (1.5, 0, 2)` and rs-poker's `(2.3, 0, 10)`
/// use β = 0. Convergence equivalence to the f32 baseline has been validated
/// empirically by single-thread (`Local`) and concurrent (`Atomic`) tests at
/// both parameter sets.
///
/// ## Status: rejected for production solves
///
/// rs-poker's paired out-of-sample exploitability A/B (`HalfBothShared` vs
/// `HalfStrategyShared`, same seeds and iteration budget) measured a **45×
/// exploitability regression** for this lane: the per-row rescale loses exactly
/// the small regret differences that decide marginal hands. It stays available
/// for experiments, but the supported 4-byte regret lane is [`Int32Regret`],
/// whose quantum is fixed rather than tied to the row's peak. This lane also
/// does not support masked updates ([`RegretLane::MASK_EXACT`] is `false`).
pub struct Int16Regret<B: StorageBackend> {
    cells: Vec<B::Cell<u16>>, // i16 bits via `as u16` / `as i16`
    scale: Vec<B::Cell<u32>>, // per-row f32 scale bits
}

impl<B: StorageBackend> Int16Regret<B> {
    #[inline]
    fn scale_load(&self, row: usize) -> f32 {
        f32::from_bits(self.scale[row].load())
    }
    #[inline]
    fn scale_store(&self, row: usize, s: f32) {
        self.scale[row].store(s.to_bits());
    }
    #[inline]
    fn code(&self, idx: usize) -> i16 {
        // Safety: i16 bits are stored as u16; reinterpret via `as i16` is
        // lossless (both are 16-bit; the bit pattern is preserved exactly).
        #[allow(clippy::cast_possible_truncation)]
        let v = self.cells[idx].load() as i16;
        v
    }
    #[inline]
    fn set_code(&self, idx: usize, q: i16) {
        // Safety: i16 → u16 via `as` reinterprets 16 bits, no truncation.
        #[allow(clippy::cast_possible_truncation)]
        self.cells[idx].store(q as u16);
    }
}

impl<B: StorageBackend> RegretLane<B> for Int16Regret<B> {
    type Config = ();
    const MASK_EXACT: bool = false;

    fn new(num_rows: usize, num_actions: usize, (): ()) -> Self {
        Self {
            cells: (0..num_rows * num_actions)
                .map(|_| B::Cell::<u16>::default())
                .collect(),
            scale: (0..num_rows).map(|_| B::Cell::<u32>::default()).collect(),
        }
    }

    fn read_row(&self, row: usize, n: usize, out: &mut [f32]) {
        let s = self.scale_load(row);
        let s = if s > 0.0 { s } else { 0.0 }; // fresh row: scale 0 ⇒ all-zero regret
        for (i, slot) in out[..n].iter_mut().enumerate() {
            *slot = crate::scaled_int::decode(self.code(row * n + i), s);
        }
    }

    fn write_row(&self, row: usize, n: usize, regret: &[f32], _update_count: usize) {
        let s = crate::scaled_int::choose_scale(&regret[..n]);
        self.scale_store(row, s);
        for (i, &r) in regret[..n].iter().enumerate() {
            self.set_code(row * n + i, crate::scaled_int::encode(r, s));
        }
    }
}

// ── i16 regret layouts ────────────────────────────────────────────────────────

/// i16 regret + f32 strategy.
pub struct HalfRegret;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for HalfRegret {
    type Regret = Int16Regret<B>;
    type Strategy = F32SumStrategy<B>;
}

/// i16 regret + u16 strategy — the deepest layout (~10.5 GB target).
pub struct HalfBoth;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for HalfBoth {
    type Regret = Int16Regret<B>;
    type Strategy = U16AvgStrategy<B>;
}

/// f32 regret + shared-weight u16 strategy. Like [`HalfStrategy`] but uses
/// [`U16AvgStrategyShared`], removing the `num_rows × 4 B` per-row weight
/// vector. Valid only when driven exclusively by `update_batch` (see
/// [`U16AvgStrategyShared`] for the contract).
pub struct HalfStrategyShared;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for HalfStrategyShared {
    type Regret = F32Regret<B>;
    type Strategy = U16AvgStrategyShared<B>;
}

/// i16 regret + shared-weight u16 strategy — the deepest shared-weight layout.
/// Like [`HalfBoth`] but uses [`U16AvgStrategyShared`], removing the
/// `num_rows × 4 B` per-row weight vector. Valid only when driven exclusively
/// by `update_batch` (see [`U16AvgStrategyShared`] for the contract).
pub struct HalfBothShared;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for HalfBothShared {
    type Regret = Int16Regret<B>;
    type Strategy = U16AvgStrategyShared<B>;
}

// ── i32 fixed-point regret store ─────────────────────────────────────────────

/// Construction parameters for [`Int32Regret`].
///
/// ```
/// use little_sorry::{Int32Config, Int32Regret, Local, RegretLane};
///
/// let cfg = Int32Config { scale: 1000.0, ..Int32Config::default() }; // 0.001 quantum
/// let lane = Int32Regret::<Local>::new(1, 1, cfg);
/// assert_eq!(lane.floor(), Some(i32::MIN as f32 / 1000.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Int32Config {
    /// Codes per unit of regret: stored code = round(regret × scale). Default
    /// `100.0` — a 0.01 quantum in the caller's regret unit, range ±2.1e7 units.
    pub scale: f32,
    /// Minimum stored regret in regret units (not codes); writes below it are
    /// clamped to it. `f32::NEG_INFINITY` (the default) uses the
    /// representation's own minimum, i.e. no pruning floor. Callers that prune
    /// derive a real floor with [`dominated_regret_after`](crate::dominated_regret_after)
    /// — never a guessed constant.
    pub floor: f32,
}

impl Default for Int32Config {
    fn default() -> Self {
        Self {
            scale: 100.0,
            floor: f32::NEG_INFINITY,
        }
    }
}

/// Cumulative regret as fixed-point i32 at a caller-chosen scale, with a floor.
///
/// This is Pluribus's regret layout (int32 regrets, Science 2019 supplement
/// pp. 13–15) in the same 4 bytes as f32, and it exists because f32 stops
/// learning at large magnitude: with a 24-bit mantissa, once a row's cumulative
/// regret reaches 10⁶ an increment below ~0.06 is rounded away, and the row no
/// longer tracks the small regrets that decide marginal decisions. Fixed-point
/// keeps a *constant* quantum (`1/scale`) all the way to ±2³¹ codes.
///
/// Three properties do the work:
///
/// - **Code-space accumulation.** The matcher drives this lane through
///   [`RegretLane::accumulate_row`], which folds the rule's `(discount,
///   increment)` split into the stored code in f64 (`code·d + inc·scale`).
///   Going through `read_row → f32 add → write_row` would round the increment
///   away exactly as f32 storage does; see the trait docs.
/// - **Stochastic rounding** keyed on `(row, update_count)` through
///   `unit_fixed::RowDraws`, so sub-quantum increments survive in expectation —
///   the same reason the u16 average lane rounds stochastically; a
///   deterministic round would freeze a row whose per-tick increment is below
///   the quantum.
/// - **A floor**, applied on every write and every accumulate, so a pruned
///   action's regret cannot fall so far that it can never recover, and a raced
///   write under the `Atomic` backend cannot go below it either. Saturates,
///   never wraps.
///
/// Readout is `code as f32 / scale`, which for |code| > 2²⁴ carries only f32
/// precision — fine for the strategy (a normalized ratio) but not for
/// checkpoints, which use the value-exact [`codes_row`](Self::codes_row) /
/// [`set_codes_row`](Self::set_codes_row) instead.
///
/// ```
/// use little_sorry::{Int32Config, Int32Regret, Local, RegretLane};
///
/// let lane = Int32Regret::<Local>::new(1, 2, Int32Config { scale: 100.0, floor: -5.0 });
/// lane.write_row(0, 2, &[1.234, -50.0], 0);
/// let mut out = [0.0f32; 2];
/// lane.read_row(0, 2, &mut out);
/// assert!((out[0] - 1.234).abs() <= 0.005, "within half a quantum");
/// assert_eq!(out[1], -5.0, "clamped to the floor");
/// assert_eq!(lane.floor(), Some(-5.0));
/// ```
pub struct Int32Regret<B: StorageBackend> {
    cells: Vec<B::Cell<u32>>, // i32 bit patterns
    scale: f32,
    floor_code: i32,
}

impl<B: StorageBackend> Int32Regret<B> {
    #[inline]
    fn code(&self, idx: usize) -> i32 {
        // i32 bits are stored as u32; `as i32` reinterprets the 32-bit pattern
        // exactly.
        #[allow(clippy::cast_possible_wrap)]
        let v = self.cells[idx].load() as i32;
        v
    }

    #[inline]
    fn set_code(&self, idx: usize, code: i32) {
        // i32 → u32 via `as` reinterprets 32 bits, no truncation.
        #[allow(clippy::cast_sign_loss)]
        self.cells[idx].store(code as u32);
    }

    /// Stochastically round a scaled (code-space) value to the nearest code on
    /// the side chosen by `u01`, then clamp to `[floor_code, i32::MAX]` and, if
    /// asked, at zero. `as i32` saturates at the type bounds and maps NaN to 0,
    /// so out-of-range values clamp rather than wrap.
    #[inline]
    fn quantize(&self, scaled: f64, u01: f32, floor_at_zero: bool) -> i32 {
        let base = scaled.floor();
        let frac = scaled - base;
        #[allow(clippy::cast_possible_truncation)]
        let code = (base as i32).saturating_add(i32::from(f64::from(u01) < frac));
        let code = code.max(self.floor_code);
        if floor_at_zero { code.max(0) } else { code }
    }

    /// Codes per regret unit this lane was built with.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Copy a row's raw codes into `out` — value-exact, for checkpoints.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < num_actions`.
    pub fn codes_row(&self, row: usize, num_actions: usize, out: &mut [i32]) {
        for (i, slot) in out[..num_actions].iter_mut().enumerate() {
            *slot = self.code(row * num_actions + i);
        }
    }

    /// Overwrite a row's raw codes from `codes` — value-exact, for restoring a
    /// checkpoint written by [`codes_row`](Self::codes_row) under the same
    /// [`Int32Config`]. No rounding and no floor: a code below the floor is
    /// stored as given and clamped on its next update.
    ///
    /// # Panics
    ///
    /// Panics if `codes.len() < num_actions`.
    pub fn set_codes_row(&self, row: usize, num_actions: usize, codes: &[i32]) {
        for (i, &c) in codes[..num_actions].iter().enumerate() {
            self.set_code(row * num_actions + i, c);
        }
    }
}

impl<B: StorageBackend> RegretLane<B> for Int32Regret<B> {
    type Config = Int32Config;

    /// # Panics
    ///
    /// Panics if `config.scale` is not a positive finite number.
    fn new(num_rows: usize, num_actions: usize, config: Int32Config) -> Self {
        assert!(
            config.scale.is_finite() && config.scale > 0.0,
            "Int32Config::scale must be positive and finite, got {}",
            config.scale
        );
        // `NEG_INFINITY` is the no-floor sentinel. Other out-of-range values
        // saturate to the representation bounds.
        #[allow(clippy::cast_possible_truncation)]
        let floor_code = if config.floor == f32::NEG_INFINITY {
            i32::MIN
        } else {
            (f64::from(config.floor) * f64::from(config.scale)).round() as i32
        };
        Self {
            cells: (0..num_rows * num_actions)
                .map(|_| B::Cell::<u32>::default())
                .collect(),
            scale: config.scale,
            floor_code,
        }
    }

    fn read_row(&self, row: usize, n: usize, out: &mut [f32]) {
        for (i, slot) in out[..n].iter_mut().enumerate() {
            *slot = self.code(row * n + i) as f32 / self.scale;
        }
    }

    fn write_row(&self, row: usize, n: usize, regret: &[f32], update_count: usize) {
        let scale = f64::from(self.scale);
        let mut draws = crate::unit_fixed::RowDraws::new(row, update_count);
        for (i, &r) in regret[..n].iter().enumerate() {
            let code = self.quantize(f64::from(r) * scale, draws.next_u01(), false);
            self.set_code(row * n + i, code);
        }
    }

    fn accumulate_row(
        &self,
        row: usize,
        n: usize,
        regret: &mut [f32],
        term: impl Fn(usize, f32) -> Option<(f32, f32)>,
        floor_at_zero: bool,
        update_count: usize,
    ) {
        let scale = f64::from(self.scale);
        let mut draws = crate::unit_fixed::RowDraws::new(row, update_count);
        for (i, slot) in regret[..n].iter_mut().enumerate() {
            let idx = row * n + i;
            let code = self.code(idx);
            let u01 = draws.next_u01(); // keep the draw stream aligned per action
            let Some((discount, increment)) = term(i, code as f32 / self.scale) else {
                continue;
            };
            let code = self.quantize(
                f64::from(code) * f64::from(discount) + f64::from(increment) * scale,
                u01,
                floor_at_zero,
            );
            self.set_code(idx, code);
            *slot = code as f32 / self.scale;
        }
    }

    fn floor(&self) -> Option<f32> {
        Some(self.floor_code as f32 / self.scale)
    }
}

// ── Zero-byte strategy lane ──────────────────────────────────────────────────

/// A strategy lane that stores nothing. For layouts whose average strategy is
/// not resident — Pluribus keeps no running average for postflop rows; the
/// average is a mean of periodic `current_into` snapshots taken by the caller
/// — so the 4 bytes per cell the average would cost are simply not allocated.
/// `accumulate` and `reset` are no-ops; `average_into` panics, and
/// `BatchedMatcher::average_into` checks [`StrategyLane::HAS_AVERAGE`] first
/// to name the layout in its message.
///
/// ```
/// use little_sorry::{BatchedMatcher, Dcfr, DiscountParams, Int32NoAverage, Local};
///
/// let m = BatchedMatcher::<Dcfr, Local, Int32NoAverage>::new(4, 3, DiscountParams::RECOMMENDED);
/// let mut ev = [0.0f32; 4];
/// let mut current = [0.0f32; 3];
/// for _ in 0..100 {
///     m.update_batch(|a, _| [1.0, -0.5, 0.2][a], &mut ev);
/// }
/// m.current_into(0, &mut current); // the only strategy readout this layout has
/// assert!((current.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// ```
pub struct NoStrategy;

impl<R: UpdateRule, B: StorageBackend> StrategyLane<R, B> for NoStrategy {
    const HAS_AVERAGE: bool = false;

    fn new(_num_rows: usize, _num_actions: usize) -> Self {
        NoStrategy
    }
    fn accumulate(&self, _: usize, _: usize, _: &R::Step, _: &[f32], _: usize) {}
    fn average_into(&self, _: usize, _: usize, _: &mut [f32]) {
        panic!("NoStrategy keeps no average lane; snapshot `current_into` instead");
    }
    fn reset(&self) {}
}

// ── i32 regret layouts ───────────────────────────────────────────────────────

/// i32 fixed-point regret + f32 strategy: 8 bytes per cell, constant regret
/// quantum at any magnitude.
pub struct Int32Full;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for Int32Full {
    type Regret = Int32Regret<B>;
    type Strategy = F32SumStrategy<B>;
}

/// i32 fixed-point regret + shared-weight u16 average: 6 bytes per cell.
/// rs-poker's preflop layout. Valid only when driven exclusively by
/// `update_batch` (see [`U16AvgStrategyShared`] for the contract).
pub struct Int32HalfShared;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for Int32HalfShared {
    type Regret = Int32Regret<B>;
    type Strategy = U16AvgStrategyShared<B>;
}

/// i32 fixed-point regret and no average lane: 4 bytes per cell. rs-poker's
/// postflop layout; the average is the caller's mean of `current_into`
/// snapshots, and `average_into` panics.
pub struct Int32NoAverage;

impl<R: UpdateRule, B: StorageBackend> Layout<R, B> for Int32NoAverage {
    type Regret = Int32Regret<B>;
    type Strategy = NoStrategy;
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discount::DiscountParams;
    use crate::rules::Dcfr;
    use crate::storage::Local;
    use crate::update_rule::UpdateRule;

    #[test]
    fn f32_regret_lane_round_trips_exactly() {
        let lane = F32Regret::<Local>::new(2, 3, ());
        let vals = [1.5f32, -2.0, 1e9];
        lane.write_row(1, 3, &vals, 0);
        let mut out = [0.0f32; 3];
        lane.read_row(1, 3, &mut out);
        for (a, b) in vals.iter().zip(&out) {
            assert_eq!(a.to_bits(), b.to_bits(), "exact f32 round-trip");
        }
        // untouched row reads zero
        lane.read_row(0, 3, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f32_sum_strategy_matches_accumulate_then_normalize() {
        // One Dcfr step: discount/weight from the rule, fold a known strategy,
        // average == normalized accumulator.
        let lane = F32SumStrategy::<Local>::new(1, 3);
        let params = DiscountParams::RECOMMENDED;
        let step = Dcfr::step(&params, 1);
        let strat = [0.2f32, 0.3, 0.5];
        lane.accumulate::<Dcfr>(0, 3, &step, &strat, 1);
        let mut out = [0.0f32; 3];
        lane.average_into::<Dcfr>(0, 3, &mut out);
        // first accumulation: cell = weight*strat, normalize == strat
        for (a, b) in strat.iter().zip(&out) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn u16_average_keeps_moving_to_horizon() {
        use crate::rules::LinearCfr; // (1, t) weighting stresses the horizon
        let lane = U16AvgStrategy::<Local>::new(1, 2);
        let mut last = [0.0f32; 2];
        let mut moved_late = false;
        for t in 1..=20_000usize {
            let step = LinearCfr::step(&(), t);
            // Alternate target so the average must keep tracking.
            let strat = if t % 2 == 0 {
                [0.7f32, 0.3]
            } else {
                [0.3f32, 0.7]
            };
            lane.accumulate::<LinearCfr>(0, 2, &step, &strat, t);
            if t > 15_000 {
                let mut cur = [0.0f32; 2];
                lane.average_into::<LinearCfr>(0, 2, &mut cur);
                if (cur[0] - last[0]).abs() > 1e-6 {
                    moved_late = true;
                }
                last = cur;
            }
        }
        assert!(moved_late, "u16 average froze before the horizon");
    }

    #[test]
    fn u16_avg_matches_f32_sum_average_within_quantum() {
        use crate::rules::Dcfr;
        use crate::storage::Local;
        let params = DiscountParams::RECOMMENDED;
        let f32_lane = F32SumStrategy::<Local>::new(1, 3);
        let u16_lane = U16AvgStrategy::<Local>::new(1, 3);
        // A known sequence of strategies under successive Dcfr steps.
        let seq = [
            [0.5f32, 0.3, 0.2],
            [0.1, 0.8, 0.1],
            [0.33, 0.33, 0.34],
            [0.6, 0.1, 0.3],
        ];
        for (t, strat) in seq.iter().enumerate() {
            let step = Dcfr::step(&params, t + 1);
            f32_lane.accumulate::<Dcfr>(0, 3, &step, strat, t + 1);
            u16_lane.accumulate::<Dcfr>(0, 3, &step, strat, t + 1);
        }
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        f32_lane.average_into::<Dcfr>(0, 3, &mut a);
        u16_lane.average_into::<Dcfr>(0, 3, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!(
                (x - y).abs() < 2.0 / u16::MAX as f32,
                "u16 avg within a couple quanta: {x} vs {y}"
            );
        }
    }

    #[test]
    fn int16_regret_round_trips_within_row_quantum() {
        let lane = Int16Regret::<Local>::new(1, 3, ());
        let regret = [1000.0f32, -250.0, 30.0];
        lane.write_row(0, 3, &regret, 0);
        let mut out = [0.0f32; 3];
        lane.read_row(0, 3, &mut out);
        // scale ≈ 1000/32767; round-trip within ~one quantum of the row peak.
        let s = 1000.0f32 / i16::MAX as f32;
        for (a, b) in regret.iter().zip(&out) {
            assert!((a - b).abs() <= s + 1e-2, "{a} vs {b}");
        }
    }

    #[test]
    fn int16_regret_rescales_on_growth_without_overflow() {
        let lane = Int16Regret::<Local>::new(1, 2, ());
        lane.write_row(0, 2, &[1.0, -1.0], 0); // tiny scale
        lane.write_row(0, 2, &[1.0e6, -5.0e5], 0); // forces a much larger scale
        let mut out = [0.0f32; 2];
        lane.read_row(0, 2, &mut out);
        assert!(
            (out[0] - 1.0e6).abs() / 1.0e6 < 1e-3,
            "rescaled high value preserved: {out:?}"
        );
        assert!(out[1] < 0.0, "sign preserved");
    }

    #[test]
    fn u16_reset_then_first_accumulate_lands_sigma1() {
        let lane = U16AvgStrategy::<Local>::new(1, 3);
        let params = DiscountParams::RECOMMENDED;
        // Run a few accumulate steps to build up state.
        for t in 1..=5usize {
            let step = Dcfr::step(&params, t);
            lane.accumulate::<Dcfr>(0, 3, &step, &[0.5, 0.3, 0.2], t);
        }
        // Reset zeros all σ̄ cells and W (call inherent to avoid rule type ambiguity).
        lane.reset_cells();
        // One accumulate with a known σ1.
        let sigma1 = [0.6f32, 0.25, 0.15];
        let step = Dcfr::step(&params, 6);
        lane.accumulate::<Dcfr>(0, 3, &step, &sigma1, 6);
        // With W zeroed before reset, frac = weight/(0+weight) = 1 → σ̄ = σ1.
        let mut out = [0.0f32; 3];
        lane.average_into::<Dcfr>(0, 3, &mut out);
        let quantum = 2.0 / u16::MAX as f32;
        for (a, b) in sigma1.iter().zip(&out) {
            assert!(
                (a - b).abs() < quantum,
                "σ̄ should equal σ1 within u16 quantum: {a} vs {b}"
            );
        }
    }

    /// Drive the same per-tick strategy through a per-row `U16AvgStrategy` and a
    /// `U16AvgStrategyShared`, simulating `update_batch` order (row 0 first, then
    /// row 1 each tick). Assert that both lanes' `average_into` agree per row
    /// within `2/u16::MAX` per component after several ticks.
    #[test]
    fn shared_weight_matches_per_row_under_batch() {
        let params = DiscountParams::RECOMMENDED;
        let per_row = U16AvgStrategy::<Local>::new(2, 3);
        let shared = U16AvgStrategyShared::<Local>::new(2, 3);

        let strategies = [
            [0.5f32, 0.3, 0.2],
            [0.1f32, 0.8, 0.1],
            [0.4f32, 0.4, 0.2],
            [0.6f32, 0.1, 0.3],
            [0.33f32, 0.33, 0.34],
            [0.2f32, 0.5, 0.3],
            [0.7f32, 0.15, 0.15],
            [0.25f32, 0.5, 0.25],
        ];

        for (t, sigma) in strategies.iter().enumerate() {
            let step = Dcfr::step(&params, t + 1);
            // simulate update_batch: row 0 first, then row 1
            per_row.accumulate::<Dcfr>(0, 3, &step, sigma, t + 1);
            per_row.accumulate::<Dcfr>(1, 3, &step, sigma, t + 1);
            shared.accumulate::<Dcfr>(0, 3, &step, sigma, t + 1);
            shared.accumulate::<Dcfr>(1, 3, &step, sigma, t + 1);
        }

        let quantum = 2.0 / u16::MAX as f32;
        for row in 0..2 {
            let mut a = [0.0f32; 3];
            let mut b = [0.0f32; 3];
            per_row.average_into::<Dcfr>(row, 3, &mut a);
            shared.average_into::<Dcfr>(row, 3, &mut b);
            for (x, y) in a.iter().zip(&b) {
                assert!(
                    (x - y).abs() < quantum,
                    "row {row}: per-row {x} vs shared {y} — diff exceeds u16 quantum"
                );
            }
        }
    }

    /// The load-bearing regression: under DCFR γ=10 over >1M updates, the
    /// stochastic-rounded u16 average must keep tracking the f32 average, where a
    /// round-to-nearest u16 recurrence (computed inline) freezes. This is the
    /// regime rs-poker's exploitability A/B exposed; PR #16 never exercised it.
    #[test]
    fn u16_stochastic_tracks_f32_where_round_to_nearest_freezes() {
        let params = DiscountParams::new(2.3, 0.0, 10.0); // rs-poker's (α, β, γ)
        let max = u16::MAX as u32;
        let quantum = 1.0 / max as f32;

        let f32_lane = F32SumStrategy::<Local>::new(1, 2);
        let u16_lane = U16AvgStrategy::<Local>::new(1, 2);

        // Inline round-to-nearest bounded average for component 0 (pre-fix behavior).
        let mut rn_code: u32 = 0;
        let mut rn_w: f32 = 0.0;

        let n = 1_200_000usize;
        let probe = 700_000usize; // far past the ~67k freeze crossover for γ=10
        let (mut f32_at_probe, mut u16_at_probe, mut rn_at_probe) = (0.0f32, 0.0f32, 0u32);

        for t in 1..=n {
            let step = Dcfr::step(&params, t);
            // Target drifts the whole way, so a correctly-tracking average must keep
            // moving even once per-step corrections fall below the u16 quantum.
            let g = t as f32 / n as f32;
            let sigma = [0.3 + 0.4 * g, 0.7 - 0.4 * g];
            f32_lane.accumulate::<Dcfr>(0, 2, &step, &sigma, t);
            u16_lane.accumulate::<Dcfr>(0, 2, &step, &sigma, t);

            let (discount, weight) = Dcfr::strategy_accumulation(&step);
            rn_w = discount * rn_w + weight;
            let frac = if rn_w > 0.0 { weight / rn_w } else { 0.0 };
            let cur = rn_code as f32 / max as f32;
            rn_code = crate::unit_fixed::encode(cur + frac * (sigma[0] - cur), max);

            if t == probe {
                let mut tmp = [0.0f32; 2];
                f32_lane.average_into::<Dcfr>(0, 2, &mut tmp);
                f32_at_probe = tmp[0];
                u16_lane.average_into::<Dcfr>(0, 2, &mut tmp);
                u16_at_probe = tmp[0];
                rn_at_probe = rn_code;
            }
        }

        let mut f = [0.0f32; 2];
        let mut u = [0.0f32; 2];
        f32_lane.average_into::<Dcfr>(0, 2, &mut f);
        u16_lane.average_into::<Dcfr>(0, 2, &mut u);
        let rn_end = rn_code as f32 / max as f32;

        // f32 ground truth genuinely kept moving past the probe (test is in-regime).
        let f32_late_move = (f[0] - f32_at_probe).abs();
        assert!(
            f32_late_move > 10.0 * quantum,
            "not exercising late movement: {f32_late_move}"
        );

        // Round-to-nearest froze: essentially no movement after the probe.
        let rn_late_move = (rn_end - rn_at_probe as f32 / max as f32).abs();
        assert!(
            rn_late_move < 2.0 * quantum,
            "round-to-nearest unexpectedly moved: {rn_late_move}"
        );

        // Stochastic u16 kept moving and ends closer to f32 than the frozen lane.
        let u16_late_move = (u[0] - u16_at_probe).abs();
        assert!(
            u16_late_move > 5.0 * quantum,
            "stochastic u16 froze: {u16_late_move}"
        );
        assert!(
            (u[0] - f[0]).abs() < (rn_end - f[0]).abs(),
            "stochastic should track f32 better than frozen RN: u16={u:?} f32={f:?} rn={rn_end}"
        );
    }
    // ── Int32Regret ──────────────────────────────────────────────────────────

    #[test]
    fn int32_round_trips_within_one_quantum() {
        // Stochastic rounding lands on the code just below or just above the
        // exact value, so the round-trip error is strictly under one quantum;
        // with the clock fixed the draw is seeded, so the result is
        // deterministic.
        let cfg = Int32Config::default();
        let lane = Int32Regret::<Local>::new(1, 4, cfg);
        let quantum = 1.0 / cfg.scale;
        for &v in &[
            0.0f32, 0.004, -0.004, 1.234, -1.234, 12_345.678, -1e6, 1e6, 999_999.9,
        ] {
            let vals = [v, -v, v * 0.5, v * 2.0];
            lane.write_row(0, 4, &vals, 7);
            let mut out = [0.0f32; 4];
            lane.read_row(0, 4, &mut out);
            for (a, b) in vals.iter().zip(&out) {
                assert!((a - b).abs() < quantum + 1e-6 * a.abs(), "{a} vs {b}");
            }
            let mut again = [0.0f32; 4];
            lane.write_row(0, 4, &vals, 7);
            lane.read_row(0, 4, &mut again);
            assert_eq!(out, again, "same (row, update_count) ⇒ same rounding");
        }
    }

    #[test]
    fn int32_saturates_at_floor_and_max() {
        let cfg = Int32Config {
            scale: 100.0,
            floor: -12.5,
        };
        let lane = Int32Regret::<Local>::new(1, 3, cfg);
        lane.write_row(0, 3, &[-1e9, 1e9, -12.5], 0);
        let mut out = [0.0f32; 3];
        lane.read_row(0, 3, &mut out);
        assert_eq!(
            out[0], -12.5,
            "below the floor reads back exactly the floor"
        );
        assert_eq!(out[1], i32::MAX as f32 / 100.0, "above the max saturates");
        assert_eq!(out[2], -12.5, "the floor itself is representable");
        assert_eq!(lane.floor(), Some(-12.5));

        // No configured floor: the representation's own minimum, no wrap.
        let lane = Int32Regret::<Local>::new(1, 1, Int32Config::default());
        lane.write_row(0, 1, &[-1e12], 0);
        lane.read_row(0, 1, &mut out[..1]);
        assert_eq!(out[0], i32::MIN as f32 / 100.0);
        assert_eq!(lane.floor(), Some(i32::MIN as f32 / 100.0));
        lane.write_row(0, 1, &[f32::NAN], 0);
        lane.read_row(0, 1, &mut out[..1]);
        assert_eq!(out[0], 0.0, "NaN stores zero rather than wrapping");
    }

    #[test]
    fn int32_codes_round_trip_exactly() {
        let lane = Int32Regret::<Local>::new(2, 3, Int32Config::default());
        let codes = [i32::MIN, -1, 0, 1, 123_456_789, i32::MAX];
        lane.set_codes_row(0, 3, &codes[..3]);
        lane.set_codes_row(1, 3, &codes[3..]);
        let mut out = [0i32; 3];
        lane.codes_row(0, 3, &mut out);
        assert_eq!(out, codes[..3]);
        lane.codes_row(1, 3, &mut out);
        assert_eq!(out, codes[3..]);
        assert_eq!(lane.scale(), 100.0);
    }

    /// The motivation, pinned: at magnitude 1e7 an f32 lane cannot absorb a
    /// 0.01 increment at all (its ulp there is 1.0), while the int32 lane —
    /// accumulating in code space — takes every one.
    #[test]
    fn int32_keeps_small_increments_at_large_magnitude() {
        let scale = 100.0f32;
        let start = 1.0e7f32; // code 1e9
        let int32 = Int32Regret::<Local>::new(
            1,
            1,
            Int32Config {
                scale,
                ..Int32Config::default()
            },
        );
        let f32_lane = F32Regret::<Local>::new(1, 1, ());
        int32.write_row(0, 1, &[start], 0);
        f32_lane.write_row(0, 1, &[start], 0);

        // Both lanes are driven exactly as the matcher drives them: the f32
        // result the rule would compute, plus the (discount, increment) split
        // for lanes that accumulate in their own precision.
        fn tick(lane: &impl RegretLane<Local>, t: usize) {
            let mut buf = [0.0f32; 1];
            lane.read_row(0, 1, &mut buf);
            let mut new = [buf[0] + 0.01];
            lane.accumulate_row(0, 1, &mut new, |_, _| Some((1.0, 0.01)), false, t);
        }
        for t in 1..=1_000usize {
            tick(&int32, t);
            tick(&f32_lane, t);
        }
        let mut code = [0i32; 1];
        int32.codes_row(0, 1, &mut code);
        let want = 1_000_000_000i64 + 1_000;
        assert!(
            (i64::from(code[0]) - want).abs() <= 10,
            "int32 within 1% of the accumulated increment: code {} vs {want}",
            code[0]
        );
        let mut buf = [0.0f32; 1];
        f32_lane.read_row(0, 1, &mut buf);
        assert_eq!(buf[0], start, "f32 lane lost every increment");
    }

    #[test]
    fn int32_accumulate_applies_floor_zero_clamp_and_skips_none() {
        let lane = Int32Regret::<Local>::new(
            1,
            3,
            Int32Config {
                scale: 100.0,
                floor: -2.0,
            },
        );
        lane.write_row(0, 3, &[1.0, -1.0, 0.5], 0);
        // action 0: discounted below zero with floor_at_zero ⇒ 0;
        // action 1: pushed far below the floor ⇒ floor; action 2: untouched.
        let mut next = [0.0; 3];
        lane.accumulate_row(
            0,
            3,
            &mut next,
            |i, old| match i {
                0 => {
                    assert_eq!(old, 1.0);
                    Some((0.5, -3.0))
                }
                1 => Some((1.0, -100.0)),
                _ => None,
            },
            true,
            1,
        );
        let mut out = [0.0f32; 3];
        lane.read_row(0, 3, &mut out);
        assert_eq!(
            out,
            [0.0, 0.0, 0.5],
            "zero clamp wins over the floor for + rules"
        );

        lane.write_row(0, 3, &[1.0, -1.0, 0.5], 0);
        let mut next = [0.0; 3];
        lane.accumulate_row(
            0,
            3,
            &mut next,
            |i, _| (i == 1).then_some((1.0, -100.0)),
            false,
            2,
        );
        lane.read_row(0, 3, &mut out);
        assert_eq!(
            out,
            [1.0, -2.0, 0.5],
            "signed rule: clamped to the configured floor"
        );
    }

    #[test]
    fn f32_accumulate_skips_inactive_cells() {
        let lane = F32Regret::<Local>::new(1, 3, ());
        lane.write_row(0, 3, &[1.0, -1.0, 0.5], 0);
        let mut next = [2.0, -2.0, 99.0];
        lane.accumulate_row(
            0,
            3,
            &mut next,
            |i, _| (i != 2).then_some((1.0, 0.0)),
            false,
            1,
        );
        let mut out = [0.0f32; 3];
        lane.read_row(0, 3, &mut out);
        assert_eq!(out, [2.0, -2.0, 0.5]);
    }

    #[test]
    fn int32_accumulate_is_unbiased_for_sub_quantum_increments() {
        // 0.003 per tick at a 0.01 quantum: round-to-nearest would freeze at 0;
        // stochastic rounding recovers 3 units over 1_000 ticks in expectation.
        let lane = Int32Regret::<Local>::new(1, 1, Int32Config::default());
        for t in 1..=1_000usize {
            let mut next = [0.0];
            lane.accumulate_row(0, 1, &mut next, |_, _| Some((1.0, 0.003)), false, t);
        }
        let mut out = [0.0f32; 1];
        lane.read_row(0, 1, &mut out);
        assert!((out[0] - 3.0).abs() < 0.3, "expected ≈ 3.0, got {}", out[0]);
    }

    #[test]
    fn int32_default_floor_tracks_scale() {
        let lane = Int32Regret::<Local>::new(
            1,
            1,
            Int32Config {
                scale: 1.0,
                ..Int32Config::default()
            },
        );
        assert_eq!(lane.floor(), Some(i32::MIN as f32));
    }

    #[test]
    #[should_panic(expected = "scale must be positive")]
    fn int32_rejects_non_positive_scale() {
        let _ = Int32Regret::<Local>::new(
            1,
            1,
            Int32Config {
                scale: 0.0,
                floor: 0.0,
            },
        );
    }

    #[test]
    fn no_strategy_is_zero_sized_and_noop() {
        assert_eq!(std::mem::size_of::<NoStrategy>(), 0);
        const {
            assert!(!<NoStrategy as StrategyLane<Dcfr, Local>>::HAS_AVERAGE);
            assert!(<F32SumStrategy<Local> as StrategyLane<Dcfr, Local>>::HAS_AVERAGE);
        }
        let lane = <NoStrategy as StrategyLane<Dcfr, Local>>::new(3, 3);
        let step = Dcfr::step(&DiscountParams::RECOMMENDED, 1);
        <NoStrategy as StrategyLane<Dcfr, Local>>::accumulate(
            &lane,
            0,
            3,
            &step,
            &[0.2, 0.3, 0.5],
            1,
        );
        <NoStrategy as StrategyLane<Dcfr, Local>>::reset(&lane);
    }

    #[test]
    #[should_panic(expected = "no average lane")]
    fn no_strategy_average_into_panics() {
        let lane = <NoStrategy as StrategyLane<Dcfr, Local>>::new(1, 2);
        let mut out = [0.0f32; 2];
        <NoStrategy as StrategyLane<Dcfr, Local>>::average_into(&lane, 0, 2, &mut out);
    }

    #[test]
    fn mask_exact_flags() {
        const {
            assert!(<F32Regret<Local> as RegretLane<Local>>::MASK_EXACT);
            assert!(<Int32Regret<Local> as RegretLane<Local>>::MASK_EXACT);
            assert!(!<Int16Regret<Local> as RegretLane<Local>>::MASK_EXACT);
        }
        assert_eq!(F32Regret::<Local>::new(1, 1, ()).floor(), None);
    }
}
