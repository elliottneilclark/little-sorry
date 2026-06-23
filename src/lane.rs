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
//! store and is the single type parameter `BatchedMatcher` will eventually
//! accept for pluggable memory layouts. [`F32Full`] is the default layout
//! (f32 regret + f32 strategy), reproducing today's behavior.

use crate::storage::{AccumCell, StorageBackend};
use crate::update_rule::UpdateRule;

// ── Traits ───────────────────────────────────────────────────────────────────

/// Cumulative-regret lane. The matcher computes new regret via `UpdateRule`
/// (unchanged math) and hands the row to `write_row`; the store owns only the
/// representation. `read_row` returns the stored regret as f32.
pub trait RegretLane<B: StorageBackend>: Sized {
    fn new(num_rows: usize, num_actions: usize) -> Self;
    fn read_row(&self, row: usize, num_actions: usize, out: &mut [f32]);
    fn write_row(&self, row: usize, num_actions: usize, regret: &[f32]);
}

/// Cumulative/average-strategy lane. Owns its accumulation recurrence and its
/// average readout; `accumulate` consumes the rule's `(discount, weight)` via
/// `R::strategy_accumulation`.
pub trait StrategyLane<R: UpdateRule, B: StorageBackend>: Sized {
    fn new(num_rows: usize, num_actions: usize) -> Self;
    fn accumulate(&self, row: usize, num_actions: usize, step: &R::Step, strategy: &[f32]);
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
    fn new(num_rows: usize, num_actions: usize) -> Self {
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
    fn write_row(&self, row: usize, n: usize, regret: &[f32]) {
        for (i, &v) in regret[..n].iter().enumerate() {
            self.cells[row * n + i].store(v.to_bits());
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
    fn accumulate(&self, row: usize, n: usize, step: &R::Step, strategy: &[f32]) {
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
    ) {
        <Self as StrategyLane<R, B>>::accumulate(self, row, n, step, s);
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

/// Strategy lane storing the bounded running average σ̄ ∈ [0,1] in u16, with one
/// f32 weight `W` per row driving the incremental update — derived from the f32
/// `sum_p/Σsum_p` recurrence: `W ← discount·W + weight`,
/// `σ̄ += (weight/W)·(σ − σ̄)`. Always in [0,1] (convex), so u16 fixed-point is
/// safe. `average_into` returns σ̄ directly (one normalize repairs rounding drift).
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
    fn set_sigma(&self, idx: usize, v: f32) {
        // Safety: encode clamps v to [0,1] then multiplies by u16::MAX (65535),
        // so the result is always in 0..=65535 — no truncation can occur.
        #[allow(clippy::cast_possible_truncation)]
        self.cells[idx].store(crate::unit_fixed::encode(v, Self::MAX) as u16);
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

    fn accumulate(&self, row: usize, n: usize, step: &R::Step, strategy: &[f32]) {
        let (discount, weight) = R::strategy_accumulation(step);
        let w_new = discount * self.w_load(row) + weight;
        self.w_store(row, w_new);
        let frac = if w_new > 0.0 { weight / w_new } else { 0.0 };
        for (i, &s) in strategy[..n].iter().enumerate() {
            let idx = row * n + i;
            let cur = self.sigma(idx);
            self.set_sigma(idx, cur + frac * (s - cur));
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
    ) {
        <Self as StrategyLane<R, B>>::accumulate(self, row, n, step, s);
    }

    pub(crate) fn average_into<R: UpdateRule>(&self, row: usize, n: usize, out: &mut [f32]) {
        <Self as StrategyLane<R, B>>::average_into(self, row, n, out);
    }
}

// ── u16 bounded-average store with a single shared weight ────────────────────

/// Strategy lane storing the bounded running average σ̄ ∈ [0,1] in u16, with a
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
    fn set_sigma(&self, idx: usize, v: f32) {
        // Safety: encode clamps v to [0,1] then multiplies by u16::MAX (65535),
        // so the result is always in 0..=65535 — no truncation can occur.
        #[allow(clippy::cast_possible_truncation)]
        self.cells[idx].store(crate::unit_fixed::encode(v, Self::MAX) as u16);
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

    fn accumulate(&self, row: usize, n: usize, step: &R::Step, strategy: &[f32]) {
        let (discount, weight) = R::strategy_accumulation(step);
        // Row 0 advances the shared W once per tick; rows ≠ 0 reuse it.
        if row == 0 {
            let w_new = discount * self.w_load() + weight;
            self.w_store(w_new);
        }
        let w = self.w_load();
        let frac = if w > 0.0 { weight / w } else { 0.0 };
        for (i, &s) in strategy[..n].iter().enumerate() {
            let idx = row * n + i;
            let cur = self.sigma(idx);
            self.set_sigma(idx, cur + frac * (s - cur));
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
    ) {
        <Self as StrategyLane<R, B>>::accumulate(self, row, n, step, s);
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
/// both parameter sets; the layout remains gated **EXPERIMENTAL** pending
/// rs-poker's out-of-sample exploitability A/B.
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
    fn new(num_rows: usize, num_actions: usize) -> Self {
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

    fn write_row(&self, row: usize, n: usize, regret: &[f32]) {
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discount::DiscountParams;
    use crate::rules::Dcfr;
    use crate::storage::Local;

    #[test]
    fn f32_regret_lane_round_trips_exactly() {
        let lane = F32Regret::<Local>::new(2, 3);
        let vals = [1.5f32, -2.0, 1e9];
        lane.write_row(1, 3, &vals);
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
        lane.accumulate::<Dcfr>(0, 3, &step, &strat);
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
            lane.accumulate::<LinearCfr>(0, 2, &step, &strat);
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
            f32_lane.accumulate::<Dcfr>(0, 3, &step, strat);
            u16_lane.accumulate::<Dcfr>(0, 3, &step, strat);
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
        let lane = Int16Regret::<Local>::new(1, 3);
        let regret = [1000.0f32, -250.0, 30.0];
        lane.write_row(0, 3, &regret);
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
        let lane = Int16Regret::<Local>::new(1, 2);
        lane.write_row(0, 2, &[1.0, -1.0]); // tiny scale
        lane.write_row(0, 2, &[1.0e6, -5.0e5]); // forces a much larger scale
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
            lane.accumulate::<Dcfr>(0, 3, &step, &[0.5, 0.3, 0.2]);
        }
        // Reset zeros all σ̄ cells and W (call inherent to avoid rule type ambiguity).
        lane.reset_cells();
        // One accumulate with a known σ1.
        let sigma1 = [0.6f32, 0.25, 0.15];
        let step = Dcfr::step(&params, 6);
        lane.accumulate::<Dcfr>(0, 3, &step, &sigma1);
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
            per_row.accumulate::<Dcfr>(0, 3, &step, sigma);
            per_row.accumulate::<Dcfr>(1, 3, &step, sigma);
            shared.accumulate::<Dcfr>(0, 3, &step, sigma);
            shared.accumulate::<Dcfr>(1, 3, &step, sigma);
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
}
