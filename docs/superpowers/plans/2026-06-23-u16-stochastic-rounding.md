# u16 Strategy Stochastic Rounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the u16 bounded-average strategy store from freezing under high-γ long-horizon DCFR by encoding σ̄ with unbiased stochastic rounding.

**Architecture:** Add a stateless counter-based PRNG and an unbiased `encode_stochastic` to `unit_fixed.rs`; thread the iteration count (`update_count`) through the `StrategyLane::accumulate` trait method so the u16 lanes can key the PRNG by `(row, action, update_count)`; switch both u16 lanes (`U16AvgStrategy`, `U16AvgStrategyShared`) from round-to-nearest to stochastic encoding. The round-to-nearest `encode` is left untouched (the export codec depends on it). The two u16 lanes back all four affected layouts, so the layout structs need no changes.

**Tech Stack:** Rust (nightly, edition 2024), `mise` task runner, `nextest`, `clippy` with `#![deny(clippy::all)]`.

## Global Constraints

- Numeric kernels use plain `f32` / `&[f32]`; no new dependencies.
- `#![deny(clippy::all)]` is enforced crate-wide — no clippy warnings allowed.
- The export codec's round-to-nearest `unit_fixed::encode` MUST remain byte-for-byte unchanged (golden round-trip tests in `quantize` depend on it).
- Single-threaded `Local` backend must stay fully deterministic.
- This is a breaking change to the public `StrategyLane` trait (a method gains a parameter). Do NOT bump the crate version — releases are separate `chore: Release` commits.
- Run all checks with `mise check` (fmt, clippy all-features, nextest, doc tests, taplo). Prefer `mise` over raw cargo.
- **Commits are deferred to the orchestrator.** This feature ships as ONE squashed commit matching `~/.config/git/commit.template`. Implement and verify each task with TDD, but do NOT run the per-task `git commit` steps below — they document logical checkpoints only. Leave the working tree for the orchestrator to commit.

---

## File Structure

- `src/unit_fixed.rs` — add `u01` (PRNG) and `encode_stochastic`; existing `encode`/`decode` unchanged. (Task 1)
- `src/lane.rs` — `StrategyLane::accumulate` gains `update_count`; the three trait impls and three inherent forwarders updated; u16 lanes switch to stochastic store; new stall-regression test. (Tasks 2 & 3)
- `src/batched_matcher.rs` — the single matcher call site passes `self.num_updates()`. (Task 2)

---

## Task 1: PRNG and unbiased stochastic encode in `unit_fixed.rs`

**Files:**
- Modify: `src/unit_fixed.rs` (add two functions + tests)

**Interfaces:**
- Consumes: nothing (pure functions).
- Produces:
  - `pub(crate) fn u01(row: usize, action: usize, update_count: usize) -> f32` — deterministic draw in `[0, 1)`.
  - `pub(crate) fn encode_stochastic(x: f32, max_code: u32, u01: f32) -> u32` — unbiased encode of `x ∈ [0,1]` to `0..=max_code`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/unit_fixed.rs` (after the existing tests):

```rust
    #[test]
    fn u01_is_deterministic_and_in_range() {
        // Same key ⇒ identical bits; draws stay in [0, 1).
        for &(r, a, t) in &[(0usize, 0usize, 1usize), (3, 1, 999), (7, 2, 67_000)] {
            assert_eq!(u01(r, a, t).to_bits(), u01(r, a, t).to_bits());
        }
        for t in 0..10_000usize {
            let u = u01(t % 5, t % 3, t);
            assert!((0.0..1.0).contains(&u), "u01 out of range: {u}");
        }
        // Distinct keys generally differ (guards against a constant generator).
        assert_ne!(u01(0, 0, 1), u01(0, 0, 2));
        assert_ne!(u01(0, 0, 1), u01(1, 0, 1));
        assert_ne!(u01(0, 0, 1), u01(0, 1, 1));
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
                assert!(c as f32 >= floor && c as f32 <= floor + 1.0, "code {c} not adjacent to {floor}");
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
            let u = u01(7, 2, t as usize);
            sum += u64::from(encode_stochastic(x, max, u));
        }
        let mean = sum as f64 / f64::from(n);
        assert!((mean - scaled).abs() < 0.05, "biased: mean {mean} vs true {scaled}");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo nextest run --lib unit_fixed`
Expected: FAIL — `cannot find function u01` / `encode_stochastic` not found.

- [ ] **Step 3: Implement the two functions**

Add to `src/unit_fixed.rs` after the existing `decode` function (before the `tests` module):

```rust
/// Deterministic, stateless draw in `[0, 1)` from a cell+tick key. Same key ⇒
/// same value; no shared mutable state, so it is reproducible and race-free
/// under the `Atomic` backend. The three key components are mixed into one word
/// and run through the splitmix64 finalizer; the top 24 bits become the fraction
/// (24 bits is exactly representable in an `f32` mantissa, so the division is
/// exact and unbiased).
#[inline]
pub(crate) fn u01(row: usize, action: usize, update_count: usize) -> f32 {
    let mut z = (row as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((action as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((update_count as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 24 bits → [0, 1).
    #[allow(clippy::cast_precision_loss)]
    let num = (z >> 40) as f32;
    num / ((1u32 << 24) as f32)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo nextest run --lib unit_fixed`
Expected: PASS — all `unit_fixed` tests green (new four plus the two existing).

- [ ] **Step 5: Check formatting and lints for the file**

Run: `mise run check:fmt && mise run check:clippy`
Expected: clean, no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/unit_fixed.rs
git commit -m "feat(unit_fixed): counter-based PRNG + unbiased stochastic encode

Claude-Session: https://claude.ai/code/session_01NB9Te4eCrGLUdw25narLFT"
```

---

## Task 2: Thread `update_count` through `StrategyLane::accumulate`

This is mechanical plumbing only — **no behavior change**. The u16 lanes still
round-to-nearest after this task; they accept the new parameter but ignore it
(named `_update_count`). Deliverable: the crate compiles and every existing test
passes unchanged.

**Files:**
- Modify: `src/lane.rs` (trait, 3 trait impls, 3 inherent forwarders, 12 test call sites)
- Modify: `src/batched_matcher.rs:207` (matcher call site)

**Interfaces:**
- Consumes: `BatchedMatcher::num_updates()` (returns the current iteration `t`, already in scope on `self`).
- Produces: new `StrategyLane::accumulate` signature with trailing `update_count: usize`, consumed by Task 3.

- [ ] **Step 1: Update the trait method signature**

In `src/lane.rs`, the `StrategyLane` trait (around line 40):

```rust
    fn accumulate(
        &self,
        row: usize,
        num_actions: usize,
        step: &R::Step,
        strategy: &[f32],
        update_count: usize,
    );
```

- [ ] **Step 2: Update `F32SumStrategy`'s trait impl**

In `src/lane.rs` (around line 105), change the signature; the body is unchanged:

```rust
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
```

- [ ] **Step 3: Update `U16AvgStrategy`'s trait impl (signature only — still round-to-nearest)**

In `src/lane.rs` (around line 212), add the parameter named `_update_count`; body unchanged:

```rust
    fn accumulate(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        strategy: &[f32],
        _update_count: usize,
    ) {
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
```

- [ ] **Step 4: Update `U16AvgStrategyShared`'s trait impl (signature only)**

In `src/lane.rs` (around line 335), add `_update_count`; body unchanged:

```rust
    fn accumulate(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        strategy: &[f32],
        _update_count: usize,
    ) {
        let (discount, weight) = R::strategy_accumulation(step);
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
```

- [ ] **Step 5: Update the three inherent `accumulate<R>` forwarders**

In `src/lane.rs` there are three identical-shaped forwarders (around lines 135, 254, 379). For **each** of the three, change to:

```rust
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
```

- [ ] **Step 6: Update the matcher call site**

In `src/batched_matcher.rs` (line ~207), change:

```rust
        self.strategy.accumulate(row, a, step, &s.strategy);
```

to:

```rust
        self.strategy
            .accumulate(row, a, step, &s.strategy, self.num_updates());
```

(After `tick()`, `self.num_updates()` equals this tick's `t`; `update_batch` holds it fixed across the per-row loop, so all rows in a batch key on the same `t`.)

- [ ] **Step 7: Update the 12 existing test call sites in `src/lane.rs`**

Append the matching iteration index as the final argument to each call. The index is the same `t` used to build that call's `step`:

- Line ~571 (`f32_sum_strategy_matches_accumulate_then_normalize`, `step` built with `t=1`):
  `lane.accumulate::<Dcfr>(0, 3, &step, &strat, 1);`
- Line ~594 (`u16_average_keeps_moving_to_horizon`, loop `t in 1..=20_000`):
  `lane.accumulate::<LinearCfr>(0, 2, &step, &strat, t);`
- Lines ~623–624 (`u16_avg_matches_f32_sum_average_within_quantum`, `step` built with `t + 1`):
  `f32_lane.accumulate::<Dcfr>(0, 3, &step, strat, t + 1);`
  `u16_lane.accumulate::<Dcfr>(0, 3, &step, strat, t + 1);`
- Line ~673 (`u16_reset_then_first_accumulate_lands_sigma1`, loop `t in 1..=5`):
  `lane.accumulate::<Dcfr>(0, 3, &step, &[0.5, 0.3, 0.2], t);`
- Line ~680 (same test, `step` built with `t=6`):
  `lane.accumulate::<Dcfr>(0, 3, &step, &sigma1, 6);`
- Lines ~717–720 (`shared_weight_matches_per_row_under_batch`, `step` built with `t + 1`):
  `per_row.accumulate::<Dcfr>(0, 3, &step, sigma, t + 1);`
  `per_row.accumulate::<Dcfr>(1, 3, &step, sigma, t + 1);`
  `shared.accumulate::<Dcfr>(0, 3, &step, sigma, t + 1);`
  `shared.accumulate::<Dcfr>(1, 3, &step, sigma, t + 1);`

- [ ] **Step 8: Run the full test suite to verify nothing changed behaviorally**

Run: `cargo nextest run`
Expected: PASS — same tests pass as before (this task adds no new test and changes no behavior).

- [ ] **Step 9: Check formatting and lints**

Run: `mise run check:fmt && mise run check:clippy`
Expected: clean — in particular no "unused variable" warning (the unused params are prefixed `_`).

- [ ] **Step 10: Commit**

```bash
git add src/lane.rs src/batched_matcher.rs
git commit -m "refactor(lane): thread update_count through StrategyLane::accumulate

Breaking: StrategyLane::accumulate gains an update_count parameter. No
behavior change yet; the u16 lanes ignore it pending stochastic rounding.

Claude-Session: https://claude.ai/code/session_01NB9Te4eCrGLUdw25narLFT"
```

---

## Task 3: Switch the u16 lanes to stochastic rounding

**Files:**
- Modify: `src/lane.rs` (both u16 lanes' `set_sigma` → stochastic store; new stall-regression test)

**Interfaces:**
- Consumes: `unit_fixed::u01`, `unit_fixed::encode_stochastic` (Task 1); `update_count` parameter (Task 2).
- Produces: u16 strategy averages that track f32 under high-γ long horizons.

- [ ] **Step 1: Write the failing stall-regression test**

Add to the `tests` module in `src/lane.rs`. Ensure the module has `use crate::update_rule::UpdateRule;` (needed for `Dcfr::strategy_accumulation`); add it if absent.

```rust
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
        assert!(f32_late_move > 10.0 * quantum, "not exercising late movement: {f32_late_move}");

        // Round-to-nearest froze: essentially no movement after the probe.
        let rn_late_move = (rn_end - rn_at_probe as f32 / max as f32).abs();
        assert!(rn_late_move < 2.0 * quantum, "round-to-nearest unexpectedly moved: {rn_late_move}");

        // Stochastic u16 kept moving and ends closer to f32 than the frozen lane.
        let u16_late_move = (u[0] - u16_at_probe).abs();
        assert!(u16_late_move > 5.0 * quantum, "stochastic u16 froze: {u16_late_move}");
        assert!(
            (u[0] - f[0]).abs() < (rn_end - f[0]).abs(),
            "stochastic should track f32 better than frozen RN: u16={u:?} f32={f:?} rn={rn_end}"
        );
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo nextest run --lib u16_stochastic_tracks_f32`
Expected: FAIL — the u16 lane still rounds to nearest, so `u16_late_move` is ~0 (the `stochastic u16 froze` assertion fires).

- [ ] **Step 3: Replace `set_sigma` with a stochastic store in `U16AvgStrategy`**

In `src/lane.rs`, in the `impl<B: StorageBackend> U16AvgStrategy<B>` block (around line 188), replace the `set_sigma` method with:

```rust
    #[inline]
    fn set_sigma_stochastic(&self, idx: usize, v: f32, u01: f32) {
        // Safety: encode_stochastic clamps v to [0,1] and clamps the code to
        // 0..=MAX, so the result always fits a u16.
        #[allow(clippy::cast_possible_truncation)]
        self.cells[idx]
            .store(crate::unit_fixed::encode_stochastic(v, Self::MAX, u01) as u16);
    }
```

Then update its `accumulate` trait impl (around line 212) to use it, renaming the parameter to `update_count` and keying the PRNG per action:

```rust
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
        for (i, &s) in strategy[..n].iter().enumerate() {
            let idx = row * n + i;
            let cur = self.sigma(idx);
            let v = cur + frac * (s - cur);
            let u = crate::unit_fixed::u01(row, i, update_count);
            self.set_sigma_stochastic(idx, v, u);
        }
    }
```

- [ ] **Step 4: Replace `set_sigma` with a stochastic store in `U16AvgStrategyShared`**

In `src/lane.rs`, in the `impl<B: StorageBackend> U16AvgStrategyShared<B>` block (around line 312), replace `set_sigma` with the same helper:

```rust
    #[inline]
    fn set_sigma_stochastic(&self, idx: usize, v: f32, u01: f32) {
        // Safety: encode_stochastic clamps v to [0,1] and clamps the code to
        // 0..=MAX, so the result always fits a u16.
        #[allow(clippy::cast_possible_truncation)]
        self.cells[idx]
            .store(crate::unit_fixed::encode_stochastic(v, Self::MAX, u01) as u16);
    }
```

Then update its `accumulate` trait impl (around line 335):

```rust
    fn accumulate(
        &self,
        row: usize,
        n: usize,
        step: &R::Step,
        strategy: &[f32],
        update_count: usize,
    ) {
        let (discount, weight) = R::strategy_accumulation(step);
        if row == 0 {
            let w_new = discount * self.w_load() + weight;
            self.w_store(w_new);
        }
        let w = self.w_load();
        let frac = if w > 0.0 { weight / w } else { 0.0 };
        for (i, &s) in strategy[..n].iter().enumerate() {
            let idx = row * n + i;
            let cur = self.sigma(idx);
            let v = cur + frac * (s - cur);
            let u = crate::unit_fixed::u01(row, i, update_count);
            self.set_sigma_stochastic(idx, v, u);
        }
    }
```

- [ ] **Step 5: Run the stall-regression test to verify it passes**

Run: `cargo nextest run --lib u16_stochastic_tracks_f32`
Expected: PASS — stochastic u16 tracks f32; round-to-nearest froze.

- [ ] **Step 6: Run the whole suite (the existing u16 within-quantum / reset / shared tests must stay green)**

Run: `cargo nextest run`
Expected: PASS — including `u16_avg_matches_f32_sum_average_within_quantum`, `u16_reset_then_first_accumulate_lands_sigma1`, `shared_weight_matches_per_row_under_batch`, and `u16_average_keeps_moving_to_horizon`. These tolerate the ~1 quantum of stochastic noise (their thresholds are `2/u16::MAX` per component over short horizons).

> If `u16_avg_matches_f32_sum_average_within_quantum` (4-step sequence, `2/u16::MAX` tolerance) flakes because a single stochastic draw rounded the other way, widen its per-component tolerance from `2.0 / u16::MAX` to `3.0 / u16::MAX` and add a one-line comment: `// +1 quantum of headroom for stochastic rounding`. Do NOT loosen any other test.

- [ ] **Step 7: Check formatting and lints**

Run: `mise run check:fmt && mise run check:clippy`
Expected: clean — `set_sigma` is fully removed (no dead-code warning), no unused `update_count`.

- [ ] **Step 8: Commit**

```bash
git add src/lane.rs
git commit -m "fix(lane): stochastic rounding in u16 strategy store

Keys a counter-based PRNG by (row, action, update_count) so sub-quantum
increments survive in expectation. Fixes the high-γ long-horizon stall
across HalfStrategy, HalfStrategyShared, HalfBoth, HalfBothShared.

Claude-Session: https://claude.ai/code/session_01NB9Te4eCrGLUdw25narLFT"
```

---

## Task 4: Full verification and docs

**Files:**
- Modify: `src/lane.rs` doc comment on `U16AvgStrategy` (note the stochastic-rounding behavior) — optional polish only if the existing doc claims round-to-nearest.

- [ ] **Step 1: Update the `U16AvgStrategy` doc comment**

In `src/lane.rs` (around line 160), the doc comment ends with "`average_into` returns σ̄ directly (one normalize repairs rounding drift)." Append a sentence:

```rust
/// safe. `average_into` returns σ̄ directly (one normalize repairs rounding drift).
/// Encoding uses **stochastic rounding** keyed by `(row, action, update_count)`,
/// so sub-quantum increments survive in expectation and the average does not
/// freeze under high-γ long-horizon DCFR (see `unit_fixed::encode_stochastic`).
```

- [ ] **Step 2: Run the full check suite**

Run: `mise check`
Expected: PASS — fmt, clippy (all features), nextest, doc tests, and taplo all green.

- [ ] **Step 3: Commit**

```bash
git add src/lane.rs
git commit -m "docs(lane): note stochastic rounding on U16AvgStrategy

Claude-Session: https://claude.ai/code/session_01NB9Te4eCrGLUdw25narLFT"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- PRNG keyed by `(row, action, update_count)` → Task 1 (`u01`) + Task 3 (call site).
- Unbiased stochastic encode, `encode` left unchanged → Task 1.
- Thread update-count (Step is rule-specific) → Task 2.
- Fix both u16 lanes → covers all four layouts → Task 3.
- Tests: stall regression → Task 3; unbiasedness + determinism → Task 1; codec/reset guards stay green → Tasks 2 & 3 Step 6.
- SemVer breaking, no version bump → Global Constraints.

**Placeholder scan:** none — every code step shows full code; every run step shows the command and expected result.

**Type consistency:** `u01(usize, usize, usize) -> f32`, `encode_stochastic(f32, u32, f32) -> u32`, and the `accumulate(.., update_count: usize)` signature are used identically across Tasks 1–3.
