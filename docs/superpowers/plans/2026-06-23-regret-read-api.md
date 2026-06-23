# `regret_into` Read API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public `BatchedMatcher::regret_into(row, out)` that reads a row's raw cumulative-regret accumulator through the layout's regret store, enabling read → perturb → `seed` warm-restart workflows.

**Architecture:** A single additive method on `BatchedMatcher`, placed beside `current_into` / `average_into`, reading through `RegretLane::read_row` so it is representation-agnostic (exact for `F32Regret`, within the scaled-int quantum for `Int16Regret`). It is the public twin of the `#[cfg(test)] raw_regret` helper.

**Tech Stack:** Rust (nightly, edition 2024), `mise`, `nextest`, `clippy` with `#![deny(clippy::all)]`.

## Global Constraints

- Additive only — no existing public signature changes (not a breaking change).
- `#![deny(clippy::all)]` — no warnings.
- Buffer-writing convention: take a caller-provided `out: &mut [f32]` (no `Vec` return), matching `current_into` / `average_into`.
- Doc comment must explain from first principles: this is the *signed accumulator* (pre-regret-matching), it round-trips with `seed`, and it decodes through the layout's regret store.
- Run all checks with `mise check`. Do NOT bump the crate version.

## File Structure

- `src/batched_matcher.rs` — add `regret_into` to the public `impl` block (near `average_into`, ~line 321); add tests to the existing `#[cfg(test)] mod tests`.

---

## Task 1: `regret_into` accessor + tests

**Files:**
- Modify: `src/batched_matcher.rs` (add method after `average_into`, ~line 321; add tests in `mod tests`)

**Interfaces:**
- Consumes: `self.regret` (`RegretLane::read_row`), `self.num_rows`, `self.num_actions`, `self.num_updates()`, `self.seed`, the `#[cfg(test)] raw_regret` helper.
- Produces: `pub fn regret_into(&self, row: usize, out: &mut [f32])`.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `src/batched_matcher.rs`. Ensure these imports are present in the module (add any that are missing): `use crate::lane::HalfRegret;`. (`Dcfr`, `DiscountParams`, `Local` are already imported.)

```rust
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
            assert!((g - w).abs() <= quantum + 1e-2, "regret_into[{i}] {g} vs seeded {w}");
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo nextest run --lib regret_into`
Expected: FAIL — `no method named regret_into found`.

- [ ] **Step 3: Implement `regret_into`**

In `src/batched_matcher.rs`, add this method immediately after `average_into` (around line 321), inside the public `impl<R, B, L> BatchedMatcher<R, B, L>` block:

```rust
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
        self.regret.read_row(row, self.num_actions, &mut out[..self.num_actions]);
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo nextest run --lib regret_into`
Expected: PASS — all five `regret_into` tests green.

- [ ] **Step 5: Run the full suite and the check gate**

Run: `mise check`
Expected: PASS — fmt, clippy (all features), nextest, doc tests, taplo all green.

- [ ] **Step 6: Stop (do not commit)**

The orchestrator squashes this feature into a single commit matching the project commit template. Leave the working tree staged-or-clean for the orchestrator to commit.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Public `regret_into` reading through the regret lane → Task 1 Step 3.
- Round-trip no-op test → `regret_into_then_seed_is_noop_on_current`.
- Known-value test → `regret_into_matches_raw_regret_helper`.
- Int16 within-quantum test → `regret_into_decodes_int16_layout_within_quantum`.
- Panic guards → `regret_into_panics_on_bad_row`, `regret_into_panics_on_short_out`.
- First-principles doc comment → Task 1 Step 3.

**Placeholder scan:** none — full code in every step.

**Type consistency:** `regret_into(&self, usize, &mut [f32])` used identically in tests and implementation; `HalfRegret` layout import noted.
