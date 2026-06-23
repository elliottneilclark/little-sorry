# Design: `regret_into` — read the cumulative-regret lane

**Date:** 2026-06-23
**Status:** Approved, ready for implementation plan
**Source report:** `2026-06-23-read-regret-api-for-warm-restart.md`
**Affects:** `BatchedMatcher` public surface (additive — no breaking change)

## Problem (from first principles)

CFR keeps, per information set, a **cumulative-regret** accumulator: a signed
`f32` per action measuring how much, in hindsight, we wish we had played that
action more. The strategy a row plays is *derived* from this accumulator by
regret matching — take the positive part of each action's regret and normalize.
So the regret accumulator is the primitive state; the strategy is a read-only
projection of it.

`BatchedMatcher` already exposes:
- `seed(regret, t0)` — **writes** the cumulative-regret lane (and sets the clock).
- `current_into` / `average_into` — **read** *derived* strategies (the
  positive-part-normalized projections), not the raw accumulator.

There is no public way to **read the raw signed accumulator**. The only reader,
`raw_regret`, is `#[cfg(test)]`. That asymmetry blocks a real use case:
rs-poker's *seat-unstuck* perturbation. When a multiway solve parks one seat in
a bad basin, the escape is to perturb that seat's regret with annealed noise and
re-equilibrate:

```text
R = read regret(row)
noise ~ N(0, σ²),  σ² = η / (1 + t)^γ      # annealed: large early, → 0
seed(|row,a| R[a] + noise[row,a], t0 = num_updates())   # keep the clock
```

The decaying noise schedule (Neelakantan et al. 2015) is what makes this a
*basin escape that still converges*: big kicks early cannot prevent convergence
because they vanish in the limit. Passing `t0 = num_updates()` re-aims the
current strategy without discarding the solve's convergence inertia. The whole
recipe is `R ← R + noise`, which is unexpressible without a public read of `R`.

## Fix

Add the public twin of `raw_regret`, symmetric with how `seed` writes:

```rust
/// Write row `r`'s cumulative regret (the signed accumulator that drives
/// regret matching, NOT the positive-part-normalized strategy) into `out`.
/// This is exactly the quantity `seed` sets, so read → modify → `seed`
/// round-trips.
pub fn regret_into(&self, row: usize, out: &mut [f32])
```

### Why it must read through the regret lane

`seed` writes through `RegretLane::write_row`, so the stored representation may
be either:
- `F32Regret` — the f32 bit pattern (exact), or
- `Int16Regret` — per-row-scaled i16 (lossy: decodes within one row-scaled
  quantum).

`regret_into` must read through the **same** `RegretLane::read_row` path so it is
representation-agnostic and symmetric with `seed`. Reading the raw cells directly
would bypass the i16 scale decode and return garbage for the half-regret layouts.
This is exactly what the existing `raw_regret` test helper and `average_regret`
already do; `regret_into` is their public, documented form.

### Round-trip contract

Because `regret_into` reads precisely what `seed` writes (through the same lane),
`seed(|a,_| { regret_into(row, buf); buf[a] }, num_updates())` is a no-op on the
next-iteration strategy for `F32Regret` (exact) and within the scaled-int quantum
for `Int16Regret`.

## Components

- `BatchedMatcher::regret_into(&self, row, out)` (`src/batched_matcher.rs`),
  placed beside `current_into` / `average_into`. Body mirrors them:
  - `assert!(row < self.num_rows, "row out of range")`
  - `assert!(out.len() >= self.num_actions, "out too short")`
  - `self.regret.read_row(row, self.num_actions, &mut out[..self.num_actions])`
- Doc comment explains, from first principles, that this is the *signed
  accumulator* (pre-regret-matching), that it round-trips with `seed`, and that
  it decodes through the layout's regret store (exact for `F32Regret`, within the
  scaled-int quantum for `Int16Regret`).
- The `#[cfg(test)] raw_regret` helper stays (other tests use it); the new
  public method does not replace it but shares the same read path.

## Tests to add

1. **Round-trip no-op.** Run a few updates; snapshot `current_into`; read with
   `regret_into`; `seed` those exact values with `t0 = num_updates()`; assert
   `current_into` is unchanged (bit-exact for `F32Regret`/`F32Full`).
2. **Known value.** Build a row's regret via a deterministic update sequence;
   assert `regret_into` equals the `raw_regret` test helper element-for-element.
3. **Int16 layout.** On a `HalfRegret`/`Int16Regret`-backed matcher, `seed` a
   known regret vector, then assert `regret_into` decodes within the scaled-int
   quantum of the seeded values.
4. **Panics.** `regret_into` panics on out-of-range row and on a short `out`
   slice (mirrors `current_into`'s guards).

## Out of scope / rejected

- **Returning a `Vec`** instead of writing into `out`. Rejected: `current_into`
  and `average_into` both take a caller-provided buffer to avoid per-call
  allocation in hot export loops; `regret_into` matches that convention.
- **Exposing a mutable handle** to the regret lane. Rejected: the read-modify-
  write is already expressible as `regret_into` + `seed`, and a mutable handle
  would leak the storage representation across the API boundary.

## Acceptance

- `regret_into` reads correctly for `F32Full`/`F32Regret` (exact) and
  `HalfRegret`/`Int16Regret` (within quantum), verified by the four tests.
- `mise check` passes (fmt, clippy all-features, nextest, doc tests, taplo).
- Additive change — no existing public signature changes.
