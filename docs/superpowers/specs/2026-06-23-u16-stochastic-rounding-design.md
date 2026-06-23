# Design: stochastic rounding for the u16 strategy store

**Date:** 2026-06-23
**Status:** Approved, ready for implementation plan
**Source report:** `2026-06-23-u16-strategy-stall-stochastic-rounding.md`
**Affects:** `HalfStrategy`, `HalfStrategyShared`, `HalfBoth`, `HalfBothShared`
**SemVer:** breaking (`StrategyLane` trait method gains a parameter) → 4.0.0 on next release

## Problem

`U16AvgStrategy::accumulate` maintains the bounded running average with
`σ̄ ← σ̄ + frac·(σ − σ̄)`, where `frac = weight / W_new`. Under DCFR the strategy
discount is `(t/(t+1))^γ`, so the running weight grows ≈ `weight·t/γ` and
therefore `frac ≈ γ/t`. The per-iteration change to a stored component is
`frac·(σ − σ̄) ≈ (γ/t)·Δ`.

The u16 quantum is `1/65535 ≈ 1.5e-5`. Round-to-nearest encoding discards any
increment below half a quantum, so once `(γ/t)·Δ < ½·quantum` the update
**rounds to zero** and `σ̄` stops moving. For γ=10, Δ≈0.1 the crossover is around
`t ≈ 67 k`; past it the u16 average is frozen regardless of how long the solve
runs. This is the classic *swamping / stagnation* failure of round-to-nearest
fixed-point accumulation (Croci, Fasi, Higham, Mary, Mikaitis, *R. Soc. Open
Sci.* 9:211631, 2022).

Observed by rs-poker's out-of-sample exploitability A/B (n=4 push/fold cell,
single thread, fixed seed, DCFR `(α,β,γ) = (2.3, 0, 10)`, lower = better):

| iterations | F32Full | HalfStrategyShared |
|---|---:|---:|
| 50 k | 95.2 | 119.8 |
| 200 k | 39.1 | 120.8 |
| 1 M | 19.0 | **128.3** |

F32 converges; the u16 average freezes near its ~50 k-iteration quality. It
reproduces single-threaded, so it is the lane representation, not the Hogwild
shared-`W` race. PR #16's equivalence test used `DiscountParams::RECOMMENDED`
(γ=2) for only 5 k iterations — below the crossover — so it never exercised the
stall.

## Fix

Apply **stochastic rounding** when encoding `σ̄`'s new value into u16. Storing a
real value whose exact u16 code is `c + f` (`f ∈ [0,1)`) becomes code `c+1` with
probability `f`, else `c`. Then `E[stored] = x`, so sub-quantum increments
survive *in expectation* instead of being deterministically discarded — the
running average keeps moving. (Gupta et al., "Deep Learning with Limited
Numerical Precision," ICML 2015, found stochastic rounding "crucial" for 16-bit
fixed-point training; Croci et al. 2022 for the `√n` vs `n` error bound.)

The randomness comes from a **counter-based PRNG keyed by `(row, action,
update_count)`** — stateless and deterministic, so it is reproducible and
race-free under the `Atomic` backend with no shared RNG state. A single
generator (`splitmix64`) is hardcoded; this is not a pluggable compile-time
parameter.

The decode path is unchanged.

## Components

### 1. Counter-based PRNG (`unit_fixed.rs`, crate-internal)

```rust
/// Deterministic, stateless draw in [0, 1) from a cell+tick key. Same key ⇒
/// same value; no shared mutable state ⇒ reproducible and race-free under the
/// Atomic backend.
fn u01(row: usize, action: usize, update_count: usize) -> f32;
```

Implementation: mix the three key components into a `u64`, run the splitmix64
finalizer, take the top 24 bits and divide by `2^24` to land in `[0, 1)` with no
bias from float rounding. Constants and exact mixing are an implementation
detail; the contract is determinism + good equidistribution of the fractional
draw.

### 2. Stochastic encode (`unit_fixed.rs`)

```rust
/// Stochastic-rounding encode of x ∈ [0,1] to a code in 0..=max_code, using a
/// caller-supplied draw u01 ∈ [0,1). Unbiased: E[result] = clamp(x)·max_code.
pub(crate) fn encode_stochastic(x: f32, max_code: u32, u01: f32) -> u32;
```

- Clamp `x` to `[0,1]` (same guard as `encode`), scale by `max_code`.
- `floor = scaled.floor()`, `frac = scaled - floor`.
- code = `floor as u32 + if u01 < frac { 1 } else { 0 }`, clamped to `..=max_code`.
- Exact integer inputs (`frac == 0`) never round up — endpoints `0` and
  `max_code` round-trip exactly.

`encode` (round-to-nearest) is **left unchanged**: the export codec
(`quantize`) and its golden bit-tests depend on its exact, deterministic
round-trip. Stochastic rounding is confined to the strategy lane.

### 3. `StrategyLane::accumulate` gains `update_count`

```rust
fn accumulate(
    &self,
    row: usize,
    num_actions: usize,
    step: &R::Step,
    strategy: &[f32],
    update_count: usize, // NEW
);
```

`R::Step` is rule-specific and does not uniformly carry `t`, so the iteration
count is threaded explicitly. This is a breaking change to the public
`StrategyLane` trait.

- **`F32SumStrategy`**: ignores `update_count` (signature only).
- **Inherent `accumulate<R>` forwarders** on each lane and all test call sites:
  updated to pass `update_count`.

### 4. Matcher passes the clock (`batched_matcher.rs`)

`BatchedMatcher::update_one` calls
`self.strategy.accumulate(row, a, step, &s.strategy, self.num_updates())`.
After `tick()` the counter equals the current iteration `t`, and `update_batch`
holds it fixed across the per-row accumulate loop, so every row in a batch keys
on the same `t` — matching the shared-tick semantics. `t ≥ 1` always
(`tick` does `fetch_incr() + 1`).

### 5. u16 lanes use stochastic store

Both `U16AvgStrategy` and `U16AvgStrategyShared` replace the per-cell
`set_sigma(idx, v)` in `accumulate` with a stochastic store:

```rust
let u = u01(row, i, update_count);
self.cells[idx].store(encode_stochastic(v, Self::MAX, u) as u16);
```

where `i` is the action index in the loop. No generics are added — these two
lane types are the single fix point and back all four affected layouts, so the
`Layout` structs (`HalfStrategy`, `HalfStrategyShared`, `HalfBoth`,
`HalfBothShared`) are untouched.

The `set_sigma`/`set_sigma_stochastic` helper shape is an implementation choice;
`sigma` (decode) and `reset_cells` are unchanged. Reset still zeroes σ̄ and `W`
so the first post-reset `accumulate` lands σ₁ exactly (`frac = 1`); with
`frac = 1` the stored value is `σ₁` whose code has `frac ≈ 0`, so stochastic
rounding does not perturb the reset-then-seed invariant beyond the existing one
quantum.

## Data flow

```
update_batch(t) ──tick──> step, counter=t
  └─ for each row:
       update_one(row, step)
         └─ derive σ (post-update strategy)
         └─ strategy.accumulate(row, n, step, σ, t)
              └─ W ← discount·W + weight ;  frac = weight/W
              └─ for action i:
                   v = σ̄ + frac·(σ_i − σ̄)
                   u = u01(row, i, t)
                   cells[idx] = encode_stochastic(v, MAX, u)
average_into(row) ──> decode each cell ──> normalize  (unchanged)
```

## Tests to add

All single-threaded `Local` for determinism.

1. **Stall regression (load-bearing).** Drive a single row with DCFR γ=10 for
   ≥1 M updates toward a moving-then-settling target whose late-stage increments
   are sub-quantum. Assert the stochastic-rounded u16 average tracks the f32
   `F32SumStrategy` average within tolerance. Contrast: a parallel computation
   using plain round-to-nearest `encode` freezes (demonstrating the bug the fix
   removes). γ matches the caller; this is the test PR #16 lacked.
2. **Unbiasedness.** Over many `update_count` values, the mean stored value of a
   fixed sub-quantum increment equals the true value within Monte-Carlo error
   (`|mean − true| < k/√N`). Exercises `encode_stochastic` directly.
3. **Determinism.** The same `(row, action, update_count)` stream produces
   identical stored codes across repeated runs — counter-based, no shared mutable
   RNG.
4. **Regression guards (keep green).** Existing `unit_fixed` round-trip tests,
   the export-codec golden bit-tests, and `u16_reset_then_first_accumulate_lands_sigma1`
   must still pass — confirming `encode` is untouched and the reset invariant holds.

## Out of scope / rejected

- **Pluggable compile-time RNG.** Considered and dropped: one hardcoded
  `splitmix64` keeps the surface minimal. The PRNG lives behind a crate-internal
  function, so swapping it later is a localized change.
- **Per-row-scaled u16 *sum* lane** (rescale-on-overflow, no shrinking
  increment). A different representation; the report defers it. Not pursued here.
- **Runtime `dyn Rng`.** Vtable dispatch in the per-cell hot loop — unacceptable.

## Acceptance

- All four u16 layouts converge (no freeze) under γ=10, long-horizon, single
  thread — verified by the stall-regression test in-tree, and ultimately by
  rs-poker's out-of-sample exploitability A/B matching `F32Full`.
- `mise check` passes (fmt, clippy all-features, nextest, doc tests, taplo).
- Export codec and reset invariants unchanged.
