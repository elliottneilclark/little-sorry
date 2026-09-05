# Little Sorry

A Rust library for regret minimization algorithms (Counterfactual Regret Minimization) used to find Nash equilibrium strategies in imperfect-information games.

## Features

- **6 CFR variants** via the `RegretMinimizer` trait:
  - **CFR+** — regret clipping at zero
  - **Discounted CFR (DCFR)** — time-based discounting with configurable parameters
  - **DCFR+** — combines DCFR discounting with CFR+ clipping
  - **Linear CFR** — linear time-weighted regrets
  - **Predictive CFR+ (PCFR+)** — uses future regret predictions
  - **Predictive DCFR+ (PDCFR+)** — combines DCFR+ discounting with predictive updates
- Zero-allocation hot path — no heap allocations during `update_regret`
- Minimal dependencies (`rand` only)
- Rock-Paper-Scissors example game (feature-gated behind `rps`)
- **Batched, storage-generic matchers** for large and concurrent solves —
  `BatchedMatcher<Rule, Backend>` owns many information sets on one shared
  iteration clock, generic over the update rule and over a single-threaded or
  lock-free atomic cell backend
- **Compact strategy export** — dependency-free fixed-point quantization of a
  solved average strategy (`quantize_dist` / `dequantize_dist`)

## Getting Started

Add this to your `Cargo.toml`:

```toml
[dependencies]
little-sorry = "4.1.0"
```

### Quick Example

```rust
use little_sorry::{CfrPlusRegretMatcher, RegretMinimizer};

let mut matcher = CfrPlusRegretMatcher::new(3);

// Run many iterations of regret updates
for _ in 0..1000 {
    let rewards = &[1.0, -0.5, 0.2];
    matcher.update_regret(rewards);
}

// Get the Nash equilibrium approximation
let strategy = matcher.best_weight();
```

All variants implement the `RegretMinimizer` trait, so you can swap algorithms generically:

```rust
use little_sorry::{DiscountedRegretMatcher, RegretMinimizer};

fn train<M: RegretMinimizer>(matcher: &mut M, iterations: usize) {
    for _ in 0..iterations {
        let rewards = &[1.0, -0.5, 0.2];
        matcher.update_regret(rewards);
    }
}
```

### Scaling up: batched matchers and strategy export

For abstraction-based or multi-threaded solvers, `BatchedMatcher` owns many
information sets ("rows") that advance together, so per-iteration discount
factors are computed once per visit instead of once per row. The update rule and
the storage backend are each one type parameter: pick `Local` for a
zero-overhead single-threaded solve or `Atomic` to update a shared matcher
lock-free from many threads. The solved average strategy reads out identically
for every rule and can be exported to compact fixed-point codes.

```rust
use little_sorry::{BatchedMatcher, Dcfr, DiscountParams, Local};
use little_sorry::{dequantize_dist, quantize_dist};

// One node owning 8 abstraction classes over 3 actions, using DCFR on the
// single-threaded backend. Swap `Dcfr` for `PdcfrPlus`, or `Local` for
// `Atomic`, with no other changes.
let node = BatchedMatcher::<Dcfr, Local>::new(8, 3, DiscountParams::RECOMMENDED);

let mut expected = [0.0; 8];
for _ in 0..1000 {
    node.update_batch(|action, _row| [1.0, -0.5, 0.2][action], &mut expected);
}

// Export row 0's average strategy compactly, then reload it.
let mut probs = [0.0; 3];
node.average_into(0, &mut probs);
let codes: Vec<u16> = quantize_dist(&probs);
let reloaded = dequantize_dist::<u16>(&codes); // decodes and renormalizes
assert!((reloaded.iter().sum::<f32>() - 1.0).abs() < 1e-6);
```

## Memory layouts

`BatchedMatcher` accepts an optional third type parameter that selects the lane
stores used for cumulative regret and the running strategy average. The default
(`F32Full`) reproduces the previous all-f32 behavior; alternative layouts trade
a small amount of precision for a meaningful reduction in RAM footprint:

| Layout | Regret | Strategy | Bytes / cell | Notes |
|--------|--------|----------|--------------|-------|
| `F32Full` (default) | f32 | f32 sum | 8 | Exact; matches scalar matchers bit-for-bit |
| `Int32Full` | i32 fixed-point | f32 sum | 8 | Constant regret quantum at any magnitude; optional floor |
| `HalfStrategy` | f32 | u16 avg | 6 (+4/row) | f32 regret, bounded u16 average; per-row W |
| `HalfStrategyShared` | f32 | u16 avg (shared W) | 6† | Like `HalfStrategy` but single shared W; `update_batch`-only |
| `Int32HalfShared` | i32 fixed-point | u16 avg (shared W) | 6† | rs-poker's preflop layout; `update_batch`-only |
| `Int32NoAverage` | i32 fixed-point | none | 4 | rs-poker's postflop layout; `average_into` panics — snapshot `current_into` |
| `HalfRegret` | i16 scaled | f32 sum | 6 (+4/row) | i16 regret with per-row scale; rejected (see note) |
| `HalfBoth` | i16 scaled | u16 avg | 4 (+8/row) | per-row W; rejected (see note) |
| `HalfBothShared` | i16 scaled | u16 avg (shared W) | 4† | Like `HalfBoth` but single shared W; `update_batch`-only; rejected (see note) |

Swapping layouts is a one-type change — `BatchedMatcher::<Dcfr, Local>` becomes
`BatchedMatcher::<Dcfr, Local, HalfStrategy>` — and `average_into`, `seed`, and
the rest of the API are unchanged.

**Footprint accounting.** The u16 strategy lane stores a per-row f32 weight `W`
(one 4-byte cell per row, independent of `num_actions`). At large action counts
this term is negligible, but at small action counts — e.g. rs-poker's ~3-action
information sets — it can shrink the net saving from ~25% to ~8% total (for
`HalfStrategy` with 3 actions: 3 × 4 B data + 4 B weight vs. 3 × 4 B data,
roughly `4/(3×4+4) ≈ 25%` of the strategy lane but only ~8% of the full
regret+strategy footprint). The `*Shared` variants (`HalfStrategyShared`,
`HalfBothShared`) replace the per-row weight vector with a single shared cell,
restoring the full ~25% strategy-lane cut at any action count. † The `†` rows
therefore achieve the same headline savings as the plain variants, but with a
stricter contract: the shared `W` is valid **only** when every row advances on
every tick, i.e. the matcher is driven exclusively via `update_batch`. Calling
`update_row(r)` on a shared-weight layout with `r ≠ 0` leaves the per-row
average undefined; use `HalfStrategy` or `HalfBoth` if you need independent
per-row updates.

> **Note:** The `HalfRegret`, `HalfBoth`, and `HalfBothShared` layouts use
> per-row-scaled i16 quantization for regret. rs-poker's paired out-of-sample
> exploitability A/B measured a **45× regression** for that lane — the per-row
> rescale loses exactly the small regret differences that decide marginal
> hands — so they are kept for experiments only. The supported 4-byte regret
> lane is `Int32Regret`, below.

### int32 fixed-point regret, floors, and pruning

f32 stops learning at large magnitude: with a 24-bit mantissa, once a row's
cumulative regret reaches 10⁶ an increment below ~0.06 is rounded away. The
`Int32*` layouts store regret as fixed-point i32 at a caller-chosen scale
(default 100 codes per unit, a 0.01 quantum up to ±2.1 × 10⁷ units) and apply
each update in code space, so the quantum is constant at any magnitude. A
configurable floor clamps stored regret from below so a pruned action can
always recover; derive it from the rule with `dominated_regret_after` rather
than reusing another solver's constant.

Regret-based pruning skips traversal into an action and therefore has no reward
for it; `update_row_masked_with` / `update_batch_masked_with` leave inactive
actions' regret exactly as stored and renormalise the expected value over the
traversed actions:

```rust
use little_sorry::{
    BatchedMatcher, Dcfr, DiscountParams, Int32Config, Int32NoAverage, Local, Scratch,
    dominated_regret_after,
};

let params = DiscountParams::RECOMMENDED;
let floor = 5.0 * dominated_regret_after::<Dcfr>(&params, 200, 1.0);
let node = BatchedMatcher::<Dcfr, Local, Int32NoAverage>::with_regret_config(
    8, 3, params, Int32Config { scale: 100.0, floor },
);
let mut scratch = Scratch::new(3);
let mut expected = [0.0; 8];
node.update_batch_masked_with(
    &mut scratch,
    |action, _row| [1.0, -0.5, 0.2][action],
    |action, _row| action != 2, // action 2 was pruned this tick
    &mut expected,
);
```

For checkpoints, `regret_lane()` exposes the lane so `Int32Regret::codes_row`
/ `set_codes_row` can export and restore raw codes without rounding through
f32.

## Building and Testing

This project uses [mise](https://mise.jdx.dev/) to manage tooling and tasks.

```bash
# Run all checks (formatting, linting, tests, TOML validation)
mise check

# Run tests
mise run check:test:nextest

# Run benchmarks
cargo bench --features rps

# Run the RPS example
cargo run --release --features rps --bin run-rps
```

## License

Licensed under the Apache License, Version 2.0.
