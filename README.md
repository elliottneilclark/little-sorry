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
little-sorry = "3.2.0"
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

| Layout | Regret | Strategy | Relative footprint | Notes |
|--------|--------|----------|--------------------|-------|
| `F32Full` (default) | f32 | f32 sum | 100% | Exact; matches scalar matchers bit-for-bit |
| `HalfStrategy` | f32 | u16 avg | ~75% | f32 regret, bounded u16 average; per-row W |
| `HalfStrategyShared` | f32 | u16 avg (shared W) | ~75%† | Like `HalfStrategy` but single shared W; `update_batch`-only |
| `HalfRegret` | i16 scaled | f32 sum | ~75% | i16 regret with per-row scale; experimental |
| `HalfBoth` | i16 scaled | u16 avg | ~50% | Deepest cut; per-row W; experimental |
| `HalfBothShared` | i16 scaled | u16 avg (shared W) | ~50%† | Like `HalfBoth` but single shared W; `update_batch`-only; experimental |

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
> per-row-scaled i16 quantization for regret. They are marked **experimental**:
> exploitability equivalence with the f32 baseline has not been formally
> verified, and lossy requantization on every write may interact with split α/β
> discounting (DCFR variants). Test carefully before using in production
> solvers.

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
cargo run --release --features rps --bin run_rps
```

## License

Licensed under the Apache License, Version 2.0.
