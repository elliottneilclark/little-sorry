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

## Getting Started

Add this to your `Cargo.toml`:

```toml
[dependencies]
little-sorry = "2.0.0"
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
