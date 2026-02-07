# Build and Test Commands

This project uses mise to manage tooling and tasks. Prefer mise commands over raw cargo commands.

```bash
# Run all checks (formatting, linting, tests, TOML validation)
mise check

# Individual checks
mise run check:fmt          # Check code formatting
mise run check:clippy       # Check for lints (all features enabled)
mise run check:test:nextest # Run tests with nextest
mise run check:test:docs    # Test documentation examples
mise run check:taplo:lint   # Lint TOML files

# Fix issues
mise fix                    # Run all fixers
mise run fix:fmt            # Format code
mise run fix:clippy         # Auto-fix lint issues
mise run fix:taplo:format   # Format TOML files

# Run the RPS example binary
cargo run --release --features rps --bin run_rps

# Run benchmarks (requires rps feature)
cargo bench --features rps
```

# Architecture

This is a Rust library implementing regret minimization algorithms for game theory applications (CFR - Counterfactual Regret Minimization).

## Core Components

- **RegretMatcher** (`src/regret_matcher.rs`): The main algorithm implementation. Tracks probability distributions over actions, cumulative regrets, and updates strategy based on regret matching. Uses `WeightedAliasIndex` for O(1) action sampling.

- **RPSRunner** (`src/rps.rs`): Example implementation using Rock-Paper-Scissors as a demonstration game. Feature-gated behind `rps`. Shows how to use `RegretMatcher` for a two-player zero-sum game.

- **LittleError** (`src/errors.rs`): Error type using `thiserror`, primarily wrapping weight distribution errors.

## Key Design Patterns

- Uses `ndarray` for numerical operations (Array1<f32> for probability/reward vectors)
- The RPS module uses `unsafe` transmute for bounded enum conversion (usize to RPSAction), clamped to valid range
- Static reward arrays use `LazyLock` for lazy initialization
- Clippy lints enforced with `#![deny(clippy::all)]`; the binary also denies `clippy::pedantic`
- Uses Rust nightly edition 2024

## Feature Flags

- `rps`: Enables the Rock-Paper-Scissors example module, binary, and benchmarks
