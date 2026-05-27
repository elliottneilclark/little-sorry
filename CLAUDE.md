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

- **RegretMinimizer** (`src/regret_minimizer.rs`): Core trait shared by every CFR variant. Defines the regret-matching interface — required methods `new`, `update_regret`, `num_updates`, `current_strategy`, `cumulative_strategy`, `cumulative_regret`; default-provided diagnostics `next_action`, `best_weight`, `regret_weight_total`, `average_regret`. Also holds the `regret_match` helper.

- **Regret matchers** — one module per algorithm, each implementing `RegretMinimizer`:
  - `CfrPlusRegretMatcher` (`src/cfr_plus.rs`) — CFR+; also re-exported as `RegretMatcher`.
  - `DiscountedRegretMatcher` (`src/dcfr.rs`) — DCFR.
  - `DcfrPlusRegretMatcher` (`src/dcfr_plus.rs`) — DCFR+.
  - `LinearCfrRegretMatcher` (`src/linear_cfr.rs`) — Linear CFR.
  - `PcfrPlusRegretMatcher` (`src/pcfr_plus.rs`) — PCFR+ (predictive).
  - `PdcfrPlusRegretMatcher` (`src/pdcfr_plus.rs`) — PDCFR+ (predictive + discounted).

- **DiscountParams** (`src/discount.rs`): Discount-factor parameters (alpha/beta/gamma) used by the discounted variants.

- **RPSRunnerGeneric** (`src/rps.rs`): Example harness using Rock-Paper-Scissors, generic over any `RegretMinimizer` (`RPSRunner` is the `CfrPlusRegretMatcher` alias). Feature-gated behind `rps`.

## Key Design Patterns

- Numerical operations use plain `Vec<f32>` / `&[f32]` slices (no `ndarray`). Shared numeric kernels live in two crate-internal modules: `vector_ops` (element-wise f32 kernels: `dot`, `add_assign`, `scaled_add_assign`, `discounted_accumulate`) and `probability` (simplex operations: uniform construction, normalization, categorical sampling)
- The RPS module uses `unsafe` transmute for bounded enum conversion (usize to RPSAction), clamped to valid range
- RPS reward vectors are compile-time `const [f32; 3]` arrays (`ROCK_REWARD`/`PAPER_REWARD`/`SCISSOR_REWARD`), returned as `&'static [f32]`
- Clippy lints enforced with `#![deny(clippy::all)]`; the binary also denies `clippy::pedantic`
- Uses Rust nightly edition 2024

## Feature Flags

- `rps`: Enables the Rock-Paper-Scissors example module, binary, and benchmarks
