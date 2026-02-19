# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 6 CFR algorithm variants via the `RegretMinimizer` trait:
  - CFR+ (regret clipping at zero)
  - Discounted CFR (DCFR) with configurable parameters
  - DCFR+ (combines DCFR discounting with CFR+ clipping)
  - Linear CFR (linear time-weighted regrets)
  - Predictive CFR+ (PCFR+) using future regret predictions
  - Predictive DCFR+ (PDCFR+) combining DCFR+ and predictive updates
- `RegretMinimizer` trait for generic algorithm usage
- `compare_algorithms` binary for comparing convergence across variants

### Changed

- Zero-allocation hot path — no heap allocations during `update_regret`
- Minimal dependencies: removed `ndarray`, `rand_distr`, and `thiserror`
- Only `rand` remains as a dependency

### Removed

- `RegretMatcher` struct (replaced by trait-based design with multiple variants)
- `ndarray` dependency (replaced with plain slices and fixed arrays)
- `rand_distr` dependency (replaced with custom `WeightedAliasIndex`)
- `thiserror` dependency

## [1.1.0] - 2025-01-20

### Changed

- Replaced `once_cell` crate with `std::cell`

## [1.0.0] - 2025-01-19

- Initial release with `RegretMatcher` for CFR regret minimization
