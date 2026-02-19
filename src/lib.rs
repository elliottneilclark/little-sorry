#![deny(clippy::all)]
//! # Little Sorry
//!
//! Regret minimization algorithms for finding Nash equilibrium strategies
//! in imperfect-information games.
//!
//! ## Available Algorithms
//!
//! All algorithms implement the [`RegretMinimizer`] trait:
//!
//! | Type | Algorithm | Key Property |
//! |------|-----------|--------------|
//! | [`CfrPlusRegretMatcher`] | CFR+ | Regret clipping at zero |
//! | [`DiscountedRegretMatcher`] | DCFR | Configurable time-based discounting |
//! | [`DcfrPlusRegretMatcher`] | DCFR+ | DCFR discounting + CFR+ clipping |
//! | [`LinearCfrRegretMatcher`] | Linear CFR | Linear time-weighted regrets |
//! | [`PcfrPlusRegretMatcher`] | PCFR+ | Predictive future regret estimates |
//! | [`PdcfrPlusRegretMatcher`] | PDCFR+ | DCFR+ discounting + predictive updates |
//!
//! ## Quick Start
//!
//! ```
//! use little_sorry::{CfrPlusRegretMatcher, RegretMinimizer};
//!
//! let mut matcher = CfrPlusRegretMatcher::new(3);
//! for _ in 0..1000 {
//!     matcher.update_regret(&[1.0, -0.5, 0.2]);
//! }
//! let strategy = matcher.best_weight();
//! assert!((strategy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
//! ```

pub mod cfr_plus;
pub mod dcfr;
pub mod dcfr_plus;
pub mod discount;
pub mod linear_cfr;
pub mod pcfr_plus;
pub mod pdcfr_plus;
pub mod regret_minimizer;

#[cfg(feature = "rps")]
pub mod rps;

pub use cfr_plus::CfrPlusRegretMatcher;
pub use dcfr::DiscountedRegretMatcher;
pub use dcfr_plus::DcfrPlusRegretMatcher;
pub use discount::DiscountParams;
pub use linear_cfr::LinearCfrRegretMatcher;
pub use pcfr_plus::PcfrPlusRegretMatcher;
pub use pdcfr_plus::PdcfrPlusRegretMatcher;
pub use regret_minimizer::RegretMinimizer;

// Re-export for backwards compatibility
pub use cfr_plus::CfrPlusRegretMatcher as RegretMatcher;
