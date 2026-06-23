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
//!
//! ## Batched, storage-generic matchers
//!
//! For large or concurrent solves, [`BatchedMatcher`] owns many information sets
//! ("rows") that advance on one shared iteration clock, so a rule's
//! time-dependent factors are computed once per batch rather than once per row.
//! It is generic over both the update rule (one of [`Dcfr`], [`DcfrPlus`],
//! [`LinearCfr`], [`PcfrPlus`], [`PdcfrPlus`]) and the cell backend ([`Local`]
//! for zero-overhead single-threaded use, [`Atomic`] for lock-free concurrent
//! updates through a shared reference). Swapping either is a one-type change.
//!
//! The solved average strategy reads out the same way for every rule and can be
//! exported compactly with [`quantize_dist`] / [`dequantize_dist`]:
//!
//! ```
//! use little_sorry::{BatchedMatcher, Dcfr, DiscountParams, Local};
//! use little_sorry::{dequantize_dist, quantize_dist};
//!
//! // One node owning 8 abstraction classes over 3 actions.
//! let node = BatchedMatcher::<Dcfr, Local>::new(8, 3, DiscountParams::RECOMMENDED);
//! let mut expected = [0.0; 8];
//! for _ in 0..1000 {
//!     // The caller supplies values from whatever layout it holds.
//!     node.update_batch(|action, _row| [1.0, -0.5, 0.2][action], &mut expected);
//! }
//!
//! let mut probs = [0.0; 3];
//! node.average_into(0, &mut probs); // normalized average strategy, any rule
//! let codes = quantize_dist::<u16>(&probs); // compact on-disk form
//! let reloaded = dequantize_dist::<u16>(&codes); // decodes AND renormalizes
//! assert!((reloaded.iter().sum::<f32>() - 1.0).abs() < 1e-6);
//! ```
//!
//! The memory layout is a third type parameter (`F32Full` by default). Swap it
//! to cut footprint without changing the API:
//!
//! ```
//! use little_sorry::{BatchedMatcher, Dcfr, DiscountParams, Local, HalfStrategy};
//! // f32 regret + u16 average strategy — ~25% smaller, same f32-facing API.
//! let node = BatchedMatcher::<Dcfr, Local, HalfStrategy>::new(8, 3, DiscountParams::RECOMMENDED);
//! node.seed(|action, _row| [10.0, 0.0, 0.0][action], 100); // warm-start row regret
//! let mut probs = [0.0; 3];
//! node.average_into(0, &mut probs);
//! assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
//! ```

pub mod batched_matcher;
pub mod cfr_plus;
pub mod dcfr;
pub mod dcfr_plus;
pub mod discount;
pub mod lane;
pub mod linear_cfr;
pub mod pcfr_plus;
pub mod pdcfr_plus;
pub mod quantize;
pub mod regret_minimizer;
pub mod rules;
pub mod storage;
pub mod update_rule;

mod probability;
mod scaled_int;
mod unit_fixed;
mod vector_ops;

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

// Batched, storage-generic machinery.
pub use batched_matcher::BatchedMatcher;
// Memory layout types.
pub use lane::{
    F32Full, F32Regret, F32SumStrategy, HalfBoth, HalfBothShared, HalfRegret, HalfStrategy,
    HalfStrategyShared, Int16Regret, Layout, RegretLane, StrategyLane, U16AvgStrategy,
    U16AvgStrategyShared,
};
pub use quantize::{FixedWidth, dequantize_dist, quantize_dist};
pub use rules::{Dcfr, DcfrPlus, LinearCfr, PcfrPlus, PdcfrPlus, PlusDiscount};
pub use storage::{Atomic, Local};
pub use update_rule::UpdateRule;

// Re-export for backwards compatibility
pub use cfr_plus::CfrPlusRegretMatcher as RegretMatcher;
