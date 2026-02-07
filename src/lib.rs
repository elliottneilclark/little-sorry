#![deny(clippy::all)]

pub mod cfr_plus;
pub mod dcfr;
pub mod dcfr_plus;
pub mod discount;
pub mod errors;
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
