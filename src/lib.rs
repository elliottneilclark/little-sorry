#![deny(clippy::all)]

pub mod errors;
pub mod regret_matcher;

#[cfg(feature = "rps")]
pub mod rps;

pub use self::regret_matcher::RegretMatcher;
