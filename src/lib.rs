#![deny(clippy::all)]

pub mod errors;
pub mod regret_matcher;
pub mod rps;

pub use self::regret_matcher::RegretMatcher;
