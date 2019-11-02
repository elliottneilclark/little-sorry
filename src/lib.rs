#![deny(clippy::all)]
#![deny(clippy::pedantic)]

pub mod regret_matcher;
pub mod rps;

pub use self::regret_matcher::RegretMatcher;
