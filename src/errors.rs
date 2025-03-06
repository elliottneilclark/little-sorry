use rand_distr::weighted::Error as WeightedError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LittleError {
    #[error("error with weigths")]
    Weights(#[from] WeightedError),

    #[error("unknown little-sorry error")]
    Unknown,
}
