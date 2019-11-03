use crate::regret_matcher::RegretMatcher;
use lazy_static::*;
use ndarray::prelude::*;
use std::cmp;
use std::mem;

use std::vec::Vec;

#[allow(dead_code)]
#[repr(usize)]
#[derive(Debug)]
enum RPSAction {
    Rock = 0,
    Paper = 1,
    Scissors = 2,
}

lazy_static! {
    static ref ROCK_REWARD: Array1<f64> = array![0.0_f64, 1.0_f64, -1.0_f64];
    static ref PAPER_REWARD: Array1<f64> = array![-1.0_f64, 0.0_f64, 1.0_f64];
    static ref SCISSOR_REWARD: Array1<f64> = array![1.0_f64, -1.0_f64, 0.0_f64];
}
impl RPSAction {
    pub fn to_reward(&self) -> ArrayView1<f64> {
        match self {
            Self::Rock => ROCK_REWARD.view(),
            Self::Paper => PAPER_REWARD.view(),
            Self::Scissors => SCISSOR_REWARD.view(),
        }
    }
}

impl From<usize> for RPSAction {
    fn from(i: usize) -> Self {
        unsafe {
            mem::transmute(cmp::max(
                cmp::min(i, Self::Scissors as usize),
                Self::Rock as usize,
            ))
        }
    }
}

#[derive(Debug, Clone)]
pub struct RPSRunner {
    pub matcher_one: RegretMatcher,
    pub matcher_two: RegretMatcher,
}

impl Default for RPSRunner {
    #[must_use]
    fn default() -> Self {
        Self::new()
    }
}

impl RPSRunner {
    #[must_use]
    pub fn new() -> Self {
        Self {
            matcher_one: RegretMatcher::new(3),
            matcher_two: RegretMatcher::new(3),
        }
    }
    pub fn run_one(&mut self) {
        let a1 = RPSAction::from(self.matcher_one.next_action());
        let a2 = RPSAction::from(self.matcher_two.next_action());
        self.matcher_one.update_regret(a1.to_reward());
        self.matcher_two.update_regret(a2.to_reward())
    }
    #[must_use]
    pub fn best_weight(&self) -> Vec<f64> {
        self.matcher_one.best_weight()
    }
}
