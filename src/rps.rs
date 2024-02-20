use crate::errors::LittleError;
use crate::regret_matcher::RegretMatcher;
use ndarray::prelude::*;
use once_cell::sync::Lazy;
use std::cmp;
use std::mem;

use std::vec::Vec;

#[allow(dead_code)]
#[repr(usize)]
#[derive(Debug, Clone, Copy)]
enum RPSAction {
    Rock = 0,
    Paper = 1,
    Scissors = 2,
}

static ROCK_REWARD: Lazy<Array1<f32>> = Lazy::new(|| array![0.0_f32, 1.0_f32, -1.0_f32]);
static PAPER_REWARD: Lazy<Array1<f32>> = Lazy::new(|| array![-1.0_f32, 0.0_f32, 1.0_f32]);
static SCISSOR_REWARD: Lazy<Array1<f32>> = Lazy::new(|| array![1.0_f32, -1.0_f32, 0.0_f32]);

impl RPSAction {
    pub fn to_reward(self) -> ArrayView1<'static, f32> {
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
    pending_reward_one: Array1<f32>,
    pending_reward_two: Array1<f32>,
}

impl Default for RPSRunner {
    #[must_use]
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl RPSRunner {
    pub fn new() -> Result<Self, LittleError> {
        Ok(Self {
            matcher_one: RegretMatcher::new(3)?,
            matcher_two: RegretMatcher::new(3)?,
            pending_reward_one: Array1::zeros(3),
            pending_reward_two: Array1::zeros(3),
        })
    }
    pub fn run_one(&mut self) {
        let a1 = RPSAction::from(self.matcher_one.next_action());
        let a2 = RPSAction::from(self.matcher_two.next_action());

        self.pending_reward_one += &a2.to_reward();
        self.pending_reward_two += &a1.to_reward();
    }
    pub fn update_regret(&mut self) -> Result<(), LittleError> {
        self.matcher_one
            .update_regret(self.pending_reward_one.view())?;
        self.matcher_two
            .update_regret(self.pending_reward_two.view())?;

        self.pending_reward_one.fill(0.0);
        self.pending_reward_two.fill(0.0);
        Ok(())
    }
    #[must_use]
    pub fn best_weight(&self) -> Vec<f32> {
        self.matcher_one.best_weight()
    }
}
