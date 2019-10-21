use little_sorry::RegretMatcher;
use ndarray::prelude::*;
use std::cmp;
use std::mem;
#[macro_use]
extern crate lazy_static;

#[allow(dead_code)]
#[repr(usize)]
#[derive(Debug)]
enum RPSAction {
    Rock = 0,
    Paper = 1,
    Scissors = 2,
}

lazy_static! {
    static ref ROCK_REWARD: Array1<f64> = array![0.0f64, 1.0f64, -1.0f64];
    static ref PAPER_REWARD: Array1<f64> = array![-1.0f64, 0.0f64, 1.0f64];
    static ref SCISSOR_REWARD: Array1<f64> = array![1.0f64, -1.0f64, 0.0f64];
}
impl RPSAction {
    pub fn to_reward(&self) -> ArrayView1<f64> {
        match self {
            RPSAction::Rock => ROCK_REWARD.view(),
            RPSAction::Paper => PAPER_REWARD.view(),
            RPSAction::Scissors => SCISSOR_REWARD.view(),
        }
    }
}

impl From<usize> for RPSAction {
    fn from(i: usize) -> RPSAction {
        unsafe {
            mem::transmute(cmp::max(
                cmp::min(i, RPSAction::Scissors as usize),
                RPSAction::Rock as usize,
            ))
        }
    }
}

fn main() {
    let mut rg_one = RegretMatcher::new(3);
    let mut rg_two = RegretMatcher::new(3);

    dbg!(&rg_one);
    for _i in 0..1_000_000_000 {
        let a1 = RPSAction::from(rg_one.next_action());
        let a2 = RPSAction::from(rg_two.next_action());
        rg_two.update_regret(a1.to_reward());
        rg_one.update_regret(a2.to_reward());
    }

    dbg!(&rg_one);
    dbg!(rg_one.best_weight());
}
