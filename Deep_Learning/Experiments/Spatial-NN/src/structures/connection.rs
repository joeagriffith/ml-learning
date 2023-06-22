use crate::config::{CONNECTION_MAX_CHECK, MAX_WEIGHT, CONNECTION_ACTIVE_DURATION};
use std::time::{SystemTime, Duration};
use rand::prelude::*;

pub struct Connection {
    weight: f32,
    activated: bool,
    check_counter: usize,
    last_activated: SystemTime,
}

impl Connection {
    pub fn new() -> Self {
        Self {
            weight: rand::thread_rng().gen_range(-MAX_WEIGHT..MAX_WEIGHT),
            activated: false,
            check_counter: 0,
            last_activated: SystemTime::now() - Duration::from_secs(1),
        }
    }

    pub fn new_one() -> Self {
        Self {
            weight: 1.0,
            activated: false,
            check_counter: 0,
            last_activated: SystemTime::now() - Duration::from_secs(1),
        }
    }

    pub fn activate(&mut self) {
        if !self.activated {
            self.activated = true;
            self.check_counter = 0;
            self.last_activated = SystemTime::now();
        }
    }

    pub fn check(&mut self) -> Option<f32> {
        if self.activated {
            // if self.check_counter < CONNECTION_MAX_CHECK {
            //     self.check_counter += 1;
            //     return Some(self.weight)
            // } else {
            //     self.activated = false;
            // }
            if self.last_activated.elapsed().unwrap() < CONNECTION_ACTIVE_DURATION {
                return Some(self.weight)
            } else {
                self.activated = false;
            }
        }
        None
    }

    pub fn is_active(&self) -> bool {
        self.activated
    }
}