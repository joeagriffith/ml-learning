use std::time::{SystemTime, Duration};

use crate::config::{REFRACTORY_DURATION, THRESHOLD_NODE_ACTIVATION};

#[derive(Clone)]
pub struct Node {
    prev_actv_time: SystemTime,
    activated: bool,
}

pub enum ActivationError {
    Cooldown,
    InsufficientActivation,
}
pub struct Cooldown;

impl Node {
    pub fn new() -> Self {
        Self {
            prev_actv_time: SystemTime::now() - Duration::from_secs(1),
            activated: false,
        }
    }

    pub fn try_activate(&mut self, activation:f32) -> Result<(), ActivationError> {
        if activation > THRESHOLD_NODE_ACTIVATION {

            if let Ok(dur) = self.prev_actv_time.elapsed() {
                if dur > REFRACTORY_DURATION {
                    self.prev_actv_time = SystemTime::now();
                    self.activated = true;
                    return Ok(())
                } else {
                    self.activated = false;
                    return Err(ActivationError::Cooldown)
                }
            } else {
                self.activated = false;
                panic!("try_activate() failed to calculate duration since last activation.");
            }
        } else {
            self.activated = false;
            Err(ActivationError::InsufficientActivation)
        }
    }

    pub fn is_activated(&self) -> bool {
        self.activated
    }
}