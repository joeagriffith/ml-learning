// Player data template and components
use crate::utils::Dead;

pub struct InsufficientFunds;

#[derive(Clone)]
pub struct PlayerData {
    health: i32,
    structure_pts: f32,
    mobile_pts: f32,
}

impl PlayerData {
    pub fn new() -> Self {
        Self {
            health: 30,
            structure_pts: 40.0,
            mobile_pts: 5.0,
        }
    }

    pub fn spend_structure_pts(&mut self, cost: f32) -> Result<(), InsufficientFunds> {
        if cost <= self.structure_pts {
            self.structure_pts -= cost;
            Ok(())
        } else {
            Err(InsufficientFunds)
        }
    }

    pub fn spend_mobile_pts(&mut self, cost: f32) -> Result<(), InsufficientFunds> {
        if cost <= self.mobile_pts {
            self.mobile_pts -= cost;
            Ok(())
        } else {
            Err(InsufficientFunds)
        }
    }

    pub fn decay(&mut self) {
        self.mobile_pts = (self.mobile_pts * 7.5).round()/10.0;
    }

    pub fn turn_reward(&mut self, turn:usize) {
        self.structure_pts += 5.0;

        let bonus = turn / 10;
        self.mobile_pts += 5.0 + bonus as f32;
    }

    pub fn damage_reward(&mut self, dmg_dealt:usize) {
        self.structure_pts += dmg_dealt as f32;
    }

    pub fn is_dead(&self) -> bool {
        self.health <= 0
    }

    pub fn take_damage(&mut self) {
        if self.health > 0 {
            self.health -= 1;
        }
    }

    pub fn dbg_moneyhack(&mut self) {
        self.structure_pts = 100000.0;
        self.mobile_pts = 100000.0;
    }
}