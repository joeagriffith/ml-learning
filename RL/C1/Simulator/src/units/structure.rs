// Structure components
use super::traits::*;
use std::fmt::{Debug, Formatter, Error};
use crate::board::*;


#[derive(Clone, Copy, PartialEq)]
pub enum StructureType {
    Wall,
    Turret,
    Support,
}

/// 
#[derive(Clone)]
pub struct Structure {
    health: f32,
    upgraded: bool,    
    range: Option<f32>,
    damage: Option<f32>,
    shielding: Option<f32>,
    coords: Coords,
    buffed_mobs: Vec<usize>, // Mob GUIDs
    struc_type: StructureType,
    upgrade_cost: f32,
}

impl Structure {
    pub fn new(struc_type:StructureType, coords:Coords) -> Structure {
        match struc_type {
            StructureType::Wall => {
                Structure {
                    health: 60.0,
                    coords, 
                    upgraded: false,
                    range: None,
                    damage: None,
                    shielding: None,
                    buffed_mobs: Vec::new(),
                    struc_type,
                    upgrade_cost: 1.0,
                }
            },
            StructureType::Support => {
                Structure {
                    health: 30.0,
                    coords,
                    upgraded: false,
                    range: Some(3.5),
                    damage: None,
                    shielding: Some(3.0),
                    buffed_mobs: Vec::new(),
                    struc_type,
                    upgrade_cost: 4.0,
                }
            },
            StructureType::Turret => {
                Structure {
                    health: 75.0,
                    upgraded: false,
                    coords,
                    range: Some(2.5),
                    damage: Some(5.0),
                    buffed_mobs: Vec::new(),
                    shielding: None,
                    struc_type,
                    upgrade_cost: 4.0,
                }
            }
        }
    }

    pub fn get_upgrade_cost(&self) -> f32 {
        self.upgrade_cost
    }

    pub fn take_damage(&mut self, amount:f32) {
        self.health -= amount;
    }

    pub fn is_dead(&self) -> bool {
        self.health <= 0.0
    }
    pub fn get_damage(&self) -> Option<f32> {
        self.damage
    }

    pub fn get_shielding(&self) -> Option<f32> {
        self.shielding
    }

    pub fn get_buffed_mobs(&self) -> &Vec<usize> {
        &self.buffed_mobs
    }

    pub fn add_buffed_mob(&mut self, id:usize) {
        self.buffed_mobs.push(id);
    }

    pub fn clear_buffed_mobs(&mut self) {
        self.buffed_mobs.clear();
    }

    pub fn get_range(&self) -> Option<f32> {
        self.range
    }

    pub fn get_health(&self) -> f32 {
        self.health
    }

    pub fn get_type(&self) -> StructureType {
        self.struc_type
    }

    pub fn is_upgraded(&self) -> bool {
        self.upgraded
    }

    pub fn upgrade(&mut self) {
        if !self.upgraded {
            self.upgraded = true;
            match self.struc_type {
                StructureType::Wall => {
                    self.health += 60.0;
                },
                StructureType::Support => {
                    self.range = Some(7.0);
                    if self.coords.y < 14 {
                        self.shielding = Some(4.0 + 0.3 * self.coords.y as f32);
                    } else {
                        self.shielding = Some(4.0 + 0.3 * (27 - self.coords.y) as f32);
                    }
                },
                StructureType::Turret => {
                    self.damage = Some(15.0);
                    self.range = Some(3.5);
                }
            }
        }
    }

    pub fn get_char(&self) -> char {
        match self.struc_type {
            StructureType::Wall => 'W',
            StructureType::Turret => 'T',
            StructureType::Support => 'S',
        }
    }

    pub fn get_str(&self) -> &str {
        match self.struc_type {
            StructureType::Wall => " W",
            StructureType::Turret => " T",
            StructureType::Support => " S",
        }
    }
}


impl Debug for Structure {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self.struc_type {
            StructureType::Wall => f.write_str("W"),
            StructureType::Turret => f.write_str("T"),
            StructureType::Support => f.write_str("S"),
        }
    }
}