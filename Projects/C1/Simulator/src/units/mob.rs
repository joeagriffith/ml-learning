// Basic unit template and functions
// use crate::game_logic::Coords;
use crate::pathfinding::BFS;
use crate::board::{MAP_BOUNDS, Coords, Grid, MapEdge};
use super::{traits::*, Structure};
use std::fmt::{Debug, Formatter, Error};

#[derive(Clone, Debug)]
pub enum MobileType {
    Scout,
    Demolisher,
    Interceptor,
}

pub struct FinishedPath;

pub struct MobileUnit {
    start_health: f32,
    curr_health: f32,
    range: f32,
    damage: f32,
    speed: usize,
    target_edge: MapEdge,

    coords: Coords,
    move_frame: usize,
    is_inter: bool,

    path: Vec<(Coords, usize)>,
    path_i: usize,
    mob_type: MobileType,
    spaces_moved: usize,
    last_direction: usize,
}

impl Debug for MobileUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "MobileUnit {{ coords: {:?}, mob_type: {:?} }}", self.coords, self.mob_type);
        Ok(())
    }
}

impl MobileUnit {
    pub fn new(mob_type:MobileType, coords:Coords) -> MobileUnit {
        match mob_type {
            MobileType::Scout => {
                MobileUnit {
                    start_health: 15.0,
                    curr_health: 15.0,
                    range: 3.5,
                    damage: 2.0,
                    speed: 1,
                    target_edge: get_target_edge(coords),
                    coords,
                    move_frame: 0,
                    is_inter: false,
                    path: Vec::new(),
                    path_i: 0,
                    mob_type: MobileType::Scout,
                    spaces_moved: 0,
                    last_direction: 0,
                }
            },
            MobileType::Demolisher => {
                MobileUnit {
                    start_health: 5.0,
                    curr_health: 5.0,
                    range: 4.5,
                    damage: 8.0,
                    speed: 2,
                    target_edge: get_target_edge(coords),
                    coords,
                    move_frame: 0,
                    is_inter: false,
                    path: Vec::new(),
                    path_i: 0,
                    mob_type: MobileType::Demolisher,
                    spaces_moved: 0,
                    last_direction: 0,
                }
            },
            MobileType::Interceptor => {
                MobileUnit {
                    start_health: 40.0,
                    curr_health: 40.0,
                    range: 4.5,
                    damage: 20.0,
                    speed: 4,
                    target_edge: get_target_edge(coords),
                    coords,
                    move_frame: 0,
                    is_inter: true,
                    path: Vec::new(),
                    path_i: 0,
                    mob_type: MobileType::Interceptor,
                    spaces_moved: 0,
                    last_direction: 0,
                }
            },
        }
    }

    pub fn take_damage(&mut self, amount:f32) {
        self.curr_health -= amount;
    }

    pub fn is_dead(&self) -> bool {
        self.curr_health <= 0.0
    }

    pub fn get_health(&self) -> f32 {
        self.curr_health
    }

    pub fn get_damage(&self) -> f32 {
        self.damage
    }

    pub fn get_type(&self) -> MobileType {
        self.mob_type.clone()
    }

    pub fn add_shielding(&mut self, amount:f32) {
        self.curr_health += amount;
    }

    pub fn get_last_direction(&self) -> usize {
        self.last_direction
    }

    pub fn get_coords(&self) -> Coords {
        self.coords
    }

    pub fn get_range(&self) -> f32 {
        self.range
    }

    pub fn get_target_edge(&self) -> MapEdge {
        self.target_edge
    }

    pub fn set_path(&mut self, path:Vec<(Coords, usize)>) {
        self.path = path;
        self.path_i = 0;
    } 

    pub fn path_is_empty(&self) -> bool {
        self.path.is_empty()
    }

    pub fn move_frame(&mut self) -> Result<(), FinishedPath> {
        self.move_frame += 1;
        if self.move_frame >= self.speed {
            self.advance()?;
            self.move_frame = 0;
        }
        Ok(())
    }

    fn advance(&mut self) -> Result<(), FinishedPath> {
        if self.path_i < self.path.len() {
            self.coords = self.path[self.path_i].0.clone();
            self.last_direction = self.path[self.path_i].1;
            self.path_i += 1;
            self.spaces_moved += 1;
            Ok(())
        } else {
            Err(FinishedPath)
        }
    }

    pub fn self_destruct(&mut self, board:&mut Grid<Option<Structure>>) {
        self.curr_health = 0.0;
        if self.spaces_moved >= 5 {
            let points = self.get_coords().neighbors_with_diag();
            for coords in points {
                if MAP_BOUNDS.is_in_arena(coords) {
                    if let Some(struc) = board.get_mut(coords).unwrap() {
                        struc.take_damage(self.start_health);
                    }
                }
            }
        }
    }

    pub fn get_char(&self) -> char {
        match self.mob_type {
            MobileType::Demolisher => 'D',
            MobileType::Scout => 'x',
            MobileType::Interceptor =>'I',
        }
    }

    pub fn get_str(&self) -> &str {
        match self.mob_type {
            MobileType::Demolisher => " D",
            MobileType::Scout => " x",
            MobileType::Interceptor =>" I",
        }
    }
}
fn get_target_edge(start_coords:Coords) -> MapEdge {
        let edge:MapEdge;
        let coords = [start_coords.x, start_coords.y];
        if coords[1] == -coords[0] + 13 {
            edge = MapEdge::TopRight;
        } else
        if coords[1] == coords[0] - 14 {
            edge = MapEdge::TopLeft;
        } else 
        if coords[1] == -coords[0] + 41 {
            edge = MapEdge::BottomLeft;
        } else 
        if coords[1] == coords[0] + 14 {
            edge = MapEdge::BottomRight;
        }
        else {
            panic!("Pathfinding could not determine which side the starting point: ({:?}) was on.", start_coords);
        }

        edge
    }