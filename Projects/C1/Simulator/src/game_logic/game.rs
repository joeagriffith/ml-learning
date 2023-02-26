// Game logic

use super::{player::PlayerData, targeting::{Target, offensive_targetting, defensive_targetting}};
use crate::{units::{FinishedPath, MobileType, MobileUnit, Structure, StructureType}};
use crate::pathfinding::BFS;
use crate::board::*;
use crate::utils::{print_game, MobList};
use rand::prelude::*;



use std::collections::HashMap;
use std::{time::{Instant, SystemTime, Duration}, alloc::System};
use std::thread;

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum GameState {
    GameInit,
    Restore,
    Deploy,
    Action,
    GameOver,
}

pub struct Observation {
    pub structure_board: Grid<Option<Structure>>,
    pub p1_data: PlayerData,
    pub p2_data: PlayerData,
    pub turn: usize,
    pub prev_turn_dmg: [usize;2],
}
impl Observation {
    pub fn new(game:&Game) -> Self {
        Self {
            structure_board: game.structure_board.clone(),
            p1_data: game.p1_data.clone(),
            p2_data: game.p2_data.clone(),
            turn: game.turn,
            prev_turn_dmg: game.prev_turn_dmg,
        }
    }
}

pub struct Game {
    pub structure_board: Grid<Option<Structure>>,

    p1_data: PlayerData,
    pub p1_moblist: MobList,
    pub p1_struc_coords: Vec<Coords>,

    p2_data: PlayerData,
    pub p2_moblist: MobList,
    pub p2_struc_coords: Vec<Coords>,

    turn: usize,
    prev_turn_dmg: [usize;2], // [damage dealt by player 1, ... by player 2]
    new_topology: bool,
    render: bool,
    done: bool,
}

impl Game {
    pub fn new(render:bool) -> Self {
        Self {
            structure_board: Grid::from_generator(|_| None),

            p1_data: PlayerData::new(),
            p1_moblist: MobList::new(),
            p1_struc_coords: Vec::new(),

            p2_data: PlayerData::new(),
            p2_moblist: MobList::new(),
            p2_struc_coords: Vec::new(),

            turn: 0,
            prev_turn_dmg: [0,0],
            new_topology: true,
            render,
            done: false,
        }
    }

    pub fn reset(&mut self) -> Observation {
        self.structure_board = Grid::from_generator(|_| None);
        self.p1_data = PlayerData::new();
        self.p2_data = PlayerData::new();
        self.p1_moblist.clear();
        self.p2_moblist.clear();
        self.p1_struc_coords.clear();
        self.p2_struc_coords.clear();
        self.turn = 0;
        self.prev_turn_dmg = [0,0];
        self.done = false;
        Observation::new(self)
    }

    fn restore(&mut self) {

        self.p1_data.decay();
        self.p2_data.decay();
        
        self.p1_data.turn_reward(self.turn);
        self.p2_data.turn_reward(self.turn);

        self.p1_data.damage_reward(self.prev_turn_dmg[0]);
        self.p2_data.damage_reward(self.prev_turn_dmg[1]);

        // Apparently players can earn structure/mobile points from certain structures
        // self.player1.farm_reward();
        // self.player2.farm_reward();
    }


    fn fill_p2_with_walls(&mut self) {
        for x in 0..28 {
            for y in 14..26 {
                let coords = xy(x, y);
                if MAP_BOUNDS.is_in_arena(coords) {
                    let structure = Structure::new(StructureType::Wall, coords);
                    self.structure_board.set(coords, Some(structure));
                    self.p2_struc_coords.push(coords);
                }
            }
        }
    }

    pub fn step(&mut self, action:[Vec<(usize,(i32,i32))>;2]) -> (Observation, f32, bool, String) {// Observation, reward, done, info
        deploy(&action[0], &mut self.p1_moblist, &mut self.p1_struc_coords, &mut self.structure_board, &mut self.p1_data);
        deploy(&action[1], &mut self.p2_moblist, &mut self.p2_struc_coords, &mut self.structure_board, &mut self.p2_data);
        self.action();
        self.restore();
        self.turn += 1;

        (Observation::new(self), self.prev_turn_dmg[0] as f32 - self.prev_turn_dmg[1] as f32, false, "".to_string())
    }


    pub fn close() {
    }

    pub fn place_random_firewall(&mut self, player: usize) -> Result<Coords, ()> {
        let mut rng = rand::thread_rng();
        let mut i = 0;
        while i < 100{
            let mut coords = xy(rng.gen_range(0..28), rng.gen_range(0..14));
            if player == 1 {
                coords.flip();
            }

            if !MAP_BOUNDS.is_in_arena(coords) {
                continue;
            }

            if self.structure_board.get(coords).unwrap().is_none() {
                let rand_type = rng.gen_range(0..3);
                let structure = match rand_type {
                    0 => Structure::new(StructureType::Wall, coords),
                    1 => Structure::new(StructureType::Turret, coords),
                    2 => Structure::new(StructureType::Support, coords),
                    _ => panic!("Invalid structure type")
                };

                self.structure_board.set(coords, Some(structure));
                if player == 0 {
                    self.p1_struc_coords.push(coords);
                } else if player == 1 {
                    self.p2_struc_coords.push(coords);
                } else {
                    panic!("invalid player");
                }
                return Ok(coords);
            }
            i += 1;
        }
        Err(())


    }

    pub fn dbg_moneyhack(&mut self) {
        self.p1_data.dbg_moneyhack();
        self.p2_data.dbg_moneyhack();
    }

    fn action(&mut self) {
        let mut phase_done = false;
        self.new_phase();

        let mut support_frame_dur = Duration::new(0, 0);
        let mut move_frame_dur = Duration::new(0, 0);
        let mut damage_frame_dur = Duration::new(0, 0);
        let mut destroy_frame_dur = Duration::new(0, 0);

        let mut timer = Instant::now();
        self.new_topology = true;
        while !phase_done {
            // println!("support frame...");
            timer = Instant::now();
            self.support_frame();
            support_frame_dur += timer.elapsed();

            // println!("Move frame...");
            timer = Instant::now();
            self.move_frame();
            move_frame_dur += timer.elapsed();

            // println!("Damage frame...");
            timer = Instant::now();
            self.damage_frame();
            damage_frame_dur += timer.elapsed();

            // println!("Destroy frame...");
            timer = Instant::now();
            self.destroy_frame();
            destroy_frame_dur += timer.elapsed();

            if self.p1_moblist.is_empty() && self.p2_moblist.is_empty() {
                phase_done = true;
            }
            
            if self.render {
                print_game(&self.structure_board, &self.p1_moblist, &self.p2_moblist);
                thread::sleep(Duration::from_millis(100));
            }
        }

        if self.p1_data.is_dead() || self.p2_data.is_dead() || self.turn >= 99 {
            self.done = true;
        }

        // println!("support_frame_dur: {:?}", support_frame_dur);
        // println!("move_frame_dur: {:?}", move_frame_dur);
        // println!("damage_frame_dur: {:?}", damage_frame_dur);
        // println!("destroy_frame_dur: {:?}", destroy_frame_dur);

    }

    fn new_phase(&mut self) {
        for structure_coord in self.p1_struc_coords.iter() {
            if self.structure_board.get(*structure_coord).unwrap().as_ref().unwrap().get_type() == StructureType::Support {
                self.structure_board.get_mut(*structure_coord).unwrap().as_mut().unwrap().clear_buffed_mobs();
            }
        }
        for structure_coord in self.p2_struc_coords.iter() {
            if self.structure_board.get(*structure_coord).unwrap().as_ref().unwrap().get_type() == StructureType::Support {
                self.structure_board.get_mut(*structure_coord).unwrap().as_mut().unwrap().clear_buffed_mobs();
            }
        }
    }
    
    fn support_frame(&mut self) {
        for structure_coords in self.p1_struc_coords.iter() {
            if let Some(shielding) = self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_shielding() {
                for (id, mob) in self.p1_moblist.iter_mut() {
                    if Coords::dist(mob.get_coords(), *structure_coords) <= self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_range().unwrap() {
                        if !self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_buffed_mobs().contains(id) {
                            mob.add_shielding(shielding);
                            self.structure_board.get_mut(*structure_coords).unwrap().as_mut().unwrap().add_buffed_mob(*id);
                        }
                    }
                }
            }
        }

        for structure_coords in self.p2_struc_coords.iter() {
            if let Some(shielding) = self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_shielding() {
                for (id, mob) in self.p2_moblist.iter_mut() {
                    if Coords::dist(mob.get_coords(), *structure_coords) <= self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_range().unwrap() {
                        if !self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_buffed_mobs().contains(id) {
                            mob.add_shielding(shielding);
                            self.structure_board.get_mut(*structure_coords).unwrap().as_mut().unwrap().add_buffed_mob(*id);
                        }
                    }
                }
            }
        }
    }

    fn move_frame(&mut self) {
        
        let mut to_remove = Vec::new();
        for (guid, mob) in self.p1_moblist.iter_mut() {

            if self.new_topology {
                let mut pathfinder = BFS::new(&self.structure_board);
                let path = pathfinder.find(mob.get_coords(), mob.get_target_edge(), mob.get_last_direction());
                mob.set_path(path);
            }

            if let Err(FinishedPath) = mob.move_frame() {
                if MAP_BOUNDS.is_at_end(MapSide::Top, mob.get_coords()) {
                    self.p2_data.take_damage();
                    to_remove.push(*guid);
                    self.prev_turn_dmg[0] += 1;
                } else {
                    mob.self_destruct(&mut self.structure_board);
                }
            }
        }
        for guid in to_remove {
            self.p1_moblist.remove(&guid);
        }


        let mut to_remove = Vec::new();
        for (guid, mob) in self.p2_moblist.iter_mut() {

            if self.new_topology {
                let mut pathfinder = BFS::new(&self.structure_board);
                let path = pathfinder.find(mob.get_coords(), mob.get_target_edge(), mob.get_last_direction());
                mob.set_path(path);
            }

            if let Err(FinishedPath) = mob.move_frame() {
                if MAP_BOUNDS.is_at_end(MapSide::Top, mob.get_coords()) {
                    self.p1_data.take_damage();
                    to_remove.push(*guid);
                    self.prev_turn_dmg[1] += 1;
                } else {
                    mob.self_destruct(&mut self.structure_board);

                }
            }
        }
        for guid in to_remove {
            self.p2_moblist.remove(&guid);
        }
    }
    
    fn damage_frame(&mut self) {
        for unit in self.p1_moblist.get_mobs() {

            if let Some(target) = match unit.get_type() {
                MobileType::Interceptor => offensive_targetting(unit, &self.p2_moblist, &Vec::new(), &self.structure_board, 1),
                _ => offensive_targetting(unit, &self.p2_moblist, &self.p2_struc_coords, &self.structure_board, 1),
             } {
                match target {
                    Target::Mob(guid) => 
                        self.p2_moblist.get_mut(&guid).take_damage(unit.get_damage()),
                    Target::Structure(idx) => 
                        self.structure_board.get_mut(self.p2_struc_coords[idx]).unwrap().as_mut().unwrap().take_damage(unit.get_damage()),
                };
            }
        }
        for structure_coords in &self.p1_struc_coords {
            if let Some(damage) = self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_damage() {
                let range = self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_range().unwrap();
                if let Some(target) = defensive_targetting(*structure_coords, range, &self.p2_moblist) {
                    match target {
                        Target::Mob(guid) => 
                            self.p2_moblist.get_mut(&guid).take_damage(damage),
                        Target::Structure(_) => panic!("Structure has targetted another structure"),
                    };
                }
            }
        }

        for unit in self.p2_moblist.get_mobs() {
            if let Some(target) = match unit.get_type() {
                MobileType::Interceptor => offensive_targetting(unit, &self.p1_moblist, &Vec::new(), &self.structure_board, 2),
                _ => offensive_targetting(unit, &self.p1_moblist, &self.p1_struc_coords, &self.structure_board, 2) 
            } {
                match target {
                    Target::Mob(guid) => 
                        self.p1_moblist.get_mut(&guid).take_damage(unit.get_damage()),
                    Target::Structure(idx) => 
                        self.structure_board.get_mut(self.p1_struc_coords[idx]).unwrap().as_mut().unwrap().take_damage(unit.get_damage()),
                };
            }
        }
        for structure_coords in &self.p2_struc_coords {
            if let Some(damage) = self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_damage() {
                let range = self.structure_board.get(*structure_coords).unwrap().as_ref().unwrap().get_range().unwrap();
                if let Some(target) = defensive_targetting(*structure_coords, range, &self.p1_moblist) {
                    match target {
                        Target::Mob(guid) => self.p1_moblist.get_mut(&guid).take_damage(damage),
                        Target::Structure(_) => panic!("Structure has targetted another structure"),
                    };
                }
            }
        }

    }

    fn is_empty(&self, coords:Coords) -> bool {
        self.structure_board.get(coords).is_none()
    }


    fn destroy_frame(&mut self) {
        // remove KIA mobs and structures

        let mut dead_guids = Vec::new();
        for guid in self.p1_moblist.get_guids() {
            if self.p1_moblist.get(guid).is_dead() {
                dead_guids.push(*guid);
            }
        }
        for guid in dead_guids {
            self.p1_moblist.remove(&guid);
        }

        let mut dead_guids = Vec::new();
        for guid in self.p2_moblist.get_guids() {
            if self.p2_moblist.get(guid).is_dead() {
                dead_guids.push(*guid);
            }
        }
        for guid in dead_guids {
            self.p2_moblist.remove(&guid);
        }

        self.new_topology = false;

        let mut i = 0;
        while i < self.p1_struc_coords.len() {
            if self.structure_board.get(self.p1_struc_coords[i]).unwrap().as_ref().unwrap().is_dead() {
                self.structure_board.set(self.p1_struc_coords[i], None);
                self.p1_struc_coords.remove(i);
                self.new_topology = true;
            } else {
                i += 1;
            }
        } 
        i = 0;
        while i < self.p2_struc_coords.len() {
            if self.structure_board.get(self.p2_struc_coords[i]).unwrap().as_ref().unwrap().is_dead() {
                self.structure_board.set(self.p2_struc_coords[i], None);
                self.p2_struc_coords.remove(i);
                self.new_topology = true;
            } else {
                i += 1;
            }
        } 
    }


}

fn deploy(ops: &Vec<(usize,(i32,i32))>, moblist: &mut MobList, struc_coords: &mut Vec<Coords>, board: &mut Grid<Option<Structure>>, player_data: &mut PlayerData) {
    for op in ops {
        let coords = xy(op.1.0, op.1.1);
        if board.get(coords).unwrap().is_none() {
            match op.0 {
                0 => {
                    if player_data.spend_structure_pts(1.0).is_ok() {
                        let struc = Structure::new(StructureType::Wall, coords);
                        board.set(coords, Some(struc));
                        struc_coords.push(coords);
                    }
                },
                1 => {
                    if player_data.spend_structure_pts(2.0).is_ok() {
                        let struc = Structure::new(StructureType::Turret, coords);
                        board.set(coords, Some(struc));
                        struc_coords.push(coords);
                    }
                },
                2 => {
                    if player_data.spend_structure_pts(4.0).is_ok() {
                        let struc = Structure::new(StructureType::Support, coords);
                        board.set(coords, Some(struc));
                        struc_coords.push(coords);
                    }
                },
                3 => {
                },
                4 => {
                    if player_data.spend_mobile_pts(1.0).is_ok() {
                        let mob = MobileUnit::new(MobileType::Scout, coords);
                        moblist.insert(mob);
                    }
                },
                5 => {
                    if player_data.spend_mobile_pts(3.0).is_ok() {
                        let mob = MobileUnit::new(MobileType::Demolisher, coords);
                        moblist.insert(mob);
                    }
                },
                6 => {
                    if player_data.spend_mobile_pts(1.0).is_ok() {
                        let mob = MobileUnit::new(MobileType::Interceptor, coords);
                        moblist.insert(mob);
                    }
                },
                _ => {
                    panic!("invalid op_code encountered");
                }
            }
        } else {
            match op.0 {
                3 => {
                    let struc = board.get_mut(coords).unwrap().as_mut().unwrap();
                    if !struc.is_upgraded() {
                        if player_data.spend_structure_pts(struc.get_upgrade_cost()).is_ok() {
                            struc.upgrade();
                        }
                    }
                },
                _ => {

                }

            }
        }
    }
}