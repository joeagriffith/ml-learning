use crate::board::{Grid, Coords};
use crate::units::{MobileUnit, Structure};
use crate::utils::MobList;

#[derive(Debug)]
pub enum Target {
    Mob(usize),
    Structure(usize),
}

pub fn offensive_targetting(unit:&MobileUnit, mobs:&MobList, structures:&Vec<Coords>, structure_board:&Grid<Option<Structure>>, player:usize) -> Option<Target> {
    let mut target = check_mobs(unit.get_coords(), unit.get_range(), mobs, player);
    if target.is_none() {
        target = check_structures(unit, structures, structure_board, player);
    }
    target
}

fn check_mobs(unit_coords:Coords, unit_range:f32, mobs:&MobList, player:usize) -> Option<Target> {
    let mut targetted_guids = Vec::new();

    // Collect vector of closest mobs
    let mut shortest_dist = unit_range;
    for guid in mobs.get_guids() {
        let dist = Coords::dist(unit_coords, mobs.get(guid).get_coords());
        if dist <= shortest_dist {
            if dist < shortest_dist {
                shortest_dist = dist;
                targetted_guids.clear();
            }
            targetted_guids.push(guid.clone());
        }
    }

    // Prioritise targets with lower health
    // Remove targets with greater than vector's minimum health
    if targetted_guids.len() > 1 {
        let mut lowest_health = mobs.get(targetted_guids.get(0).unwrap()).get_health();
        let mut i = 1;
        while i < targetted_guids.len() {
            let health = mobs.get(targetted_guids.get(i).unwrap()).get_health();
            if health < lowest_health {
                lowest_health = health;
                i = 0;
            } else {
                if health > lowest_health {
                    targetted_guids.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    // Prioritise targets which are furthest into the unit's size of the arena
    // Remove targets with greater than vector's minimum delta_y
    if targetted_guids.len() > 1 {
        let mut target_y:i8 = 0;
        if player == 2 {
            target_y = 27;
        }
        let mut min_delta_y = (mobs.get(targetted_guids.get(0).unwrap()).get_coords().y as i8 - target_y).abs();
        let mut i = 1;
        while i < targetted_guids.len() {
            let delta_y = (mobs.get(targetted_guids.get(i).unwrap()).get_coords().y as i8 - target_y).abs();
            if delta_y < min_delta_y {
                min_delta_y = delta_y;
                i = 0;
            } else {
                if delta_y > min_delta_y {
                    targetted_guids.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    // Prioritise targets which are closest to an edge, or max dist from center
    // Remove targets with less than vector's maximum distance from board center
    if targetted_guids.len() > 1 {
        let mut max_dist_from_center = mobs.get(targetted_guids.get(0).unwrap()).get_coords().dist_from_center();
        let mut i = 1;
        while i < targetted_guids.len() {
            let dist_from_center = mobs.get(targetted_guids.get(i).unwrap()).get_coords().dist_from_center();
            if dist_from_center > max_dist_from_center {
                max_dist_from_center = dist_from_center;
                i = 0;
            } else {
                if dist_from_center < max_dist_from_center {
                    targetted_guids.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    // In case that logic does not identify a unique unit, the most recent unit created is chosen.
    // same result if unique unit is identified.
    let len = targetted_guids.len();
    if len >= 1 {
        return Some(Target::Mob(targetted_guids[len-1]));
    }

    None

}

fn check_structures(unit:&MobileUnit, structures:&Vec<Coords>, structure_board:&Grid<Option<Structure>>, player:usize) -> Option<Target> {
    let mut targetted_idxs = Vec::new();

    let mut shortest_dist = unit.get_range();
    for i in 0..structures.len() {
        let dist = Coords::dist(unit.get_coords(), structures[i]);
        if dist <= shortest_dist {
            if dist < shortest_dist {
                shortest_dist = dist;
                targetted_idxs.clear();
            }
            targetted_idxs.push(i);
        }
    }

    // Prioritise targets with lower health
    // Remove targets with greater than vector's minimum health
    if targetted_idxs.len() > 1 {
        let mut lowest_health = structure_board.get(structures[targetted_idxs[0]]).unwrap().as_ref().unwrap().get_health();
        let mut i = 1;
        while i < targetted_idxs.len() {
            let health = structure_board.get(structures[targetted_idxs[i]]).unwrap().as_ref().unwrap().get_health();
            if health < lowest_health {
                lowest_health = health;
                i = 0;
            } else {
                if health > lowest_health {
                    targetted_idxs.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    // Prioritise targets which are furthest into the unit's size of the arena
    // Remove targets with greater than vector's minimum delta_y
    if targetted_idxs.len() > 1 {
        let mut target_y:i8 = 0;
        if player == 2 {
            target_y = 27;
        }
        let mut min_delta_y = (structures[targetted_idxs[0]].y as i8 - target_y).abs();
        let mut i = 1;
        while i < targetted_idxs.len() {
            let delta_y = (structures[targetted_idxs[i]].y as i8 - target_y).abs();
            if delta_y < min_delta_y {
                min_delta_y = delta_y;
                i = 0;
            } else {
                if delta_y > min_delta_y {
                    targetted_idxs.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    // Prioritise targets which are closest to an edge, or max dist from center
    // Remove targets with less than vector's maximum distance from board center
    if targetted_idxs.len() > 1 {
        let mut max_dist_from_center = structures.get(targetted_idxs[0]).unwrap().dist_from_center();
        let mut i = 1;
        while i < targetted_idxs.len() {
            let dist_from_center = structures.get(targetted_idxs[i]).unwrap().dist_from_center();
            if dist_from_center > max_dist_from_center {
                max_dist_from_center = dist_from_center;
                i = 0;
            } else {
                if dist_from_center < max_dist_from_center {
                    targetted_idxs.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    // In case that logic does not identify a unique unit, the most recent unit created is chosen.
    // same result if unique unit is identified.
    let len = targetted_idxs.len();
    if len >= 1 {
        return Some(Target::Structure(targetted_idxs[len-1]));
    }
    
    None
}

pub fn defensive_targetting(structure_coords:Coords, structure_range:f32, mobs:&MobList) -> Option<Target> {
    let mut player = 1;
    if structure_coords.y > 13 {
        player = 2;
    }
    check_mobs(structure_coords, structure_range, mobs, player)
}
