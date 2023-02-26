// API

mod game_logic;
mod units;
mod pathfinding;
mod utils;
mod board;

pub use game_logic::*;
pub use board::*;

use std::time::{SystemTime, Duration};
use rand::prelude::*;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn dmg_test() -> PyResult<usize> {
    Ok(damage_test())
}

#[pyfunction]
fn rand_test(render:bool) -> PyResult<()> {
    Ok(random_test(render))
}

#[pymodule]
fn c1_term_sim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(dmg_test))?;
    m.add_wrapped(wrap_pyfunction!(rand_test))?;
    Ok(())
}

pub fn damage_test() -> usize {
    let timer = SystemTime::now();
    let mut game = Game::new(false);
    let p2_wall_points = [[15, 26], [15, 25], [14, 24], [13, 23], [12, 22], [11, 21], [10, 20], [9, 19], [8, 18], [7, 17], [6, 16], [7, 16], [5, 15], [6, 15], [0, 14], [1, 14], [2, 14], [3, 14], [4, 14], [5, 14]];
    let p2_turr_points = [[12, 19]];//, [10, 17], [8, 15]];
    let p1_scout_points = [[6, 7],[6, 7],[6, 7],[6, 7],[6, 7]];
    let p1_support_points = [[10, 10], [9, 9], [8, 8]];

    let mut action = [Vec::new(), Vec::new()];
    for point in p2_wall_points {
        action[1].push((0,(point[0], point[1])));
    }
    for point in p2_turr_points {
        action[1].push((1, (point[0], point[1])));
    }
    for point in p1_scout_points {
        action[0].push((4, (point[0], point[1])));
    }
    for point in p1_support_points {
        action[0].push((2, (point[0], point[1])));
    }
    let (observation, reward, done, info) = game.step(action);
    println!("took {} us", timer.elapsed().unwrap().as_micros());
    assert_eq!(5, observation.prev_turn_dmg[0]);
    observation.prev_turn_dmg[0]
}

pub fn random_test(render:bool) {
    let mut iters = 5;
    let mut duration = Duration::from_secs(0);
    let mut rng = rand::thread_rng();

    for _ in 0..iters {

        let mut game = Game::new(render);
        let mut action = [Vec::new(), Vec::new()];
        game.dbg_moneyhack();
        let mut mob_coords = [Vec::new(), Vec::new()];
        let mut struc_coords = [Vec::new(), Vec::new()];
        for player in 0..=1 {
            for _ in 0..30 {
                if let Ok(coords) = game.place_random_firewall(player) {
                    struc_coords[player].push(coords);
                }
            }

            for _ in 0..10 {
                let rand_i = rng.gen_range(0..14);
                let mut iter = 0;
                while iter < 100 {

                    let coords:Coords;
                    if rng.gen_bool(0.5) {
                        if player == 1 {
                            coords = MAP_BOUNDS.coords_on_edge(MapEdge::TopLeft)[rand_i];
                        } else {
                            coords = MAP_BOUNDS.coords_on_edge(MapEdge::BottomLeft)[rand_i];
                        }
                    } else {
                        if player == 1 {
                            coords = MAP_BOUNDS.coords_on_edge(MapEdge::TopRight)[rand_i];
                        } else {
                            coords = MAP_BOUNDS.coords_on_edge(MapEdge::BottomRight)[rand_i];
                        }
                    }
                    if !struc_coords[player].contains(&coords) {
                        mob_coords[player].push(coords);
                        break;
                    }
                    iter += 1;
                }
            }

            for &coords in mob_coords[player].iter() {
                action[player].push((rng.gen_range(4..=6) as usize, (coords.x, coords.y)));
            }


        }

        let timer = SystemTime::now();
        let (observation, reward, done, info) = game.step(action);
        duration += timer.elapsed().unwrap();


    }
    println!("{} iterations took {} ms, at an average of {}us per iteration", iters, duration.as_millis(), duration.as_micros() / iters);
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

