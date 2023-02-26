// use c1_term_sim::Game;

use std::time::{Duration, SystemTime};

fn main() {
    // game.simulate_mobs();

//     let p1_scout_points = [[25, 11]];
    let start_time = SystemTime::now();
    let iters = 1000;
    for _ in 0..iters {
        // println!("-----------------------------------------------------");
        // let mut game = Game::new(false);
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
        // game.step(action);
    }
    println!("{} iterations took {:?} for an average of {}us per iteration",iters, start_time.elapsed().unwrap(), start_time.elapsed().unwrap().as_micros()/iters);

}

// pub fn simulate_mobs(&mut self) {
//     let p1_scout_points = [[25, 11]];
//     let p2_wall_points = [[15, 26], [15, 25], [14, 24], [13, 23], [12, 22], [11, 21], [10, 20], [9, 19], [8, 18], [7, 17], [6, 16], [7, 16], [5, 15], [6, 15], [0, 14], [1, 14], [2, 14], [3, 14], [4, 14], [5, 14]];
//     for point in p1_scout_points {
//         let coords = xy(point[0], point[1]);
//         self.p1_moblist.insert(MobileUnit::new(MobileType::Scout, coords));
//     }
//     for point in p2_wall_points {
//         let coords = xy(point[0], point[1]);
//         let structure = Structure::new(StructureType::Wall);
//         self.structure_board.set(coords, Some(structure));
//         self.p2_struc_coords.push(coords);
//     }
//     self.action();
// }