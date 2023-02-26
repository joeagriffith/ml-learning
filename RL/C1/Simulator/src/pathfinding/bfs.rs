use std::collections::LinkedList;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter, Error, Display};
use std::thread::current;
use std::{time::{Instant, SystemTime, Duration}, alloc::System};
use std::thread;
use crate::board::{Grid, Coords, MAP_BOUNDS, xy, MapEdge, BOARD_SIZE};
use crate::units::Structure;

#[derive(Clone, Copy, Debug)]
struct Node {
    visited_idealness: bool,
    visited_validate: bool,
    blocked: bool,
    pathlength: u32,
}
impl Node {
    pub fn new() -> Self {
        Self {
            visited_idealness: false,
            visited_validate: false,
            blocked: false,
            pathlength: u32::MAX,
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        if self.blocked {
            write!(f, "  ")?;
        }else if self.visited_validate {
            write!(f, " {}", self.pathlength)?;
        } else {
            write!(f, " _")?;
        }
        Ok(())
    }
}

const HORIZONTAL:usize = 1;
const VERTICAL: usize = 2;

// Need to compare whether reference to board or a clone of the board is faster
pub struct BFS {
    // board: &'a Grid<Option<Structure>>,
    game_map: Grid<Node>,
}
impl BFS {
    // remove reference to board in struct, take a reference to p1_structs and p2_structs in new instead
    pub fn new(board:&Grid<Option<Structure>>) -> Self {
        let mut res = Self {
            // board,
            game_map: Grid::from_generator(|_| Node::new()),
        };

        for x in 0..28 {
            for y in 0..28 {
                let coords = xy(x,y);
                if board.get(coords).unwrap().is_some() {
                    res.game_map.get_mut(coords).unwrap().blocked = true;
                }
            }
        }

        res
    }

    pub fn find(&mut self, start_point:Coords, target_edge:MapEdge, last_direction:usize) -> Vec<(Coords, usize)> {

        let ideal_tile = self.idealness_search(start_point, target_edge);
        self.validate(ideal_tile, target_edge);
        
        return self.get_path(start_point, target_edge, last_direction);
    }

    

    fn idealness_search(&mut self, start_point:Coords, target_edge:MapEdge) -> Coords {


        let mut queue = LinkedList::new();
        queue.push_back(start_point.clone());
        let mut best_idealness = self.get_idealness(start_point, target_edge);
        self.game_map.get_mut(start_point).unwrap().visited_idealness = true;
        let mut most_ideal = start_point;

        while !queue.is_empty() {
            let search_location = queue.pop_front().unwrap();
            for neighbour in search_location.neighbors() {

                if !MAP_BOUNDS.is_in_arena(neighbour) || self.game_map.get(neighbour).unwrap().blocked || self.game_map.get(neighbour).unwrap().visited_idealness{
                    continue;
                }

                if MAP_BOUNDS.is_on_edge(target_edge, neighbour) {
                    return neighbour;
                }

                // let (x,y) = (neighbour.x, neighbour.y);
                let current_idealness = self.get_idealness(neighbour, target_edge);

                if current_idealness > best_idealness {
                    best_idealness = current_idealness;
                    most_ideal = neighbour;
                }

                self.game_map.get_mut(neighbour).unwrap().visited_idealness = true;
                queue.push_back(neighbour);
            }
        }
        most_ideal
    }


    fn get_idealness(&self, location:Coords, target_edge:MapEdge) -> u32 {
        if MAP_BOUNDS.is_on_edge(target_edge, location) {
            return u32::MAX;
        }

        let coords = [location.x as u32, location.y as u32];
        let board_size_u32 = BOARD_SIZE as u32;
        
        let a = match target_edge {
            MapEdge::TopLeft | MapEdge::TopRight =>
                board_size_u32 * coords[1],
            MapEdge::BottomLeft | MapEdge::BottomRight =>
                board_size_u32 * (board_size_u32 - 1 - coords[1]),
        };

        let b = match target_edge {
            MapEdge::TopRight | MapEdge::BottomRight =>
                coords[0],
            MapEdge::TopLeft | MapEdge::BottomLeft =>
                board_size_u32 - 1 - coords[0]
        };

        a + b
    }

    fn validate(&mut self, ideal_tile:Coords, target_edge:MapEdge) {
        let mut queue = VecDeque::new();
        if MAP_BOUNDS.is_on_edge(target_edge, ideal_tile) {
            for location in *MAP_BOUNDS.coords_on_edge(target_edge) {
                if !self.game_map.get(location).unwrap().blocked {
                    queue.push_back(location.clone());
                    let end_node = self.game_map.get_mut(location).unwrap();
                    end_node.pathlength = 0;
                    end_node.visited_validate = true;
                }
            }
        } else {
            queue.push_back(ideal_tile.clone());
            let ideal_node = self.game_map.get_mut(ideal_tile).unwrap();
            ideal_node.pathlength = 0;
            ideal_node.visited_validate = true;
        }

        while !queue.is_empty() {
            let current_location = queue.pop_front().unwrap();
            let current_pathlength = self.game_map.get(current_location).unwrap().pathlength;
            for neighbour in current_location.neighbors() {

                if !MAP_BOUNDS.is_in_arena(neighbour) || self.game_map.get(neighbour).unwrap().blocked || self.game_map.get(neighbour).unwrap().visited_validate {
                    continue;
                }
                let neighbour_node = self.game_map.get_mut(neighbour).unwrap();
                neighbour_node.pathlength = current_pathlength + 1;
                neighbour_node.visited_validate = true;
                queue.push_back(neighbour);
            }
        }
    }

    fn get_path(&mut self, start_point:Coords, target_edge:MapEdge, last_direction:usize) -> Vec<(Coords, usize)> {
        let mut path = vec![];
        let mut current_coord = start_point.clone();
        let mut move_direction:usize = last_direction;
        
        let mut dbg_i = 0;
        while self.game_map.get(current_coord).unwrap().pathlength != 0 {

            let next_coord = self.choose_next_move(current_coord, move_direction, target_edge);
            
            if current_coord.x == next_coord.x {
                move_direction = VERTICAL;
            } else {
                move_direction = HORIZONTAL;
            }

            path.push((next_coord.clone(), move_direction));
            current_coord = next_coord; 

            dbg_i += 1;

            if dbg_i > 1000 {
                println!("start_point: {:?}", start_point);
                println!("{:?}", self.game_map);
                thread::sleep(Duration::from_secs(1000));
            }
        }
        
        path
    }

    fn choose_next_move(&self, current_coord:Coords, prev_move_direction:usize, target_edge:MapEdge) -> Coords {
        let neighbours = current_coord.neighbors();
        let mut ideal_neighbour = current_coord;
        let mut best_pathlength = self.game_map.get(current_coord).unwrap().pathlength;
        for neighbour in neighbours {
            if !MAP_BOUNDS.is_in_arena(neighbour) || self.game_map.get(neighbour).unwrap().blocked {
                continue;
            }

            let mut new_best = false;
            let current_pathlength = self.game_map.get(neighbour).unwrap().pathlength;

            if current_pathlength > best_pathlength {
                continue;
            } else if current_pathlength < best_pathlength {
                new_best = true;
            }

            if !new_best && !self.better_direction(current_coord, neighbour, ideal_neighbour, prev_move_direction, target_edge) {
                continue;
            }

            ideal_neighbour = neighbour;
            best_pathlength = current_pathlength;
        }

        ideal_neighbour
    }

    fn better_direction(&self, prev_tile:Coords, new_tile:Coords, prev_best:Coords, prev_move_direction:usize, target_edge:MapEdge) -> bool {

        if prev_move_direction == HORIZONTAL && new_tile.x != prev_best.x {
            if prev_tile.y == new_tile.y {
                return false
            }
            return true
        }
        if prev_move_direction == VERTICAL && new_tile.y != prev_best.y {
            if prev_tile.x == new_tile.x {
                return false
            }
            return true
        }
        if prev_move_direction == 0 {
            if prev_tile.y == new_tile.y {
                return false
            }
            return true
        }

        let direction = self.get_direction_from_endpoints(target_edge);
        if new_tile.y == prev_best.y {
            if direction[0] == 1 && new_tile.x > prev_best.x {
                return true
            }
            if direction[0] == -1 && new_tile.x < prev_best.x {
                return true
            }
            return false
        }
        if new_tile.x == prev_best.x {
            if direction[1] == 1 && new_tile.y > prev_best.y {
                return true
            }
            if direction[1] == -1 && new_tile.y < prev_best.y {
                return true
            }
            return false
        }
        return true
    }

    fn get_direction_from_endpoints(&self, target_edge:MapEdge) -> [i8;2] {
        let direction:[i8;2];

        match target_edge {
            MapEdge::BottomLeft => direction = [-1,-1],
            MapEdge::BottomRight => direction = [1, -1],
            MapEdge::TopRight => direction = [1,1],
            MapEdge::TopLeft => direction = [-1,1],
        }

        direction
    }

}

