use crate::{{board::*}, units::*};
use super::MobList;


pub fn print_game(board: &Grid<Option<Structure>>, p1_mobs: &MobList, p2_mobs: &MobList) {
    let mut grid = Grid::from_generator(|c| {
        if MAP_BOUNDS.is_in_arena(c) {
            if let Some(struc) = board.get(c).unwrap() {
                struc.get_str()
            } else {
                " ."
            }
        } else {
            "  "
        }
    });

    for unit in p1_mobs.get_mobs() {
        let coords = unit.get_coords();
        let char = unit.get_str();
        grid.set(coords, char);
    }
    for unit in p2_mobs.get_mobs() {
        let coords = unit.get_coords();
        let char = unit.get_str();
        grid.set(coords, char);
    }
    println!("{:#?}", grid);
}