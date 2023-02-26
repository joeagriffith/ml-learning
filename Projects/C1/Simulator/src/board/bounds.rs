
use super::coords::*;
use enum_iterator::IntoEnumIterator;
use lazy_static::*;

lazy_static! {
    /// Global cache of the shape of the game map.
    pub static ref MAP_BOUNDS: MapBounds = MapBounds::new();
}

/// Side-length of the game board.
pub const BOARD_SIZE: usize = 28;

/// Cached representation of the shape of the game map.
pub struct MapBounds {
    pub is_on_edge: [[[bool; BOARD_SIZE]; BOARD_SIZE]; 4],
    pub edge_lists: [[Coords; BOARD_SIZE / 2]; 4],
    pub arena: [[bool; BOARD_SIZE]; BOARD_SIZE],
}

impl MapBounds {
    /// Compute a cache of the map bounds.
    pub fn new() -> Self {
        fn populate(
            is_on_edge: &mut [[[bool; BOARD_SIZE]; BOARD_SIZE]; 4],
            edge_lists: &mut [[Coords; BOARD_SIZE / 2]; 4],
            edge: MapEdge,
            calc_coords: impl Fn(usize) -> [usize; 2]
        ) {
            for i in 0..BOARD_SIZE / 2 {
                let c = calc_coords(i);
                is_on_edge[edge as usize][c[0]][c[1]] = true;
                let c = Coords::from(c);
                edge_lists[edge as usize][i] = c;
            }
        }

        let mut is_on_edge = [[[false; BOARD_SIZE]; BOARD_SIZE]; 4];
        let mut edge_lists = [[ORIGIN; BOARD_SIZE / 2]; 4];

        populate(&mut is_on_edge, &mut edge_lists, MapEdge::TopRight,
                       |i| [BOARD_SIZE / 2 + i, BOARD_SIZE - 1 - i]
        );
        populate(&mut is_on_edge, &mut edge_lists, MapEdge::TopLeft,
                       |i| [BOARD_SIZE / 2 - 1 - i, BOARD_SIZE - 1 - i]
        );
        populate(&mut is_on_edge, &mut edge_lists, MapEdge::BottomLeft,
                       |i| [BOARD_SIZE / 2 - 1 - i, i]
        );
        populate(&mut is_on_edge, &mut edge_lists, MapEdge::BottomRight,
                       |i| [BOARD_SIZE / 2 + i, i]
        );

        let mut arena = [[false; BOARD_SIZE]; BOARD_SIZE];

        for i in 0..4 {
            for j in 0..BOARD_SIZE {
                for k in 0..BOARD_SIZE {
                    arena[j][k] |= is_on_edge[i][j][k];
                }
            }
        }

        for y in 0..BOARD_SIZE {
            let mut toggled = false;
            for x in 0..BOARD_SIZE {
                if arena[x][y] {
                    if toggled {
                        break;
                    } else {
                        toggled = true;
                    }
                } else if toggled {
                    arena[x][y] = true;
                }
            }
        }

        MapBounds {
            is_on_edge,
            edge_lists,
            arena
        }
    }

    /// Is the given coord in the arena?
    pub fn is_in_arena(&self, coords: Coords) -> bool {
        coords.x >= 0 &&
            coords.y >= 0 &&
            coords.x < BOARD_SIZE as i32 &&
            coords.y < BOARD_SIZE as i32 &&
            self.arena[coords.x as usize][coords.y as usize]
    }

    /// Is the given coord on the given edge?
    pub fn is_on_edge(&self, edge: MapEdge, coords: Coords) -> bool {
        coords.x >= 0 &&
            coords.y >= 0 &&
            coords.x < BOARD_SIZE as i32 &&
            coords.y < BOARD_SIZE as i32 &&
            self.is_on_edge[edge as usize][coords.x as usize][coords.y as usize]
    }

    /// Reference to all the coords on that edge.
    pub fn coords_on_edge(&self, edge: MapEdge) -> &[Coords; BOARD_SIZE / 2] {
        &self.edge_lists[edge as usize]
    }

    pub fn is_at_end(&self, side: MapSide, coords: Coords) -> bool {
        match side {
            MapSide::Top => self.is_on_edge(MapEdge::TopLeft, coords) || self.is_on_edge(MapEdge::TopRight, coords),
            MapSide::Bottom => self.is_on_edge(MapEdge::BottomLeft, coords) || self.is_on_edge(MapEdge::BottomRight, coords),
        }
    }
}


/// Edge of the map.
#[repr(usize)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, IntoEnumIterator, Debug)]
pub enum MapEdge {
    TopRight = 0,
    TopLeft = 1,
    BottomLeft = 2,
    BottomRight = 3,
}

pub enum MapSide {
    Top,
    Bottom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_in_arena() {
        for x in 0..28 {
            for y in 0..28 {
                if y >= -x + 13 && y>= x - 14 && y <= x + 14 && y <= -x + 41{
                    assert!(MAP_BOUNDS.is_in_arena(xy(x, y)));
                } else {
                    assert!(!MAP_BOUNDS.is_in_arena(xy(x, y)));
                }
            }
        }
    }

    #[test]
    fn is_on_edge() {
        for x in 0..28 {
            for y in 0..28 {
                if y == -x + 13 {
                    assert!(MAP_BOUNDS.is_on_edge(MapEdge::BottomLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::TopLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::BottomRight, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::TopRight, xy(x, y)));
                } else if y == x - 14 {
                    assert!(MAP_BOUNDS.is_on_edge(MapEdge::BottomRight, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::TopRight, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::BottomLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::TopLeft, xy(x, y)));
                } else if y == x + 14 {
                    assert!(MAP_BOUNDS.is_on_edge(MapEdge::TopLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::TopRight, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::BottomLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::BottomRight, xy(x, y)));
                } else if y == -x + 41 {
                    assert!(MAP_BOUNDS.is_on_edge(MapEdge::TopRight, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::TopLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::BottomLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::BottomRight, xy(x, y)));
                } else {
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::TopLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::TopRight, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::BottomLeft, xy(x, y)));
                    assert!(!MAP_BOUNDS.is_on_edge(MapEdge::BottomRight, xy(x, y)));
                }
            }
        }
    }

    #[test]
    fn coords_on_edge() {

        let mut i_bottomleft = 0;
        let mut i_bottomright = 0;
        let mut i_topleft = 0;
        let mut i_topright = 0;

        let mut bottomleft = [xy(0,0); BOARD_SIZE / 2];
        let mut bottomright = [xy(0,0); BOARD_SIZE / 2];
        let mut topleft = [xy(0,0); BOARD_SIZE / 2];
        let mut topright = [xy(0,0); BOARD_SIZE / 2];

        for x in (0..28).rev() {
            for y in (0..28) {
                if y == -x + 13 {
                    bottomleft[i_bottomleft] = xy(x, y);
                    i_bottomleft += 1;
                }
            }
        }
        for x in 0..28 {
            for y in 0.. 28 {
                if y == x - 14 {
                    bottomright[i_bottomright] = xy(x, y);
                    i_bottomright += 1;
                } 
            }
        }
        for x in (0..28).rev() {
            for y in (0..28).rev() {

                if y == x + 14 {
                    topleft[i_topleft] = xy(x, y);
                    i_topleft += 1;
                }
            }
        }
        for x in 0..28 {
            for y in (0..28).rev() {
                if y == -x + 41 {
                    topright[i_topright] = xy(x, y);
                    i_topright += 1;
                }
            }
        } 

        assert_eq!(MAP_BOUNDS.coords_on_edge(MapEdge::BottomLeft), &bottomleft);
        assert_eq!(MAP_BOUNDS.coords_on_edge(MapEdge::BottomRight), &bottomright);
        assert_eq!(MAP_BOUNDS.coords_on_edge(MapEdge::TopLeft), &topleft);
        assert_eq!(MAP_BOUNDS.coords_on_edge(MapEdge::TopRight), &topright);
    }

    #[test]
    fn is_at_end() {
        for x in 0..28 {
            for y in 0..28 {
                let coords = xy(x, y);
                if y == -x + 13 {
                    assert!(MAP_BOUNDS.is_at_end(MapSide::Bottom, coords));
                    assert!(!MAP_BOUNDS.is_at_end(MapSide::Top, coords));
                } else if y == x - 14 {
                    assert!(MAP_BOUNDS.is_at_end(MapSide::Bottom, coords));
                    assert!(!MAP_BOUNDS.is_at_end(MapSide::Top, coords));
                } else if y == x + 14 {
                    assert!(MAP_BOUNDS.is_at_end(MapSide::Top, coords));
                    assert!(!MAP_BOUNDS.is_at_end(MapSide::Bottom, coords));
                } else if y == -x + 41 {
                    assert!(MAP_BOUNDS.is_at_end(MapSide::Top, coords));
                    assert!(!MAP_BOUNDS.is_at_end(MapSide::Bottom, coords));
                }
            }
        }
    }
}