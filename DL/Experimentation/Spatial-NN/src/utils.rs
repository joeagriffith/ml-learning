use ndarray::{ArrayView, ArrayView1};
use ndarray::{Array1, arr1};

pub fn get_connection_coords_2d(pos:(usize,usize), size:usize) -> [Option<(usize, usize, usize)>;4] {
    let (row, col) = pos;
    let mut res = [None;4];

    if let Some(sub_row) = row.checked_sub(1) {
        let up = (1, col, sub_row);
        if is_valid_2d(&up, size) {
            res[0] = Some(up);
        }
    }

    let down = (1, col, row);
    if is_valid_2d(&down, size) {
        res[1] = Some(down);
    }

    if let Some(sub_col) = col.checked_sub(1) {
        let left = (0, row, sub_col);

        if is_valid_2d(&left, size) {
            res[2] = Some(left);
        }
    }

    let right = (0, row, col);
    if is_valid_2d(&right, size) {
        res[3] = Some(right);
    }

    res
} 

fn is_valid_2d(pos:&(usize,usize,usize), size:usize) -> bool {
    pos.1 < size && pos.2 < size-1
}


pub fn get_connection_coords_3d(pos:(usize,usize,usize), size:usize) -> [Option<(usize, usize, usize, usize)>;6] {
    let (lev, row, col) = pos;
    let mut res = [None;6];

    if let Some(sub_lev) = lev.checked_sub(1) {
        let down_lev = (0, row, col, sub_lev);
        if is_valid_3d(&down_lev, size) {
            res[0] = Some(down_lev);
        }
    }

    let up_lev = (0, row, col, lev);
    if is_valid_3d(&up_lev, size) {   
        res[1] = Some(up_lev);    
    }

    if let Some(sub_row) = row.checked_sub(1) {
        let up = (1, lev, col, sub_row);
        if is_valid_3d(&up, size) {
            res[2] = Some(up);
        }
    }

    let down = (1, lev, col, row);
    if is_valid_3d(&down, size) {
        res[3] = Some(down);
    }

    if let Some(sub_col) = col.checked_sub(1) {
        let left = (2, lev, row, sub_col);
        if is_valid_3d(&left, size) {
            res[4] = Some(left);
        }
    }

    let right = (2, lev, row, col);
    if is_valid_3d(&right, size) {
        res[5] = Some(right);
    }

    //transposed next 2

    res
} 

fn is_valid_3d(pos:&(usize,usize,usize,usize), size:usize) -> bool {
    pos.1 < size && pos.2 < size && pos.3 < size-1
}

pub fn sigmoid(input:f32) -> f32 {
    1.0 / (1.0 + (-input).exp())
}

pub fn get_neighbours_3d(pos:(usize,usize,usize), size:usize) -> Vec<(usize, usize, usize)> {
    let mut res = Vec::new();

    if let Some(sub_lev) = pos.0.checked_sub(1) {
        res.push((sub_lev, pos.1, pos.2));
    }
    if pos.0 +1 < size {
        res.push((pos.0+1, pos.1, pos.2));
    }

    if let Some(sub_row) = pos.1.checked_sub(1) {
        res.push((pos.0, sub_row, pos.2));
    }
    if pos.1 +1 < size {
        res.push((pos.0, pos.1+1, pos.2));
    }

    if let Some(sub_col) = pos.2.checked_sub(1) {
        res.push((pos.0, pos.1, sub_col));
    }
    if pos.2 +1 < size {
        res.push((pos.0, pos.1, pos.2+1));
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;
 
    #[test]
    fn get_connection_coords_3d_111() {
        let pos = (1,1,1);
        let out = get_connection_coords_3d(pos, 3);
        let target:[Option<(usize, usize,usize,usize)>;6] = [
            Some((0,1,1,0)), Some((0,1,1,1)), 
            Some((1,1,1,0)), Some((1,1,1,1)), 
            Some((2,1,1,0)), Some((2,1,1,1)),
        ];
        for i in 0..6 {
            assert_eq!(target[i].unwrap(), out[i].unwrap());
        }
    }

    #[test]
    fn get_connection_coords_3d_000() {
        let pos = (0,0,0);
        let out = get_connection_coords_3d(pos, 3);
        let target:[Option<(usize, usize, usize, usize)>;6] = [
            None, Some((0,1,1,1)),
            None, Some((1,1,1,1)),
            None, Some((2,1,1,1)),
        ];
    }

    #[test]
    fn get_connection_coords_3d_neighbours_share() {
        let pos1 = (1,1,1);
        let pos2 = (1,1,2);
        let pos1_coords = get_connection_coords_3d(pos1, 3);
        let pos2_coords = get_connection_coords_3d(pos2, 3);
        assert_eq!(pos1_coords[5], pos2_coords[4]);
    }

    #[test]
    fn get_connection_coords_3d_max() {
        let pos = (2,2,2);
        let out = get_connection_coords_3d(pos, 3);
        let target:[Option<(usize, usize, usize, usize)>;6] = [
            Some((0,2,2,1,)), None,
            Some((1,2,2,1)), None,
            Some((2,2,2,1)), None,
        ];
        for i in 0..6 {
            assert_eq!(target[i], out[i]);
        }
    }
}