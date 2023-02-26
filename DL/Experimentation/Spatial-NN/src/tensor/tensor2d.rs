use ndarray::{Array2, Array3, Zip};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::prelude::*;

use crate::structures::*;
use crate::utils::{get_connection_coords_2d, sigmoid};

pub struct Tensor2D {
    nodes: Array2<Node>,
    conn_pos_lookup: Array2<[Option<(usize, usize, usize)>;4]>,
    connections: Array3<Connection>,
}

impl Tensor2D {
    pub fn new(size:usize) -> Tensor2D {
        let mut res = Tensor2D {
            nodes: Array2::from_elem((size, size), Node::new()),
            conn_pos_lookup: Array2::from_elem((size, size), [None;4]),
            // conn_pos_lookup: Array2::from_shape_fn((size, size), get_connection_coords),
            connections: Array3::from_shape_simple_fn((2, size, size-1), Connection::new),
        };

        res.update_conn_pos_lookup();
        res
    }
    pub fn new_ones(size:usize) -> Tensor2D {
        let mut res = Tensor2D {
            nodes: Array2::from_elem((size, size), Node::new()),
            // conn_pos_lookup: Array2::from_shape_fn((size, size), get_connection_coords),
            conn_pos_lookup: Array2::from_elem((size, size), [None;4]),
            connections: Array3::from_shape_simple_fn((2, size, size-1), Connection::new_one),
        };
        res.update_conn_pos_lookup();
        res
    }

    fn update_conn_pos_lookup(&mut self) {
        let shape = self.nodes.shape();
        for row in 0..shape[0] {
            for col in 0..shape[1] {
                self.conn_pos_lookup[(row,col)] = get_connection_coords_2d((row,col), shape[0]);
            }
        }
    }

    pub fn try_activate_random_node(&mut self) -> Result<(), ActivationError> {
        let mut rng = rand::thread_rng();
        let shape = [self.nodes.shape()[0], self.nodes.shape()[1]];
        let node_pos = (rng.gen_range(0..shape[0]), rng.gen_range(0..shape[1]));

        if self.nodes[node_pos].try_activate(1.0).is_ok() {
            for opt_conn_pos in self.conn_pos_lookup[node_pos] {
                if let Some(conn_pos) = opt_conn_pos {
                    self.connections[conn_pos].activate();
                }
            }
        }


        Ok(())
    }

    pub fn update(&mut self) {
        let shape = [self.nodes.shape()[0], self.nodes.shape()[1]];
        let mut new_activations = Array2::<f32>::zeros((shape[0], shape[1]));
        for row in 0..shape[0] {
            for col in 0..shape[1] {
                // let mut total_activation = 0.0;
                let node_pos = (row,col);

                for opt_conn_pos in self.conn_pos_lookup[node_pos] {
                    if let Some(conn_pos) = opt_conn_pos {
                        if let Some(activation) = self.connections[conn_pos].check() {
                            new_activations[node_pos] += activation;
                        }
                    }
                }
            }
        }
        Zip::from(&mut self.nodes).and(&new_activations).and(&self.conn_pos_lookup).for_each(|node, activation, opt_conn_poses| {
            if node.try_activate(*activation).is_ok() {
                for opt_conn_pos in opt_conn_poses {
                    if let Some(conn_pos) = *opt_conn_pos {
                        self.connections[conn_pos].activate();
                    }
                }
            }
        })
    }

    pub fn print(&self) {
        let shape = [self.nodes.shape()[0], self.nodes.shape()[1]];
        let mut board = Array2::from_elem((shape[0] * 2 - 1, shape[1] * 2 - 1), " ");
        for row in 0..shape[0] {
            for col in 0..shape[1] {
                board[(row*2, col*2)] = if self.nodes[(row, col)].is_activated() {
                    "X"
                } else {
                    "O"
                };

                if let Some(conn_pos) = self.conn_pos_lookup[(row,col)][1] {
                    board[(2*row+1, col*2)] = if self.connections[conn_pos].is_active() {
                        "#"
                    } else {
                        "|"
                    };
                }
                
                if let Some(conn_pos) = self.conn_pos_lookup[(row,col)][3] {
                    board[(row*2, col*2+1)] = if self.connections[conn_pos].is_active() {
                        "="
                    } else {
                        "-"
                    };
                }



            }
        }
        
        for row in 0..shape[0]*2-1 {
            for col in 0..shape[1]*2-1 {
                print!("{}", board[(row,col)]);
            }
            println!("");
        }

    }
}


#[cfg(test)]
mod tests {
    use super::Tensor2D;

    #[test]
    fn tensor2d_basic() {
        let mut tensor = Tensor2D::new(4);
        tensor.print();

        assert_eq!(1,1)
    }
}
