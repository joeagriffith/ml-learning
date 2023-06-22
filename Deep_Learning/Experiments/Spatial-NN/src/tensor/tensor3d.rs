use ndarray::{Array3, Array4, Zip};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::prelude::*;

use crate::config::{THRESHOLD_NODE_ACTIVATION, NUM_THREADS};
use crate::structures::*;
use crate::utils::{get_connection_coords_3d, sigmoid, get_neighbours_3d};

pub struct Tensor3D {
    actv_nodes_pos: Vec<(usize, usize, usize)>,
    nodes: Array3<Node>,
    conn_pos_lookup: Array3<[Option<(usize, usize, usize, usize)>;6]>,
    connections: Array4<Connection>,
}

impl Tensor3D {
    pub fn new(size:usize) -> Tensor3D {
        let mut res = Tensor3D {
            actv_nodes_pos: Vec::<(usize, usize, usize)>::new(),
            nodes: Array3::from_elem((size, size, size), Node::new()),
            conn_pos_lookup: Array3::from_elem((size, size, size), [None;6]),
            connections: Array4::from_shape_simple_fn((3, size, size, size-1), Connection::new),
        };

        res.update_conn_pos_lookup();
        res
    }
    /// Creates a new 3d Tensor with all weights of 1.0.
    pub fn new_ones(size:usize) -> Tensor3D {
        let mut res = Tensor3D {
            actv_nodes_pos: Vec::new(),
            nodes: Array3::from_elem((size, size, size), Node::new()),
            conn_pos_lookup: Array3::from_elem((size, size, size), [None;6]),
            connections: Array4::from_shape_simple_fn((3, size, size, size-1), Connection::new_one),
        };

        res.update_conn_pos_lookup();
        res
    }

    fn update_conn_pos_lookup(&mut self) {
        let shape = [self.nodes.shape()[0], self.nodes.shape()[1], self.nodes.shape()[2]];
        for lev in 0..shape[0] {
            for row in 0..shape[1] {
                for col in 0..shape[2] {
                    self.conn_pos_lookup[(lev,row,col)] = get_connection_coords_3d((lev,row,col), shape[0]);
                }
            }
        }
    }

    pub fn try_activate_random_node(&mut self) -> Result<(), ActivationError> {
        let mut rng = rand::thread_rng();
        let shape = [self.nodes.shape()[0], self.nodes.shape()[1], self.nodes.shape()[2]];
        let node_pos = (rng.gen_range(0..shape[0]), rng.gen_range(0..shape[1]), rng.gen_range(0..shape[2]));
 
        if self.nodes[node_pos].try_activate(1.0).is_ok() {
            for opt_conn_pos in self.conn_pos_lookup[node_pos] {
                if let Some(conn_pos) = opt_conn_pos {
                    self.connections[conn_pos].activate();
                }
            }
            self.actv_nodes_pos.push(node_pos);
        }


        Ok(())
    }

    pub fn try_activate_node(&mut self, pos:(usize, usize, usize), activation:f32) {
        if self.nodes[pos].try_activate(activation).is_ok() {
            for opt_conn_pos in self.conn_pos_lookup[pos] {
                if let Some(conn_pos) = opt_conn_pos {
                    self.connections[conn_pos].activate();
                }
            }
        }
        self.actv_nodes_pos.push(pos);
    }

    pub fn update(&mut self) {
        let shape = [self.nodes.shape()[0], self.nodes.shape()[1], self.nodes.shape()[2]];
        let mut new_activations = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
        let mut new_actv_nodes_pos:Vec<(usize, usize, usize)> = Vec::new();
        let mut checked = Array3::<bool>::from_elem((shape[0], shape[1], shape[2]), false);
        for node_pos in &self.actv_nodes_pos {
            for neighbour_pos in get_neighbours_3d(node_pos.clone(), shape[0]) {
                // println!("checking node at pos: {:?}", neighbour_pos);
                if checked[neighbour_pos] == false {
                    for opt_conn_pos in self.conn_pos_lookup[neighbour_pos] {
                        if let Some(conn_pos) = opt_conn_pos {
                            if let Some(activation) = self.connections[conn_pos].check() {
                                new_activations[neighbour_pos] += activation;
                            }
                        }
                    }
                    checked[neighbour_pos] = true;
                    if new_activations[neighbour_pos] > THRESHOLD_NODE_ACTIVATION {
                        new_actv_nodes_pos.push(neighbour_pos);
                    }
                }
            }
        }
        self.actv_nodes_pos = new_actv_nodes_pos;

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

    pub fn node_is_active(&self, pos:(usize, usize, usize)) -> Option<bool> {
        if pos.0 < self.nodes.shape()[0] && pos.1 < self.nodes.shape()[1] && pos.2 < self.nodes.shape()[2] {
            Some(self.nodes[pos].is_activated())
        } else {
            None
        }
    }

    pub fn print(&self) {
        let shape = [self.nodes.shape()[0], self.nodes.shape()[1], self.nodes.shape()[2]];
        let mut board = Array3::from_elem((shape[0], shape[1] * 2 - 1, shape[2] * 2 - 1), " ");
        for lev in 0..shape[0] {
            for row in 0..shape[1] {
                for col in 0..shape[2] {
                    board[(lev,row*2, col*2)] = if self.nodes[(lev,row, col)].is_activated() {
                        "X"
                    } else {
                        "O"
                    };

                    if let Some(conn_pos) = self.conn_pos_lookup[(lev,row,col)][3] {
                        board[(lev, 2*row+1, col*2)] = if self.connections[conn_pos].is_active() {
                            "#"
                        } else {
                            "|"
                        };
                    }

                    if let Some(conn_pos) = self.conn_pos_lookup[(lev,row,col)][5] {
                        board[(lev,row*2, col*2+1)] = if self.connections[conn_pos].is_active() {
                            "="
                        } else {
                            "-"
                        };
                    }

                }
            }
        }
        for lev in 0..shape[0] {
            for row in 0..shape[1]*2-1 {
                for col in 0..shape[2]*2-1 {
                    print!("{}", board[(lev,row,col)]);
                }
                println!("");
            }
            println!("");
        }
    }


    pub fn print_top(&self) {
        let shape = [self.nodes.shape()[0], self.nodes.shape()[1], self.nodes.shape()[2]];
        let mut board = Array3::from_elem((shape[0], shape[1] * 2 - 1, shape[2] * 2 - 1), " ");
        for lev in shape[0]-1..shape[0] {
            for row in 0..shape[1] {
                for col in 0..shape[2] {
                    board[(lev,row*2, col*2)] = if self.nodes[(lev,row, col)].is_activated() {
                        "X"
                    } else {
                        "O"
                    };

                    if let Some(conn_pos) = self.conn_pos_lookup[(lev,row,col)][3] {
                        board[(lev, 2*row+1, col*2)] = if self.connections[conn_pos].is_active() {
                            "#"
                        } else {
                            "|"
                        };
                    }

                    if let Some(conn_pos) = self.conn_pos_lookup[(lev,row,col)][5] {
                        board[(lev,row*2, col*2+1)] = if self.connections[conn_pos].is_active() {
                            "="
                        } else {
                            "-"
                        };
                    }

                }
            }
        }
        for lev in shape[0]-1..shape[0] {
            for row in 0..shape[1]*2-1 {
                for col in 0..shape[2]*2-1 {
                    print!("{}", board[(lev,row,col)]);
                }
                println!("");
            }
            println!("");
        }
    }
}


#[cfg(test)]
mod tests {
    use super::Tensor3D;

    #[test]
    fn tensor3d_basic() {
        let mut tensor1 = Tensor3D::new_ones(4);

        tensor1.try_activate_node((1,1,1), 1.0);
        tensor1.print();
        tensor1.update();
        println!("===============================");
        tensor1.print();

        assert_eq!(1,2)
    }
}

        // For every node