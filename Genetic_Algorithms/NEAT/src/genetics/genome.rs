use std::collections::HashMap;
use ndarray::Array2;

use crate::config::{INPUTS, BIAS, OUTPUTS};
use crate::util::VecSet;

use super::{ConnectionGene, NodeGene, conn_hashcode};

///An encoding for a neural network, which provides mutation, crossover & speciation functionality
#[derive(Clone)]
pub struct Genome {
    nodes: VecSet<NodeGene>, 
    connections: VecSet<ConnectionGene>,
}

impl Genome {

    /// Constructs a new Genome with zero connections and zero nodes.
    pub fn new() -> Self {
        Self {
            nodes: VecSet::new(),
            connections: VecSet::new(),
        }
    }

    /// Constructs a new Genome with input, bias & output nodes, but no connections.
    pub fn new_init(node_pool:&HashMap<usize, NodeGene>) -> Self {
        Self {
            nodes: {
                // Store a clone of input, bias and outputs nodes
                let mut init_nodes = VecSet::new();
                for i in 0..(INPUTS + BIAS as usize + OUTPUTS) {
                    init_nodes.push(node_pool.get(&i).unwrap().clone());
                }
                init_nodes
            },
            connections: VecSet::new(),
        }
    }
    
    /// Constructs a new Genome with input, bias & output nodes.
    /// Each input and bias node has a connection to each output node.
    pub fn new_fully_connected(node_pool: &HashMap<usize, NodeGene>, conn_pool: &HashMap<usize, ConnectionGene>) -> Self {
        Self {
            nodes: {
                let mut init_nodes = VecSet::new();
                for i in 0..(INPUTS + BIAS as usize + OUTPUTS) {
                    init_nodes.push(node_pool.get(&i).unwrap().clone());
                }
                init_nodes
            },
            connections: {
                let mut init_conns = VecSet::new();
                for from in 0..(INPUTS + BIAS as usize) {
                    for to in (INPUTS + BIAS as usize)..(INPUTS + BIAS as usize + OUTPUTS) {
                        init_conns.push(conn_pool.get(&conn_hashcode(from, to))
                        .unwrap().clone());
                    }
                }
                init_conns
            }
        }
    }

    /// Returns the adjacency matrix of the genome's nodes.
    /// Connections are one-way.
    /// Row corresponds to 'from' & column to 'to' node.
    fn adjacency_matrix(&self) -> Array2<u8> {
        if let Some(max_innov) = self.nodes.max_innov() {
            let mut res = Array2::<u8>::zeros((max_innov + 1, max_innov + 1));
            for connection in self.connections.iter() {
                let from = connection.get_from();
                let to = connection.get_to();
                res[(from, to)] = 1;
            }
            return res
        } else {
            return Array2::<u8>::zeros((0,0));
        }
    }

    /// Checks whether a connection already exists between some 'from_node' to some 'to_node'
    /// Utilises recursive wizardry
    pub fn is_connected(&self, from_node_innov:usize, to_node_innov:usize) -> bool {
        let adjacency_matrix = self.adjacency_matrix();
        for to in 0..self.nodes.len() {
            if adjacency_matrix[(from_node_innov, to)] == 1 {
                if to == to_node_innov {
                    return true
                } else {
                    if self.is_connected(to, to_node_innov) == true {
                        return true
                    }
                }
            }
        }
        false

        // OPTION 1: looks like O(c^2) atleast (in my medicated state atleast) where c is connections
        //create empty Vec of node_innovs
        //iterate over connections
            // If conn_from = from_node_inov
                // if conn_to = to_node_innov
                    // return true
                // else 
                    // store conn_to in Vec

        //while Vec.len > 0
            // Iterate over connections
                //If conn_from = from_node_innov
                    //If conn_to = to_node_innov
                        //return true
                    // Else 
                        //store conn_to in Vec
        // Return False

        // OPTION 2 .. looks  like O(2N + C) n is nodes, c is conns


        // Build directed graph of nodes from connection genes. 
        // Similar to the NN built in feed forward, except next_nodes are stored instead of input_nodes
            // We can then do a DFS/BFS from the from_node_innov for the to_node_innov
    }

    //================================GETTERS==========================//
    pub fn get_nodes(&self) -> &VecSet<NodeGene> { &self.nodes }
    pub fn get_nodes_mut(&mut self) -> &mut VecSet<NodeGene> { &mut self.nodes }
    pub fn get_connections(&self) -> &VecSet<ConnectionGene> { &self.connections }
    pub fn get_connections_mut(&mut self) -> &mut VecSet<ConnectionGene> { &mut self.connections }


















    //=============================DEBUGGING===============================//

    #[allow(dead_code)]
    pub fn print(&self) {
        println!("======Nodes======");
        for node in self.nodes.iter() {
            node.print();
        }
        println!("===Connections===");
        for conn in self.connections.iter() {
            conn.print();
        }
    }

    //ensure genome has required nodes for input, bias and output
    #[allow(dead_code)]
    fn has_ibo(&self) -> bool {
        let mut res = true;
        for i in 0..(INPUTS + BIAS as usize + OUTPUTS) {
            if !self.nodes.contains_innov(i) {
                res = false;
            }
        }
        res
    }

    #[allow(dead_code)]
    pub fn new_init_xor() -> Genome {
        let mut g = Genome {
            nodes: {
                // Store a clone of input, bias and outputs nodes
                let mut init_nodes = VecSet::new();
                init_nodes.push(NodeGene::new(crate::NodeType::Input, 0, 0.0, 0.75));
                init_nodes.push(NodeGene::new(crate::NodeType::Input, 1, 0.0, 0.5));
                init_nodes.push(NodeGene::new(crate::NodeType::Bias, 2, 0.0, 0.1));
                init_nodes.push(NodeGene::new(crate::NodeType::Output, 3, 1.0, 0.5));
                init_nodes.push(NodeGene::new(crate::NodeType::Hidden, 4, 0.5, 0.75));
                init_nodes.push(NodeGene::new(crate::NodeType::Hidden, 5, 0.5, 0.5));
                init_nodes
            },
            connections: VecSet::new(),
        };
        g.xor_connect();
        g
    }
    
    #[allow(dead_code)]
    pub fn xor_rand_connect(&mut self) {
        
        self.connections.push(ConnectionGene::new(0, 0, 4));
        self.connections.push(ConnectionGene::new(1, 0, 5));
        self.connections.push(ConnectionGene::new(2, 1, 4));
        self.connections.push(ConnectionGene::new(3, 1, 5));

        self.connections.push(ConnectionGene::new(4, 4, 3));
        self.connections.push(ConnectionGene::new(5, 5, 3));

        self.connections.push(ConnectionGene::new(6, 2, 3));
        self.connections.push(ConnectionGene::new(7, 2, 4));
        self.connections.push(ConnectionGene::new(8, 2, 5));
    }

    #[allow(dead_code)]
    pub fn xor_connect(&mut self) {
        self.connections.push(ConnectionGene::new_explicit(0, 20.0, true, 0, 4));
        self.connections.push(ConnectionGene::new_explicit(1, -20.0, true, 0, 5));
        self.connections.push(ConnectionGene::new_explicit(2, 20.0, true, 1, 4));
        self.connections.push(ConnectionGene::new_explicit(3, -20.0, true, 1, 5));

        self.connections.push(ConnectionGene::new_explicit(4, 20.0, true, 4, 3));
        self.connections.push(ConnectionGene::new_explicit(5, 20.0, true, 5, 3));

        self.connections.push(ConnectionGene::new_explicit(6, 10.0, true, 2, 4));
        self.connections.push(ConnectionGene::new_explicit(7, -30.0, true, 2, 5));
        self.connections.push(ConnectionGene::new_explicit(8, 30.0, true, 2, 3));
    }
}






















#[cfg(test)]
mod tests {
    use super::Genome;

    #[test]
    fn is_connected_xor() {
        let genome = Genome::new_init_xor();
        assert!(genome.is_connected(0, 3));
        assert!(genome.is_connected(1, 3));
        assert!(genome.is_connected(2, 3));
        assert!(!genome.is_connected(4, 1));
        assert!(!genome.is_connected(1, 2));
        assert!(!genome.is_connected(3, 1));
    }
}