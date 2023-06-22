use super::gene::Gene;
use std::collections::HashMap;

// Specifies whether a node is an input, bias, output or hidden node.
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum NodeType {
    Input,
    Bias,
    Output,
    Hidden
}


/// Encoding for a node in a neural network
#[derive(Clone, Debug)]
pub struct NodeGene {
    node_type: NodeType,
    innovation_number: usize,
    x: f64,
    y: f64,
}

impl Gene for NodeGene {
    fn get_innov(&self) -> usize { self.innovation_number }
}


impl NodeGene {

    /// Constructs a new NodeGene.
    /// Must be fully specified 
    pub fn new(node_type:NodeType, innovation_number:usize, x:f64, y:f64) -> NodeGene {
        NodeGene {
            node_type,
            innovation_number,
            x,
            y, 
        }
    }

    //=========================GETTERS & SETTERS=========================//

    pub fn get_nodetype(&self) -> NodeType { self.node_type }
    pub fn get_x(&self) -> f64 { self.x }
    pub fn get_y(&self) -> f64 { self.y }

    //=============================DEBUGGING===============================//

    pub fn print(&self) {
        println!("{:?} Node: {}, x : {}, y: {} ", self.get_nodetype(), self.get_innov(), self.get_x(), self.get_y());
    }
    #[allow(dead_code)]
    pub fn xor_node_gene_pool() -> HashMap<usize, NodeGene> {

        let mut node_gene_pool:HashMap<usize, NodeGene> = HashMap::new();

        // Insert input, bias & output nodes into pool, required to create population
        node_gene_pool.insert(0, NodeGene::new(NodeType::Input, 0, 0.0, 0.75));
        node_gene_pool.insert(1, NodeGene::new(NodeType::Input, 1, 0.0, 0.5));
        node_gene_pool.insert(2, NodeGene::new(NodeType::Bias, 2, 0.0, 0.1));
        node_gene_pool.insert(3, NodeGene::new(NodeType::Output, 3, 1.0, 0.5));
        node_gene_pool.insert(4, NodeGene::new(NodeType::Hidden, 4, 0.5, 0.75));
        node_gene_pool.insert(5, NodeGene::new(NodeType::Hidden, 5, 0.5, 0.5));

        node_gene_pool
    }
}
