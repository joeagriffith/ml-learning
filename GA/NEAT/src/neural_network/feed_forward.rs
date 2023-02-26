use std::collections::HashMap;

use crate::config::{INPUTS, BIAS, OUTPUTS};
use crate::genetics::{Genome, Gene};
use super::node::Node;

/// A basic Feedforward neural network with no recurrency.
pub struct FeedForwardNetwork {
    nodes: HashMap<usize, Node> // innovation_number -> Node. Connections are stored in their 'to' Nodes
}

impl FeedForwardNetwork {

    /// Constructs a new Neural network from a given genome.
    pub fn new(genome:&Genome) -> FeedForwardNetwork {

        let mut network = FeedForwardNetwork {
            nodes:HashMap::new(),
        };

        // Generate Nodes for each node_gene in the genome
        for node_gene in genome.get_nodes().iter() {
            network.nodes.insert(node_gene.get_innov(), Node::new());
        }

        // Inform each node which node's are it's inputs, and with what weights
        for conn_gene in genome.get_connections().iter() {
            if conn_gene.is_enabled() {
                let from_inno = conn_gene.get_from();
                let to = network.nodes.get_mut(&conn_gene.get_to()).unwrap();
                to.push(from_inno, conn_gene.get_weight());
            }
        }

        network
    }


    fn evaluate_node(&mut self, node_id:usize) -> f64 {

        // If activation has already been calculated or node is an input, we can retrieve it.
        let activation = self.nodes.get(&node_id).unwrap().get_activation();
        if activation.is_some() { 
            return activation.unwrap()
        }

        // Otherwise we have to calculate it. We will set the node's activation for later quick use.
        let mut new_activation = 0.0;
        let input_node_ids = self.nodes.get(&node_id).unwrap().get_inputs_node_ids().clone();
        let weights = self.nodes.get(&node_id).unwrap().get_weights().clone();

        // RECURSIVE: for each input, check if it has an activation. If not evaluate that node first, then retrieve it's activation
        for i in 0..input_node_ids.len() {
            let input_node_id = input_node_ids[i];
            let input_activation = self.evaluate_node(input_node_id);
            new_activation += input_activation * weights[i];
        }

        // Apply sigmoid activation function
        new_activation = 1.0 / (1.0 + (-new_activation).exp());
        self.nodes.get_mut(&node_id).unwrap().set_activation(Some(new_activation));

        new_activation
    }

    fn clear_activations(&mut self) {
        for node in self.nodes.values_mut() {
            node.set_activation(None);
        }
    }
    
    /// feeds a supplied input into the net and returns the calculated output.
    pub fn activate(&mut self, inputs:Vec<f64>) -> [f64;OUTPUTS] {
        
        //Ensure no residual information remains from previous activations of network.
        self.clear_activations();
        if inputs.len() != INPUTS {
            panic!("FFNN received input of incorrect length");
        }

        let mut output = [0.0;OUTPUTS];

        // Set input nodes activation to input
        for i in 0..INPUTS {
            self.nodes.get_mut(&i).unwrap().set_activation(Some(inputs[i]));
        }

        // Set bias activation to -1 (bias to each node is manipulated by it's corresponding weight)
        if BIAS {
            self.nodes.get_mut(&INPUTS).unwrap().set_activation(Some(-1.0));
        }

        // Evaluate each output node, which recursively evaluates dependent nodes on previous hidden and input "layers"
        for i in 0..OUTPUTS {
            output[i] = self.evaluate_node(INPUTS + BIAS as usize + i);
        }
        //println!("input: {:?} output: {:?}", inputs, output);
        
        output
    }

}