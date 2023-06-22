/// A node of a neural network which contains IDs of the node which are inputs to itself. 
pub struct Node {
    activation: Option<f64>,
    input_node_ids: Vec<usize>, // (node_id, weight)
    weights: Vec<f64>,
}

impl Node {

    /// Creates a new node with empty input_node_ids, weights and a 'None' activation
    pub fn new() -> Self {
        Self {
            activation: None,
            input_node_ids: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// stores a reference to another input node.
    pub fn push(&mut self, node_id:usize, weight:f64) {
        if self.input_node_ids.contains(&node_id) { panic!("tried to insert duplicate input node_id to Node.") }
        else {
            self.input_node_ids.push(node_id);
            self.weights.push(weight);
        }
    }

    pub fn get_activation(&self) -> &Option<f64> { &self.activation }
    pub fn get_inputs_node_ids(&self) -> &Vec<usize> { &self.input_node_ids }
    pub fn get_weights(&self) -> &Vec<f64> { &self.weights }

    pub fn set_activation(&mut self, activation:Option<f64>) { self.activation = activation; }
}