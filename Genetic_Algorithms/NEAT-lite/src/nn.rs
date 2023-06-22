use super::matrix::*;

#[derive(Clone)]
pub struct NeuralNetwork {
    inputs: usize,
    pub layers: Vec<Layer>,
    pub fitness: f64,

}

impl NeuralNetwork {

    pub fn new(inputs:usize) -> NeuralNetwork {
        NeuralNetwork {
            inputs,
            layers: Vec::new(),
            fitness: 0.0,
        }
    }

    pub fn add_layer(&mut self, nodes:usize) {
        self.layers.push(Layer::new(nodes))
    }

    pub fn feedforward(&self, input:&Vec<f64>, output:&mut Vec<f64>){
        let mut calc = Matrix::from_vec(input);
        for layer in &self.layers {
            calc = dot(&layer.weights, &calc);
            calc.element_wise_add(&layer.biases);
            calc.element_wise_sigmoid();
        }
        *output = calc.to_vec();
    }

    pub fn compile(&mut self) {
        let mut prev_nodes = self.inputs;
        for layer in &mut self.layers {
            layer.init_weights(prev_nodes);
            prev_nodes = layer.nodes;
        }
    }

    pub fn print(&self) {
        for layer in &self.layers {
            println!("Weights:");
            layer.weights.dbg();
            println!("Biases:");
            layer.biases.dbg();
        }
    }

}

#[derive(Clone)]
pub struct Layer {
    nodes: usize,
    pub weights: Matrix,
    pub biases: Matrix,
}

impl Layer {
    fn new(nodes:usize) -> Layer {
        Layer {
            nodes,
            biases: Matrix::new_rand(nodes, 1),
            weights: Matrix::new(0, 0, 0.0),
        }
    }

    fn init_weights (&mut self, input_nodes:usize) {
        self.weights = Matrix::new_rand(self.nodes, input_nodes);
    }
}


// produces a child NN from two parent NNs where each weight is randomly chosen from the two parents
// the mutation_rate is the chance the a weight being randomised. This occurs independently for each weight.
pub fn crossover(nn1:&NeuralNetwork, nn2:&NeuralNetwork, mutation_rate:f64) -> NeuralNetwork {
    let mut res = nn1.clone();
    for i in 0..res.layers.len() {
        let res_layer = res.layers.get_mut(i).unwrap();
        let nn2_layer = nn2.layers.get(i).unwrap();
        res_layer.weights.crossover(&nn2_layer.weights, mutation_rate);
        res_layer.biases.crossover(&nn2_layer.biases, mutation_rate);
    }
    res
}