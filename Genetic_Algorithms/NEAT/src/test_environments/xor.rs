use crate::config::{INPUTS, BIAS, OUTPUTS};
use crate::genetics::Genome;
use crate::neural_network::FeedForwardNetwork;

pub fn xor(organism: &Genome) -> f64 {
    if INPUTS != 2 || OUTPUTS != 1 || BIAS != true {
        panic!("invalid inputs, bias and outputs for xor");
    }
    let mut output:[f64;OUTPUTS];
    let mut distance:f64;
    // let mut neural_network = NeuralNetwork::new(organism);
    let mut neural_network = FeedForwardNetwork::new(organism);

    // organism.print();
    let mut input = vec![0.0;INPUTS];

    input = vec![0.0, 0.0];
    output = neural_network.activate(input.clone());
    distance = (0f64 - output[0]).abs();
    // println!("input: {:?}     output: {:?}", input, output);
    
    input = vec![1.0, 0.0];
    output = neural_network.activate(input.clone());
    distance += (1f64 - output[0]).abs();
    // println!("input: {:?}     output: {:?}", input, output);
    
    input = vec![0.0, 1.0];
    output = neural_network.activate(input.clone());
    distance += (1f64 - output[0]).abs();
    // println!("input: {:?}     output: {:?}", input, output);
    
    input = vec![1.0, 1.0];
    output = neural_network.activate(input.clone());
    distance += (0f64 - output[0]).abs();
    // println!("input: {:?}     output: {:?}", input, output);

    let fitness = 4f64 - distance;
    // println!("fitness: {}", fitness.powi(2));
    fitness.powi(2)
}