mod matrix;
mod nn;
mod population;

use std::env;
use nn::NeuralNetwork;



fn main() {
    let args: Vec<String> = env::args().collect();

    let some_population_size = args[1].parse::<usize>();
    let some_mutation_rate = args[2].parse::<f64>();

    if some_population_size.is_err() {
        panic!("Invalid argument received for population_size, must be integer.");
    }
    if some_mutation_rate.is_err() {
        panic!("Invalid argument supplied for mutation_rate, must be float.");
    }

    if *some_mutation_rate.as_ref().unwrap() < 0.0 || *some_mutation_rate.as_ref().unwrap() > 1.0 {
        panic!("Invalid argument supplied, mutation rate must be between 0 and 1 inclusive.")
    }

    let mut population = population::Population::new(some_population_size.unwrap(), some_mutation_rate.unwrap());
    let mut champion:Option<NeuralNetwork> = None;
    while champion.is_none() {
        population.generate();
        if population.get_max_fitness() > 15.5f64 {
            champion = Some(population.evaluate())
        }
        println!("Highest Fitness: {}", population.get_max_fitness());
    }
    println!("Training Finished, Champion:");
    champion.unwrap().print();

}