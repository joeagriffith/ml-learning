mod population;
mod dna;
use std::env;

use population::Population;

fn main() {
    let args: Vec<String> = env::args().collect();
    let target_string = args[1].to_owned();
    let some_population_size = args[2].parse::<usize>();
    let some_mutation_rate = args[3].parse::<f32>();

    
    if some_population_size.is_err() {
        panic!("Invalid argument received for population_size, must be integer.")
    }

    if some_mutation_rate.is_err() {
        panic!("Invalid argument received for mutation_rate, must be float.")
    }

    if *some_mutation_rate.as_ref().unwrap() < 0.0 || *some_mutation_rate.as_ref().unwrap() > 1.0 {
        panic!("Invalid argument received for mutation_rate, must be between 0.0 and 1.0 inclusive.")
    }
    
    let mut population = Population::new(target_string, some_population_size.unwrap(), some_mutation_rate.unwrap());
    let mut fittest = population.evaluate();
    let mut generations = 0;
    
    while fittest.fitness < population.get_target().len().pow(2) {
        // io::stdin().read_line(&mut input).expect("FAILED TO READ LINE");
        generations += 1;
        population.generate();
        fittest = population.evaluate();
        println!("{:?} = {}", fittest.dna, population.get_max_fitness());
    }
    println!("SUCCESS in generation {}: {:?}", generations, fittest.dna);
}


