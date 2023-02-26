use rand::Rng;
use super::nn::*;

// simple xor classification problem
// 00 = 0
// 01 = 1
// 10 = 1
// 11 - 0
pub fn xor_test(organism: &mut NeuralNetwork) -> f64 {

    let mut output:Vec<f64> = Vec::new();
    let mut distance: f64;
    
    organism.feedforward(&vec![0f64, 0f64], &mut output);
    distance = (0f64 - output[0]).abs();
    
    organism.feedforward(&vec![0f64, 1f64], &mut output);
    distance += (1f64 - output[0]).abs();
    
    organism.feedforward(&vec![1f64, 0f64], &mut output);
    distance += (1f64 - output[0]).abs();
    
    organism.feedforward(&vec![1f64, 1f64], &mut output);
    distance += (0f64 - output[0]).abs();

    let fitness = (4f64 - distance).powi(2);
    organism.fitness = fitness;
    fitness
}


pub struct Population {
    nns: Vec<NeuralNetwork>,
    length: usize, 
    mating_pool: Vec<usize>,
    mutation_rate: f64,
    max_fitness: f64,
    fitness_sum: f64,
}

impl Population {

    pub fn new(length: usize, mutation_rate:f64) -> Population {

        let mut nns = Vec::new();

        for _i in 0..length {
            let mut nn = NeuralNetwork::new(2);
            
            nn.add_layer(2);
            nn.add_layer(1);
            nn.compile();

            nns.push(nn);
        }

        let mut pop = Population {
            nns,
            length,
            mating_pool: Vec::new(),
            mutation_rate,
            max_fitness: 0.0,
            fitness_sum: 0.0,
        };
        pop.calculate_fitness();
        pop.update_mating_pool();
        pop
    }

    // calculates the fitness of each member in the population
    pub fn calculate_fitness(&mut self) {
        let mut curr_fitness;
        self.max_fitness = 0.0;
        self.fitness_sum = 0.0;
        for nn in &mut self.nns {
            curr_fitness = xor_test(nn);
            self.fitness_sum += curr_fitness;
            if curr_fitness > self.max_fitness {
                self.max_fitness = curr_fitness;
            }
        }
    }

    // creates a pool of ids, where ids are push x times, depending on how fit the member is compared to the rest of the population
    fn update_mating_pool(&mut self) {
        self.mating_pool = Vec::new();
        for i in 0..self.length {
            let n = (self.nns.get(i).unwrap().fitness as f64 / self.fitness_sum as f64 * self.length as f64) as u16;
            for _j in 0..n {
                self.mating_pool.push(i);
            }
        }
        if self.mating_pool.len() == 0 {
            for i in 0..self.length {
                self.mating_pool.push(i);
            }
        }
    }

    // randomly chooses 2 parents from the mating pool, fitter agents are more frequent in the pool so are more likely to be chosen
    fn reproduce(&mut self) {
        let mut new_pop = Vec::new();

        for _i in 0..self.length {
            if self.mating_pool.len() == 0 { panic! ("EMPTY MATING POOL! NOT ENOUGH LONELY SINGLES IN YOUR AREA");}
            let parent1 = self.nns.get(*self.mating_pool.get(rand::thread_rng().gen_range(0..self.mating_pool.len())).unwrap()).unwrap();
            let parent2 = self.nns.get(*self.mating_pool.get(rand::thread_rng().gen_range(0..self.mating_pool.len())).unwrap()).unwrap();

            let child = crossover(parent1, parent2, self.mutation_rate);

            new_pop.push(child);
        }
        self.nns = new_pop;
    }

    // returns the fittest network
    pub fn evaluate(&self) -> NeuralNetwork {
        let mut fittest:NeuralNetwork = self.nns.get(0).unwrap().clone();
        for i in 0..self.length {
            if self.nns.get(i).unwrap().fitness > fittest.fitness {
                fittest = self.nns.get(i).unwrap().clone();
            }
        }
        fittest
    }

    pub fn generate(&mut self) {
        self.reproduce();
        
        self.calculate_fitness();
        self.update_mating_pool();
    }

    pub fn get_max_fitness (&self) -> f64 { self.max_fitness }
}