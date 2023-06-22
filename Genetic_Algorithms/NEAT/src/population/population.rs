use std::collections::HashMap;

use crate::config::{POPULATION_SIZE, TARGET_SPECIES_NUM, COMPATABILITY_MODIFIER, COMPATABILITY_THRESHOLD, COMPATABILITY_MIN, ELITISM, NUM_THREADS};
use crate::genetics::{NodeGene, ConnectionGene, Genome, distance, crossover};
use crate::speciation::{Species};

use crate::util::ThreadPool;
use std::sync::{Arc, Mutex, RwLock};

/// The container for all the genomes, generational data and population-wide functions.
pub struct Population {
    champion: Option<Genome>,
    pub organisms: Arc<RwLock<Vec<Genome>>>,
    to_mutate: [bool;POPULATION_SIZE], // prevents genomes reproduced via elitism from being mutated.
    fitness_arr: Arc<Mutex<[f64;POPULATION_SIZE]>>,
    max_fitness: f64,

    species_vec: Vec<Species>,
    compatability_threshold: f64,
    pop_avg_adj_fit: f64,
    species_id_counter: usize,
    generation: usize,
    gens_stagnated: usize,
}

impl Population {
    /// Creates a new population of initialised genomes.
    pub fn new(node_pool:&HashMap<usize, NodeGene>) -> Self {
        Self {

            organisms: 
            {
                let v = Arc::new(RwLock::new(Vec::new()));
                let mut v_writer = v.write().unwrap();
                for _i in 0..POPULATION_SIZE {
                    v_writer.push(Genome::new_init(node_pool));
                }
                drop(v_writer);
                v
            },
            to_mutate: [true;POPULATION_SIZE],
            fitness_arr: Arc::new(Mutex::new([0.0;POPULATION_SIZE])),
            max_fitness: 0.0,
            champion: None,

            species_vec: Vec::new(),
            compatability_threshold: COMPATABILITY_THRESHOLD,
            pop_avg_adj_fit: 0.0,
            species_id_counter: 0,
            generation: 0,
            gens_stagnated: 0,
        }
    }
    pub fn new_fully_connected(node_pool:&HashMap<usize, NodeGene>, conn_pool:&HashMap<usize, ConnectionGene>) -> Self {
        Self {

            organisms: 
            {
                let mut v = Arc::new(RwLock::new(Vec::new()));
                let mut v_writer = v.write().unwrap();
                for _i in 0..POPULATION_SIZE {
                    v_writer.push(Genome::new_fully_connected(node_pool, conn_pool));
                }
                drop(v_writer);
                v
            },
            to_mutate: [true;POPULATION_SIZE],
            fitness_arr: Arc::new(Mutex::new([0.0;POPULATION_SIZE])),
            max_fitness: 0.0,
            champion: None,

            species_vec: Vec::new(),
            compatability_threshold: COMPATABILITY_THRESHOLD,
            pop_avg_adj_fit: 0.0,
            species_id_counter: 0,
            generation: 0,
            gens_stagnated: 0,
        }
    }

    /// Runs each genome through the supplied environment.
    /// Updates the champion genome if a better one is found.
    pub fn calculate_fitnesses(&mut self, env:for<'r> fn(&'r Genome) -> f64 ) {
        self.gens_stagnated += 1;

        let pool = ThreadPool::new(NUM_THREADS, &self.fitness_arr, &self.organisms);
        for i in 0..POPULATION_SIZE {
            pool.execute(i, env);
        }
        drop(pool);
        let fitnesses = self.fitness_arr.lock().unwrap();
        let organisms = self.organisms.read().unwrap();
        for i in 0..POPULATION_SIZE {
            if fitnesses[i] > self.max_fitness {
                self.max_fitness = fitnesses[i];
                self.champion = Some(organisms[i].clone());
                self.gens_stagnated = 0;
            }
        }
    }

    /// Iterates over each genome in the population and places them in their most similar species.
    /// If the genome's distance from the the most similar species is too large, above the compatability threshold, then a new species is created.
    pub fn speciate(&mut self) {
    
        for species in self.species_vec.iter_mut() {
            species.new_generation();
        }
    
        // Insert each genome into MOST SIMILAR species
        for i in 0..POPULATION_SIZE {
            let mut smallest_dist = f64::MAX;
            let mut index = usize::MAX;
            for j in 0..self.species_vec.len() { 
                let dist = distance(&self.organisms.read().unwrap()[i], self.species_vec[j].get_rep()); 
                if dist < smallest_dist { 
                    smallest_dist = dist;
                    index = j;
                }
            }
            if smallest_dist < self.compatability_threshold {
                self.species_vec[index].insert(i, self.fitness_arr.lock().unwrap()[i]);
            }
            else {
                let length = self.species_vec.len();
                self.species_vec.push(Species::new(self.species_id_counter, self.organisms.read().unwrap()[i].clone()));
                self.species_vec[length].insert(i, self.fitness_arr.lock().unwrap()[i]);
                self.species_id_counter += 1;

            }
        }
    
        // // Remove all extinct species from species_vec
        let mut extinct_species_indexes = Vec::<usize>::new();
        for i in 0..self.species_vec.len() {
            if self.species_vec[i].is_extinct() {
                extinct_species_indexes.push(i);
            }
        }
        let mut iter = 0;
        for index in extinct_species_indexes.iter() {
            self.species_vec.remove(index - iter);
            iter += 1;
        }
    
        // Calculate adjusted fitnesses for reproduction and update species representatives
        let mut pop_avg_adj_fit = 0.0;
        for species in self.species_vec.iter_mut() {
            pop_avg_adj_fit += species.calc_adj_fitnesses();
            species.sort_by_fitness();
            // dont update rep when only caring about topology
            // let median_genome = self.organisms[species.get_median_id()].clone();
            // species.set_rep(median_genome);
        }
        self.pop_avg_adj_fit = pop_avg_adj_fit / POPULATION_SIZE as f64;
    
        // Adjust compatability_threshold to help reduce/increase number of species in next generation
        if self.species_vec.len() > TARGET_SPECIES_NUM {
            self.compatability_threshold += COMPATABILITY_MODIFIER;
        } else if self.species_vec.len() < TARGET_SPECIES_NUM && self.compatability_threshold > COMPATABILITY_MIN{
            self.compatability_threshold -= COMPATABILITY_MODIFIER;
        }
        println!("new CT: {}", self.compatability_threshold);
    }


    /// Creates an entirely new population via intra-species crossover.
    /// Parents are randomly selected with a probability directly proportional to their fitness.
    pub fn reproduce(&mut self) {
        self.generation += 1;
        let new_pop = Arc::new(RwLock::new(Vec::<Genome>::new()));
        let mut pop_idx = 0;
        let mut new_pop_writer = new_pop.write().unwrap();
        for species in &self.species_vec {
            let allowed_offspring = species.allowed_offspring(self.pop_avg_adj_fit);

            for i in 0..allowed_offspring {
                if i < ELITISM {
                    new_pop_writer.push(self.organisms.read().unwrap()[species.get_first()].clone());
                    self.to_mutate[pop_idx] = false;
                    pop_idx += 1;
                } else {
                    let parent1_id = species.get_fit_member_id();
                    let parent2_id = species.get_fit_member_id();
                    let fitnesses = self.fitness_arr.lock().unwrap();
                    let p1_fitter_than_p2: bool = fitnesses[parent1_id] > fitnesses[parent2_id];
                    new_pop_writer.push(crossover(&self.organisms.read().unwrap()[parent1_id], &self.organisms.read().unwrap()[parent2_id], p1_fitter_than_p2));
                    self.to_mutate[pop_idx] = true;
                    pop_idx += 1;
                }
            }
        }

        // A fill to ensure popultation stays at POPULATION_SIZE
        // MIGHT not be necessary
        while new_pop_writer.len() < POPULATION_SIZE {
            if let Some(g) = &self.champion {
                new_pop_writer.push(g.clone());
                self.to_mutate[pop_idx] = true;
                pop_idx += 1;
            } else {
                panic!("Champion not found whent attempting to fill incomplete population.")
            }
        }
        drop(new_pop_writer);
        self.organisms = new_pop;
    }

    /// Prints analysis data to the console, to help the user track the progress.
    pub fn generation_info(&self) {
        // print generation number
        // print generation best fitness
        // print generation mean fitness
        // print each species, ID, num_members, best_fitness
        println!("===== Generation {} ======", self.generation);
        println!("  ID    Num Members      Mean Fitness       Best Fitness      Gens Stagnant");
        println!("Pop,          {POPULATION_SIZE},          {:.2},         {:.2}              {}",  self.get_mean_fitness(), self.max_fitness, self.gens_stagnated);
        for species in self.species_vec.iter() {
            println!("{}               {}           {:.2}             {:.2}           {}", species.get_id(), species.len(), species.get_mean_fitness(), species.get_best_fitness(), species.get_gens_stagnated());
        }

    }
    pub fn get_champion(&self) -> Option<Genome> { self.champion.clone() }

    pub fn get_mean_fitness(&self) -> f64 { self.fitness_arr.lock().unwrap().iter().sum::<f64>() / POPULATION_SIZE as f64}
    pub fn get_species_vec(&self) -> &Vec<Species> { &self.species_vec }
    //pub fn get_genome(&self, g_id:usize) -> &Genome { &self.organisms.read().unwrap()[g_id] }
    //pub fn get_genome_mut(&mut self, g_id:usize) -> &mut Genome { &mut self.organisms.write().unwrap()[g_id]}
    pub fn get_fitness_vec(&self) -> [f64;POPULATION_SIZE] { self.fitness_arr.lock().unwrap().clone() }
    pub fn get_max_fitness(&self) -> f64 { self.max_fitness }
    pub fn species_len(&self) -> usize { self.species_vec.len() }
    pub fn should_mutate(&self, g_id:usize) -> bool {self.to_mutate[g_id]}

    //=============================DEBUGGING===============================//

    #[allow(dead_code)]
    // pub fn get_organisms_mut(&mut self) -> &mut Vec<Genome> { &mut self.organisms.write().unwrap() }
    // pub fn get_organisms(&self) -> &Vec<Genome> { &self.organisms.read().unwrap() }

    pub fn print_species_rep(&self) {
        for species in self.species_vec.iter() {
            species.get_rep().print();
        }
    }

    pub fn new_xor() -> Self {
        Self {

            organisms: 
            {
                let mut v = Arc::new(RwLock::new(Vec::new()));
                let mut v_writer = v.write().unwrap();
                for _i in 0..POPULATION_SIZE {
                    v_writer.push(Genome::new_init_xor());
                }
                drop(v_writer);
                v
            },
            to_mutate: [true;POPULATION_SIZE],
            fitness_arr: Arc::new(Mutex::new([0.0;POPULATION_SIZE])),
            max_fitness: 0.0,
            champion: None,

            species_vec: Vec::new(),
            compatability_threshold: COMPATABILITY_THRESHOLD,
            pop_avg_adj_fit: 0.0,
            species_id_counter: 0,
            generation: 0,
            gens_stagnated: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::population::Population;
    use crate::genetics::{NodeGene}; 
    use crate::config::{POPULATION_SIZE};
    use crate::test_environments::xor;

    #[test]
    fn xor_zero_fitness_test() {
        let node_gene_pool= NodeGene::xor_node_gene_pool();
        let mut population = Population::new(&node_gene_pool);
        population.calculate_fitnesses(xor);
        let sum:f64 = population.get_fitness_vec().iter().sum();
        assert!(sum == (POPULATION_SIZE * 4) as f64);
    }
}
