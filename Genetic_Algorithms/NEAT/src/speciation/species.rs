use rand::prelude::*;

use crate::config::DROPOFF_AGE;
use crate::genetics::Genome;

/// A collection of references to genomes which enables crossover of similar topologies and selection of parents based on their fitnesses.
pub struct Species {
    id: usize,
    representative: Genome,
    members: Vec<(usize, f64, f64)>, // (g_ids, fitness, adj_fitness)

    total_fitness: f64,
    max_ever_fitness: f64,
    best_fitness: f64,
    total_adj_fitness: f64,
    gens_stagnated: usize,
    // stagnated: bool,
}
impl Species {

    /// Creates a new Species with the supplied representative.
    pub fn new(id: usize, representative:Genome) -> Self {
        Self {
            id,
            representative,
            members: Vec::new(),

            gens_stagnated: 0,
            total_fitness: 0.0,
            best_fitness: 0.0,
            total_adj_fitness: 0.0,
            max_ever_fitness: 0.0,
            // stagnated: false,
        }
    }

    pub fn new_generation(&mut self) {
        self.members = Vec::new();
        self.total_fitness = 0.0;
        self.total_adj_fitness = 0.0;
        self.best_fitness = 0.0;
        self.gens_stagnated += 1;
    }

    /// Inserts a new genome into the species and updates relevent parameters
    pub fn insert(&mut self, g_id:usize, mut fitness:f64) {
        if fitness > self.max_ever_fitness {
            self.max_ever_fitness = fitness;
            self.gens_stagnated = 0;
        }
        if fitness > self.best_fitness {
            self.best_fitness = fitness;
        }
        if self.gens_stagnated > DROPOFF_AGE {
            fitness *= 0.001;
        }
        self.members.push((g_id, fitness, -1.0));
        self.total_fitness += fitness;
    }

    pub fn sort_by_fitness(&mut self) {
        self.members.sort_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());
    }

    /// Returns sum of adj_fitnesses
    pub fn calc_adj_fitnesses(&mut self) -> f64 {
        let length = self.members.len() as f64;
        let mut sum = 0.0;
        for i in 0..self.members.len() {
            self.members[i].2 = self.members[i].1 / length;
            sum += self.members[i].2;
        }
        self.total_adj_fitness = sum;
        sum
    }

    /// Calculates how many new genomes of this species should be created for the next generation.
    pub fn allowed_offspring(&self, pop_avg_adj_fit: f64) -> usize {
        let average_adj_fitness = self.total_adj_fitness / self.members.len() as f64;
        ((average_adj_fitness / pop_avg_adj_fit) * self.members.len() as f64) as usize
    }

    pub fn is_extinct(&self) -> bool {
        self.members.len() == 0
    }
    
    /// Returns a random genome reference with probability directly proportional to each member's fitness
    pub fn get_fit_member_id(&self) -> usize {
        let mut rem_adj_fit = rand::thread_rng().gen_range(0.0..self.total_adj_fitness);
        let mut index = 0;
        while rem_adj_fit > 0.0 {
            rem_adj_fit -= self.members[index].2;
            index += 1;
        }
        if index >= self.members.len() {
            index = 0;
        }
        self.members[index].0
    }

    pub fn get_first(&self) -> usize {
        self.members[0].0
    }

    /// Returns g_id of the middle member so that self.representative can be updated to median genome.
    /// Members must have been sorted by fitness using sort_by_fitness()
    pub fn get_median_id(&mut self) -> usize {
        if self.members.len() < 3 { 
            self.members[0].0
        } else {
            self.members[self.members.len() / 2].0
        }
    }

    //=========================GETTERS & SETTERS=========================//

    pub fn get_gens_stagnated(&self) -> usize { self.gens_stagnated }
    pub fn get_best_fitness(&self) -> f64 { self.best_fitness }
    pub fn get_mean_fitness(&self) -> f64 { self.total_fitness / self.len() as f64}
    // pub fn is_stagnated(&self) -> bool { self.stagnated }
    pub fn len(&self) -> usize { self.members.len() }
    pub fn get_rep(&self) -> &Genome { &self.representative }
    pub fn get_id(&self) -> usize { self.id }

    pub fn set_rep(&mut self, g:Genome) { self.representative = g; } 

    //=============================DEBUGGING===============================//

    #[allow(dead_code)]
    pub fn get_g_ids(&self) -> Vec<usize> {
        let mut res: Vec<usize> = Vec::new();
        for (g_id, _fitness, _adj_fitness) in &self.members {
            res.push(*g_id);
        }
        res
    }
}

