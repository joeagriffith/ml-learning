use super::dna;
use rand::Rng;

pub struct Population {
    dnas: Vec<dna::DNA>,
    pop_size: usize, 
    target: String,
    mating_pool: Vec<usize>,
    mutation_rate: f32,
    max_fitness:usize,
    fitness_sum: usize,
}

impl Population {

    pub fn new(target: String, pop_size: usize, mutation_rate:f32) -> Population {

        let mut dnas:Vec<dna::DNA> = Vec::new();

        for _i in 0..pop_size {
            let mut s = String::new();
            for _j in 0..target.len() {
                s.push(dna::random_char());
            }
            let dna = dna::DNA::new(s);
            dnas.push(dna);
        }

        let mut pop = Population {
            dnas,
            pop_size,
            target,
            mating_pool: Vec::new(),
            mutation_rate,
            max_fitness: 0,
            fitness_sum: 0,
        };
        pop.calculate_fitness();
        pop.update_mating_pool();
        pop
    }

    pub fn calculate_fitness(&mut self) {
        let mut curr_fitness;
        self.max_fitness = 0;
        self.fitness_sum = 0;
        for dna in &mut self.dnas {
            curr_fitness = dna.calc_fitness(&self.target);
            self.fitness_sum += curr_fitness;
            if curr_fitness > self.max_fitness {
                self.max_fitness = curr_fitness;
            }
        }
    }

    fn update_mating_pool(&mut self) {
        self.mating_pool = Vec::new();
        for i in 0..self.pop_size {
            let n = (self.dnas.get(i).unwrap().fitness as f32 / self.fitness_sum as f32 * self.pop_size as f32) as u16;
            for _j in 0..n {
                self.mating_pool.push(i);
            }
        }
    }

    fn reproduce(&mut self) {
        let mut new_pop:Vec<dna::DNA> = Vec::new();

        for _i in 0..self.pop_size {
            let parent1 = self.dnas.get(*self.mating_pool.get(rand::thread_rng().gen_range(0..self.mating_pool.len())).unwrap()).unwrap();//.chars();
            let parent2 = self.dnas.get(*self.mating_pool.get(rand::thread_rng().gen_range(0..self.mating_pool.len())).unwrap()).unwrap();//.chars();
            // let parent1 = self.get_parent().unwrap();
            // let parent2 = self.get_parent().unwrap();

            let child = dna::crossover(parent1, parent2, self.mutation_rate);

            new_pop.push(child);
        }
        self.dnas = new_pop;
    }

    pub fn evaluate(&self) -> dna::DNA {
        let mut fittest:dna::DNA = self.dnas.get(0).unwrap().clone();
        for i in 0..self.pop_size {
            if self.dnas.get(i).unwrap().fitness > fittest.fitness {
                fittest = self.dnas.get(i).unwrap().clone();
            }
        }
        fittest
    }

    pub fn generate(&mut self) {
        self.reproduce();
        
        self.calculate_fitness();
        self.update_mating_pool();
        // self.calc_probs();
    }

    pub fn get_max_fitness (&self) -> usize { self.max_fitness }
    pub fn get_target (&self) -> &String { &self.target }
}