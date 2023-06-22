mod neural_network;
mod genetics;
mod config;
mod util;
mod speciation;
mod mutation;
mod population;
pub mod test_environments;

use std::collections::HashMap;
use std::time::Instant;

pub use genetics::{Genome, ConnectionGene, conn_hashcode, NodeGene, NodeType};
pub use speciation::{Species};
pub use neural_network::FeedForwardNetwork;
pub use config::{INPUTS, BIAS, OUTPUTS};
pub use population::Population;
pub use mutation::{mutate};
pub use util::softmax;

pub struct Neat {
    node_pool: HashMap<usize, NodeGene>, // hashcode -> NodeGene, hashcode is (INPUTS + BIAS as usize + OUTPUTS) + innov_num of the connection consumed
    connection_pool: HashMap<usize, ConnectionGene>, // hashcode -> ConnectionGene, hashcode is (from_node_innov * MAX_NODES + to_node_innov)
    population: Population,
}

impl Neat {
    pub fn new() -> Self {
        let new_node_pool =  {
            let mut hashmap:HashMap<usize, NodeGene> = HashMap::new();
            for i in 0..INPUTS {
                let input_node_gene = NodeGene::new(
                    NodeType::Input, 
                    i, 
                    0.0, 
                    1.0 - ((i + 1) as f64 / (INPUTS + BIAS as usize + 1) as f64));
                hashmap.insert(i, input_node_gene);
            }
            if BIAS {
                let bias_node_gene = NodeGene::new(NodeType::Bias, 
                    INPUTS, 
                    0.0, 
                    1.0 - ((INPUTS + 1) as f64 / (INPUTS + BIAS as usize + 1) as f64 ));
                hashmap.insert(INPUTS, bias_node_gene);
            }
            for i in 0..OUTPUTS {
                let output_node_gene = NodeGene::new(
                    NodeType::Output, 
                    INPUTS + BIAS as usize + i, 
                    1.0, 
                    1.0 - ((i + 1) as f64 / (OUTPUTS + 1) as f64));
                hashmap.insert(INPUTS + BIAS as usize + i, output_node_gene);
            }
            hashmap
        };
        Self {
            node_pool: new_node_pool.clone(),
            connection_pool: HashMap::new(),
            population: Population::new(&new_node_pool),
        }
    }
    pub fn new_fully_connected() -> Self {
        let new_node_pool =  {
            let mut hashmap:HashMap<usize, NodeGene> = HashMap::new();
            for i in 0..INPUTS {
                let input_node_gene = NodeGene::new(
                    NodeType::Input, 
                    i, 
                    0.0, 
                    1.0 - ((i + 1) as f64 / (INPUTS + BIAS as usize + 1) as f64));
                hashmap.insert(i, input_node_gene);
            }
            if BIAS {
                let bias_node_gene = NodeGene::new(NodeType::Bias, 
                    INPUTS, 
                    0.0, 
                    1.0 - ((INPUTS + 1) as f64 / (INPUTS + BIAS as usize + 1) as f64 ));
                hashmap.insert(INPUTS, bias_node_gene);
            }
            for i in 0..OUTPUTS {
                let output_node_gene = NodeGene::new(
                    NodeType::Output, 
                    INPUTS + BIAS as usize + i, 
                    1.0, 
                    1.0 - ((i + 1) as f64 / (OUTPUTS + 1) as f64));
                hashmap.insert(INPUTS + BIAS as usize + i, output_node_gene);
            }
            hashmap
        };
        let new_conn_pool = {
            let mut hashmap:HashMap<usize, ConnectionGene> = HashMap::new();
            let mut conn_inno = 0;
            for from_inno in 0..(INPUTS + BIAS as usize) {
                for to_inno in (INPUTS + BIAS as usize)..(INPUTS + BIAS as usize + OUTPUTS) {
                    hashmap.insert(conn_hashcode(from_inno, to_inno), ConnectionGene::new(conn_inno, from_inno, to_inno));
                    conn_inno += 1;
                }
            }
            hashmap
        };
        Self {
            node_pool: new_node_pool.clone(),
            connection_pool: new_conn_pool.clone(),
            population: Population::new_fully_connected(&new_node_pool, &new_conn_pool),
        }
    }

    // // Returns best fitness of new generation
    // pub fn run_generation(&mut self) -> f64 { 
    //     self.population.run_generation(xor)
    // }
    // pub fn run_generation(&mut self, env:for<'r> fn(&'r Genome) -> f64 ) -> f64 {
    //     self.reproduce();
    //     self.calculate_fitnesses(env);
    //     self.speciate();
    //     println!("{} species", self.species_vec.len());
    //     self.max_fitness
    // }


    pub fn print_pools(&self) {
        println!("NodePool\n{:?}", self.get_node_pool());
        println!("\n");
        println!("ConnPool\n{:?}", self.get_connection_pool());
    }

    pub fn train(&mut self, env:for<'r> fn(&'r Genome) -> f64, target_fitness:f64) -> Genome {
        self.population.calculate_fitnesses(env);
        let mut best_fitness = self.population.get_max_fitness();
        self.population.speciate();
        self.population.generation_info();

        while best_fitness <= target_fitness {

            let mut now = Instant::now();
            self.population.reproduce();
            // println!("reproduce() took {:.2?}", now.elapsed());

            now = Instant::now();
            mutate(self);
            // println!("mutate() took {:.2?}", now.elapsed());

            now = Instant::now();
            self.population.calculate_fitnesses(env);
            // println!("calc_fit() took {:.2?}", now.elapsed());

            now = Instant::now();
            self.population.speciate();
            // println!("speciate() took {:.2?}", now.elapsed());

            //self.population.print_species_rep();

            self.population.generation_info();

            best_fitness = self.population.get_max_fitness();
        }
        self.population.get_champion().unwrap().clone()
    }



    pub fn get_population(&self) -> &Population { &self.population }
    pub fn get_population_mut(&mut self) -> &mut Population { &mut self.population }
    pub fn get_node_pool(&self) -> &HashMap<usize, NodeGene> { &self.node_pool }
    pub fn get_node_pool_mut(&mut self) -> &mut HashMap<usize, NodeGene> { &mut self.node_pool }
    pub fn get_connection_pool(&self) -> &HashMap<usize, ConnectionGene> { &self.connection_pool}
    pub fn get_connection_pool_mut(&mut self) -> &mut HashMap<usize, ConnectionGene> { &mut self.connection_pool }

    //=============================DEBUGGING===============================//

    #[allow(dead_code)]
    pub fn new_xor() -> Self {
        
        let mut node_pool:HashMap<usize, NodeGene> = HashMap::new();
        node_pool.insert(0, NodeGene::new(crate::NodeType::Input, 0, 0.0, 0.75));
        node_pool.insert(1, NodeGene::new(crate::NodeType::Input, 1, 0.0, 0.5));
        node_pool.insert(2, NodeGene::new(crate::NodeType::Bias, 2, 0.0, 0.1));
        node_pool.insert(3, NodeGene::new(crate::NodeType::Output, 3, 1.0, 0.5));
        node_pool.insert(4, NodeGene::new(crate::NodeType::Hidden, 4, 0.5, 0.75));
        node_pool.insert(5, NodeGene::new(crate::NodeType::Hidden, 5, 0.5, 0.5));

        let mut conn_pool:HashMap<usize, ConnectionGene> = HashMap::new();
        conn_pool.insert(conn_hashcode(0, 4), ConnectionGene::new_explicit(0, 20.0, true, 0, 4));
        conn_pool.insert(conn_hashcode(0, 5), ConnectionGene::new_explicit(1, -20.0, true, 0, 5));
        conn_pool.insert( conn_hashcode(1, 4), ConnectionGene::new_explicit(2, 20.0, true, 1, 4));
        conn_pool.insert( conn_hashcode(1, 5), ConnectionGene::new_explicit(3, -20.0, true, 1, 5));

        conn_pool.insert( conn_hashcode(4, 3), ConnectionGene::new_explicit(4, 20.0, true, 4, 3));
        conn_pool.insert( conn_hashcode(5, 3), ConnectionGene::new_explicit(5, 20.0, true, 5, 3));

        conn_pool.insert( conn_hashcode(2, 4), ConnectionGene::new_explicit(6, 10.0, true, 2, 4));
        conn_pool.insert( conn_hashcode(2, 5), ConnectionGene::new_explicit(7, -30.0, true, 2, 5));
        conn_pool.insert( conn_hashcode(2, 3), ConnectionGene::new_explicit(8, 30.0, true, 2, 3));
        Self {
            node_pool: node_pool.clone(),
            connection_pool: conn_pool,
            population: Population::new(&node_pool),
        }
    }  
}

// #[cfg(test)]
// mod tests {
//     use std::collections::HashMap;
//     use crate::population::Population;
//     use crate::genetics::{NodeGene, NodeType}; 
//     use crate::config::{INPUTS, BIAS, OUTPUTS, POPULATION_SIZE};
//     use crate::test_environments::xor;

    // #[test]
    // fn xor_train_weights() {
    //     let node_gene_pool = NodeGene::xor_node_gene_pool();
    //     let mut population = Population::new_xor();
    //     population.calculate_fitnesses(xor);
    //     let mut best_fitness = population.get_max_fitness();
    //     while best_fitness < 15.9 {
    //         best_fitness = population.run_generation(xor);
    //         println!("fitness = {best_fitness}");
    //     }
    //     assert!(true);
    // }

// }