use rand::prelude::*;

use crate::config::WEIGHT_RANDOM_MAX;
use super::gene::Gene;
use super::conn_hashcode;

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    innovation_number: usize,
    weight: f64,
    enabled: bool,
    from: usize, // from_node_guid
    to: usize, //to_node_guid
}

impl Gene for ConnectionGene {
    fn get_innov(&self) -> usize {self.innovation_number}
}

impl ConnectionGene {

    /// Constructs a new ConnectionGene from a specified node to a specified node
    /// weight is random between -WEIGHT_RANDOM_MAX to WEIGHT_RANDOM_MAX
    /// enabled: true
    pub fn new(innovation_number:usize, from:usize, to:usize) -> Self {
        Self {
            innovation_number,
            weight: rand::thread_rng().gen_range(-1.0..1.0) * WEIGHT_RANDOM_MAX,
            enabled: true,
            from,
            to,        
        }
    }

    /// Constructs a ConnectionGene where every field is specified.
    pub fn new_explicit(innovation_number:usize, weight: f64, enabled: bool, from:usize, to:usize) -> Self {
        Self {
            innovation_number,
            weight,
            enabled,
            from,
            to,
        }
    }

    /// Returns a unique value dependent on self's 'from' and 'to' node ids.
    pub fn hashcode(&self) -> usize {
        conn_hashcode(self.from, self.to)
    }

    //=================================GETTERS & SETTERS=======================================//

    pub fn get_weight(&self) -> f64 { self.weight }
    pub fn is_enabled(&self) -> bool { self.enabled }
    pub fn get_from(&self) -> usize { self.from }
    pub fn get_to(&self) -> usize { self.to }
    
    pub fn set_weight(&mut self, weight: f64) { self.weight = weight; }
    pub fn set_enabled(&mut self, enabled: bool) { self.enabled = enabled; }

    //=====================================DEBUGGING============================================//

    #[allow(dead_code)]
    pub fn print(&self) {
        println!("conn {}: Node {} -> Node {}    weight: {}    enabled: {}", 
                self.get_innov(), 
                self.get_from(), 
                self.get_to(),
                self.get_weight(), 
                self.is_enabled()
            );
    }

}

