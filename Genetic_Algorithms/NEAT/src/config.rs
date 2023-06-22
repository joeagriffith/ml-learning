pub const INPUTS:usize = 3;
pub const OUTPUTS:usize = 3;
pub const BIAS:bool = true;
pub const ELITISM:usize = 2;
pub const POPULATION_SIZE:usize = 1000;
pub const NUM_THREADS:usize = 4;

pub const MAX_NODES:usize = (2 as usize).pow(20); // for calculating connection_gene hashcodes

// Speciation hyperparameters
pub const TARGET_SPECIES_NUM:usize = 50; // The ideal amount of species we would like to have at any given generation
pub const COMPATABILITY_THRESHOLD:f64 = 8.0;
pub const COMPATABILITY_MIN:f64 = 0.005;
pub const COMPATABILITY_MODIFIER:f64 = 0.5; // The amount by which we modify COMPATABILITY_THRESHOLD each generation, whether we are above or below the TARGET_SPECIES_NUM
pub const DROPOFF_AGE:usize =50; // If a species' fitness does not improve in this many generations, it will be culled.
pub const C1:f64 = 1.0;
pub const C2:f64 = 1.0;
pub const C3:f64 = 0.4;

pub const WEIGHT_RANDOM_MAX:f64 = 5.0; // the absolute maximum value for connection weights
pub const WEIGHT_SHIFT_MAX_PCT:f64 = 0.1; // The percentage amount by which mutate_weight_shift() modifies connection weights.

// Probabilities for each mutation function
// pub const M_TOPOLOGY_NODE:f64 = 0.22; // when stagnated, X chance to mutate new node, and 1-X chance to mutate new link (will mutate new node if new link fails)
pub const M_CONN_ENABLED:f64 = 0.00; // mutate_link_enabled()
pub const M_WEIGHT_RANDOM:f64 = 0.4; // mutate_weight_random()
pub const M_WEIGHT_SHIFT:f64 = 0.8; // mutate_weight_shift()
pub const M_NODE:f64 = 0.02; // mutate_new_node()
pub const M_CONN:f64 = 0.1; // mutate_new_connection()
pub const MAX_ITER:usize = 100; // Max iterations for functions which loop until valid value