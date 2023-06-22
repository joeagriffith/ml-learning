mod mutations;
mod mutate;

pub use mutations::{
    mutate_new_node, 
    mutate_new_conn, 
    mutate_conn_enabled, 
    mutate_weight_random, 
    mutate_weight_shift
};

pub use mutate::{mutate};