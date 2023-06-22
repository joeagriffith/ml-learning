use crate::Neat;
use rand::prelude::*;
use crate::config::{POPULATION_SIZE, M_WEIGHT_SHIFT, M_WEIGHT_RANDOM, M_CONN_ENABLED, M_NODE, M_CONN};
use super::mutations::{mutate_weight_shift, mutate_weight_random, mutate_conn_enabled, mutate_new_node, mutate_new_conn};

pub fn mutate(neat:&mut Neat) {
    let mut rand = rand::thread_rng();

    for g_id in 0..POPULATION_SIZE {
        if neat.get_population().should_mutate(g_id) {
            if rand.gen_range(0.0..1.0) < M_WEIGHT_SHIFT {
                mutate_weight_shift(neat, g_id);
            }
            if rand.gen_range(0.0..1.0) < M_WEIGHT_RANDOM {
                mutate_weight_random(neat, g_id);
            }
            if rand.gen_range(0.0..1.0) < M_CONN_ENABLED {
                mutate_conn_enabled(neat, g_id);
            }
            if rand.gen_range(0.0..1.0) < M_NODE {
                mutate_new_node(neat, g_id);
            }
            if rand.gen_range(0.0..1.0) < M_CONN {
                mutate_new_conn(neat, g_id);
            }
        }
    }
}