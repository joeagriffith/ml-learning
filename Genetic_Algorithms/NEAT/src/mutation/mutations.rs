use rand::prelude::*;
use crate::genetics::{NodeGene, NodeType, ConnectionGene, conn_hashcode, Gene};
use crate::config::{MAX_ITER, INPUTS, BIAS, OUTPUTS, WEIGHT_RANDOM_MAX, WEIGHT_SHIFT_MAX_PCT};
use crate::Neat;

/// Mutates a new node into a genome by splitting a random existing connection into two new connections with a new node in between.
/// Fails if and only if the genome does not have any connections
/// Checks the node_pool to check whether the new node has already been evolved by other genomes, and if so uses this node.
pub fn mutate_new_node(neat:&mut Neat, g_id:usize) -> bool {
    
    if neat.get_population().organisms.read().unwrap()[g_id].get_connections().len() == 0 {
        return false
    }

    let mut rand_conn_innov = neat.get_population().organisms.read().unwrap()[g_id].get_connections().rand_innov();
    let mut from = neat.get_population().organisms.read().unwrap()[g_id].get_connections().get_by_innov(rand_conn_innov).get_from();
    let mut to = neat.get_population().organisms.read().unwrap()[g_id].get_connections().get_by_innov(rand_conn_innov).get_to();

    let mut iter = 0;
    while !neat.get_connection_pool().get(&conn_hashcode(from, to)).unwrap().is_enabled() || neat.get_population().organisms.read().unwrap()[g_id].get_nodes().get_by_innov(from).get_nodetype() == NodeType::Bias {
        iter += 1;
        if iter > MAX_ITER {
            return false
        }
        rand_conn_innov = neat.get_population().organisms.read().unwrap()[g_id].get_connections().rand_innov();
        from = neat.get_population().organisms.read().unwrap()[g_id].get_connections().get_by_innov(rand_conn_innov).get_from();
        to = neat.get_population().organisms.read().unwrap()[g_id].get_connections().get_by_innov(rand_conn_innov).get_to();
    }

    let weight = neat.get_population().organisms.read().unwrap()[g_id].get_connections().get_by_innov(rand_conn_innov).get_weight();
    let from_x = neat.get_population().organisms.read().unwrap()[g_id].get_nodes().get_by_innov(from).get_x();
    let from_y = neat.get_population().organisms.read().unwrap()[g_id].get_nodes().get_by_innov(from).get_y();
    let to_x = neat.get_population().organisms.read().unwrap()[g_id].get_nodes().get_by_innov(to).get_x();
    let to_y = neat.get_population().organisms.read().unwrap()[g_id].get_nodes().get_by_innov(to).get_y();
    
    let new_node_guid = INPUTS + BIAS as usize + OUTPUTS + rand_conn_innov;
    let length = neat.get_node_pool().len();
    let new_node = neat.get_node_pool_mut().entry(new_node_guid)
        .or_insert(NodeGene::new(
            NodeType::Hidden, 
            length,
            (from_x + to_x) / 2.0,
            (from_y + to_y) / 2.0,
       ))
        .clone();
    let new_node_innov = new_node.get_innov();

    let new_conn_1_guid = conn_hashcode(from, new_node_innov);
    let new_conn_2_guid = conn_hashcode(new_node_innov, to);

    let conn_gene_pool_len = neat.get_connection_pool().len();
    let mut new_conn_1 = neat.get_connection_pool_mut().entry(new_conn_1_guid)
        .or_insert(ConnectionGene::new(conn_gene_pool_len, from, new_node_innov))
        .clone();
    new_conn_1.set_weight(1.0);

    let conn_gene_pool_len = neat.get_connection_pool().len();
    let mut new_conn_2 = neat.get_connection_pool_mut().entry(new_conn_2_guid)
        .or_insert(ConnectionGene::new(conn_gene_pool_len, new_node_innov, to))
        .clone();
    new_conn_2.set_weight(weight);

    let genome = &mut neat.get_population().organisms.write().unwrap()[g_id];
    genome.get_connections_mut().remove(rand_conn_innov);
    genome.get_connections_mut().insert_sorted(new_conn_1);
    genome.get_connections_mut().insert_sorted(new_conn_2);
    genome.get_nodes_mut().insert_sorted(new_node);
    
    true
}

//dont mutate link when from => ... => to already exists
/// Mutates a new connection in a genome by selecting two random nodes.
/// Ensures that these nodes are not already connected, even by intermediaries, to prevent duplicates.
/// Will attempt to find a suitable connection MAX_ITER times before failing.
pub fn mutate_new_conn(neat:&mut Neat, g_id:usize) -> bool {

    // let mut rand_conn_innov = neat.get_population().organisms.read().unwrap()[g_id].get_connections().rand_innov();
    // let mut from = neat.get_population().organisms.read().unwrap()[g_id].get_connections().get_by_innov(rand_conn_innov).get_from();
    // let mut to = neat.get_population().organisms.read().unwrap()[g_id].get_connections().get_by_innov(rand_conn_innov).get_to();
    //select rand from_node in genome
    // select rand to_node in genome
    //if  they are already connected (directly or indirectly), then choose again until they arent or until iter > MAX_ITER

    for _i in 0..MAX_ITER{
        let mut from_innov = neat.get_population().organisms.read().unwrap()[g_id].get_nodes().rand_innov();
        let mut to_innov = neat.get_population().organisms.read().unwrap()[g_id].get_nodes().rand_innov();
        let from_x = neat.get_population().organisms.read().unwrap()[g_id].get_nodes().get_by_innov(from_innov).get_x();
        let to_x = neat.get_population().organisms.read().unwrap()[g_id].get_nodes().get_by_innov(to_innov).get_x();
        if from_x == to_x {
            continue;
        } else if from_x > to_x {
            let temp = from_innov;
            from_innov = to_innov;
            to_innov = temp;
        }

        if neat.get_population().organisms.read().unwrap()[g_id].is_connected(from_innov, to_innov) {
            continue;
        }

        let length = neat.get_connection_pool().len();
        let new_connection = neat.get_connection_pool_mut().entry(conn_hashcode(from_innov, to_innov))
            .or_insert(ConnectionGene::new(length, from_innov, to_innov))
            .clone();
        
        neat.get_population().organisms.write().unwrap()[g_id].get_connections_mut().insert_sorted(new_connection);
        return true
    }
    false
}

/// Toggle the 'enabled' boolean of a random connection in the genome.
pub fn mutate_conn_enabled(neat:&mut Neat, g_id:usize) {
    let genome = &mut neat.get_population().organisms.write().unwrap()[g_id];
    if genome.get_connections().len() > 0 {
        let rand_connection = genome.get_connections_mut().rand_element_mut();
        rand_connection.set_enabled(!rand_connection.is_enabled());
    }
}

/// Mutates the weight of a random connection in a genome to between the range +-WEIGHT_RANDOM_MAX
pub fn mutate_weight_random(neat:&mut Neat, g_id:usize) {
    let genome = &mut neat.get_population().organisms.write().unwrap()[g_id];
    if genome.get_connections().len() > 0 {
        let rand_connection = genome.get_connections_mut().rand_element_mut();
        rand_connection.set_weight(rand::thread_rng().gen_range(-1.0..1.0) * WEIGHT_RANDOM_MAX);
    }
}

/// Mutates the weight of a random connection in a genome by a shift value in range +- WEIGHT_SHIFT_MAX_PCT
pub fn mutate_weight_shift(neat:&mut Neat, g_id:usize) {
    let genome = &mut neat.get_population().organisms.write().unwrap()[g_id];
    if genome.get_connections().len() > 0 {
        let rand_connection = genome.get_connections_mut().rand_element_mut();
        rand_connection.set_weight(rand_connection.get_weight() * (1.0 + WEIGHT_SHIFT_MAX_PCT * rand::thread_rng().gen_range(-1..1) as f64));
    }
}








#[cfg(test)]
mod tests {
    use crate::Neat;
    use crate::genetics::{NodeGene, NodeType, ConnectionGene, Genome, conn_hashcode};
    use std::collections::HashMap;

    use super::{mutate_new_node, mutate_new_conn, mutate_conn_enabled};

    // #[test]
    // pub fn test_mutate_new_node() {
    //     let mut neat = Neat::new(); 
    //     *neat.get_node_pool_mut() = HashMap::new();
    //     *neat.get_connection_pool_mut() = HashMap::new();
    //     neat.get_node_pool_mut().insert(0, NodeGene::new(NodeType::Input, 0, 0.0, 0.0));
    //     neat.get_node_pool_mut().insert(1, NodeGene::new(NodeType::Output, 1, 1.0, 0.0));
    //     neat.get_connection_pool_mut().insert(conn_hashcode(0, 1), ConnectionGene::new_explicit(0, 0.5, true, 0, 1));

    //     let mut genome = Genome::new();
    //     genome.get_nodes_mut().push(neat.get_node_pool().get(&0).unwrap().clone());
    //     genome.get_nodes_mut().push(neat.get_node_pool().get(&1).unwrap().clone());
    //     genome.get_connections_mut().push(neat.get_connection_pool().get(&conn_hashcode(0, 1)).unwrap().clone());

    //     neat.get_population_mut().get_organisms_mut()[0] = genome;
    //     mutate_new_node(&mut neat, 0);

    //     mutate_new_node(&mut neat, 0);
    //     assert!(neat.get_node_pool().len() == 4);
    //     assert!(neat.get_connection_pool().len() == 5);
    //     assert!(neat.get_population().get_organisms()[0].get_nodes().len() == 4);
    //     assert!(neat.get_population().get_organisms()[0].get_connections().len() == 3);
    // }

    // #[test]
    // fn test_mutate_new_link() {
    //     let mut neat = Neat::new_xor();
        
    //     //ful xor genome
    //     let mut genome = Genome::new_init_xor();
    //     //remove one connection to make incomplete xor
    //     genome.get_connections_mut().remove(3);
    //     //replace genome0 in population with incomplete xor genome;
    //     *neat.get_population_mut().get_genome_mut(0) = genome;
    //     //attempt to mutate the link we removed, it is the only possible link that can mutate
    //     mutate_new_conn(&mut neat, 0);
    //     assert!(neat.get_connection_pool().len() == 9);
    //     assert!(neat.get_node_pool().len() == 6);
    //     assert!(neat.get_population().get_genome(0).get_nodes().len() == 6);
    //     assert!(neat.get_population().get_genome(0).get_connections().len() == 9);
    //     assert!(neat.get_population().get_genome(0).get_connections().contains_innov(3));
    // }

    // // #[test]
    // // fn test_mutate_weight_rand() {
    // //     let mut neat = Neat::new_xor();
    // //     mutate_new_conn(&mut neat, 0);
    // //     let weight_before = neat.get_population().get_genome(0).get_connections().get(0).get_weight();
    // //     mutate_weight_random(&mut neat, 0);
    // //     let weight_after = neat.get_population().get_genome(0).get_connections().get(0).get_weight();
    // //     assert!(weight_before != weight_after);
    // // }

    // // #[test]
    // // fn test_mutate_weight_shift() {
    // //     let mut neat = Neat::new_xor();
    // //     mutate_new_conn(&mut neat, 0);
    // //     let weight_before = neat.get_population().get_genome(0).get_connections().get(0).get_weight();
    // //     mutate_weight_shift(&mut neat, 0);
    // //     let weight_after = neat.get_population().get_genome(0).get_connections().get(0).get_weight();
    // //     assert!(weight_before != weight_after);
    // // }

    // #[test]
    // fn test_mutate_link_enabled() {
    //     let mut neat = Neat::new_xor();
    //     mutate_new_conn(&mut neat, 0);
    //     mutate_conn_enabled(&mut neat, 0);
    //     assert!(!neat.get_population().get_genome(0).get_connections().get(0).is_enabled());
    // }
}