use super::{Genome, Gene};
use crate::config::{C1, C2, C3, MAX_NODES};
use rand::prelude::*;


/// Returns a value for the similarity between two genomes based on their nodes, connections and weights.
/// This value is used in determining which species a genome belongs to.
pub fn distance(g1:&Genome, g2:&Genome) -> f64 {

    let mut num_disjoint:usize = 0;
    let mut num_excess:usize = 0;
    let mut num_similar:usize = 0;
    let mut avg_weight_diff:f64 = 0.0;
    let mut max_num_enabled;

    let mut g1_iter = g1.get_connections().iter();
    let mut g2_iter = g2.get_connections().iter();

    // Ensure g1 has the largest innovation number
    if g2.get_connections().max_innov() > g1.get_connections().max_innov() {
        g1_iter = g2.get_connections().iter();
        g2_iter = g1.get_connections().iter();
    }

    let mut g1_num_enabled:usize = 0;
    let mut g2_num_enabled:usize = 0;

    // Calculate number of similar genes, disjoint genes & avg_weight_diff
    let mut g1_curr = g1_iter.next();
    let mut g2_curr = g2_iter.next();
    while g1_curr.is_some() && g2_curr.is_some() {

        if !g1_curr.unwrap().is_enabled() {
            g1_curr = g1_iter.next();
            continue;
        }
        if !g2_curr.unwrap().is_enabled() {
            g2_curr = g2_iter.next();
            continue;
        }

        let g1_innov = g1_curr.unwrap().get_innov();
        let g2_innov = g2_curr.unwrap().get_innov();

        if g1_innov == g2_innov {
            num_similar += 1;
            avg_weight_diff += (g1_curr.unwrap().get_weight() - g2_curr.unwrap().get_weight()).abs();
            g1_curr = g1_iter.next();
            g2_curr = g2_iter.next();
            g1_num_enabled += 1;
            g2_num_enabled += 1;
        } else if g1_innov > g2_innov {
            num_disjoint += 1;
            g2_curr = g2_iter.next();
            g2_num_enabled += 1;
        } else if g1_innov < g2_innov {
            num_disjoint += 1;
            g1_curr = g1_iter.next();
            g1_num_enabled += 1;
        }
    }
    if num_similar > 0 {
        avg_weight_diff /= num_similar as f64;
    }

    // Calculate number of excess connection genes in longer genome
    while g1_curr.is_some() {
        if g1_curr.unwrap().is_enabled() {
            num_excess += 1;
            g1_num_enabled += 1;
        }
        g1_curr = g1_iter.next();
    }
    
    if g1_num_enabled > g2_num_enabled {
        max_num_enabled = g1_num_enabled;
    } else {
        max_num_enabled = g2_num_enabled;
    }
    if max_num_enabled == 0 {
        max_num_enabled = 1;
    }

    (C1 * num_excess as f64 / max_num_enabled as f64) 
    +
    (C2 * num_disjoint as f64 / max_num_enabled as f64)
    + 
    (C3 * avg_weight_diff)
}

/// Creates a new Genome from two parent genomes.
/// Uses a zipper-like method of selecting genes from parents. 
pub fn crossover(g1:&Genome, g2:&Genome, mut g1_fitter_than_g2: bool) -> Genome {
    
    let mut g1_iter = g1.get_connections().iter();
    let mut g2_iter = g2.get_connections().iter();
    let mut child = Genome::new();
    let mut rng = rand::thread_rng();

    if g1_fitter_than_g2 {
        *child.get_nodes_mut() = g1.get_nodes().clone();
    } else {
        *child.get_nodes_mut() = g2.get_nodes().clone();
    }

    let child_connections = child.get_connections_mut();

    // Ensure g1 has the larger innovation number
    if g2.get_connections().max_innov() > g1.get_connections().max_innov() {
        g1_iter = g2.get_connections().iter();
        g2_iter = g1.get_connections().iter();
        g1_fitter_than_g2 = !g1_fitter_than_g2; // Flip as we are swapping g1 and g2.
    }

    let mut g1_curr = g1_iter.next();
    let mut g2_curr = g2_iter.next();
    while g1_curr.is_some() && g2_curr.is_some() {

        let g1_innov = g1_curr.unwrap().get_innov();
        let g2_innov = g2_curr.unwrap().get_innov();

        if g1_innov == g2_innov {
            if rng.gen::<bool>() {
                child_connections.push(g1_curr.unwrap().clone());
            } else {
                child_connections.push(g2_curr.unwrap().clone());
            }
            g1_curr = g1_iter.next();
            g2_curr = g2_iter.next();
        } else if g1_innov > g2_innov {
            if !g1_fitter_than_g2 { // Only include dijoint genes of the fitter parent
                child_connections.push(g2_curr.unwrap().clone());
            }
            g2_curr = g2_iter.next();
        } else if g1_innov < g2_innov {
            if g1_fitter_than_g2 { // Only include disjoint genes of the fitter parent
                child_connections.push(g1_curr.unwrap().clone());
            }
            g1_curr = g1_iter.next();
        }
    }

    // Iterate over excess genes
    if g1_fitter_than_g2 { // Only include excess genes if longer gene is fitter than shorter gene
        while g1_curr.is_some() {
            child_connections.push(g1_curr.unwrap().clone());
            g1_curr = g1_iter.next();
        }
    }

    child
}

// gives each connection a unique identifier based on the innovation numbers of its 'from' and 'to' nodes.
pub fn conn_hashcode(from_node_inno:usize, to_node_inno:usize) -> usize {
    from_node_inno * MAX_NODES + to_node_inno + 1
}

#[cfg(test)]
mod tests {
    use crate::{Neat, test_environments::xor};

    use super::super::node_gene::{NodeGene, NodeType};
    use super::super::connection_gene::ConnectionGene;
    use super::Genome;
    use super::crossover;

    #[test] 
    fn crossover_xor() {
        let neat = Neat::new();

        // Complete XOR genome
        let genome1 = Genome::new_init_xor();

        // Incomplete XOR genome
        let mut genome2 = Genome::new_init(neat.get_node_pool());
        genome2.get_nodes_mut().push(NodeGene::new(NodeType::Hidden, 4, 0.5, 0.75));

        //Layer 0
        genome2.get_connections_mut().push(ConnectionGene::new_explicit(0, 20.0, true, 0, 4));
        genome2.get_connections_mut().push(ConnectionGene::new_explicit(2, 20.0, true, 1, 4));

        //Layer 1
        genome2.get_connections_mut().push(ConnectionGene::new_explicit(4, 20.0, true, 4, 3));
    
        //Biases
        genome2.get_connections_mut().push(ConnectionGene::new_explicit(6, 10.0, true, 2, 4));

        let child = crossover(&genome1, &genome2, true);
        assert!(xor(&child) > 15.9);
    }
}