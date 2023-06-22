mod genome;
mod connection_gene;
mod node_gene;
mod gene;
mod util;

pub use genome::Genome;
pub use node_gene::{NodeGene, NodeType};
pub use connection_gene::ConnectionGene;
pub use gene::Gene;
pub use util::{distance, crossover, conn_hashcode};