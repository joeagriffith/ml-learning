Customisations I made to the algorithm

modified how genomes are speciated,
    - Original algorithm says to compare genomes against species representative until distance < compat_thresh
    - My algorithm checks every species and speciates the genome into the species with the minimum distance, only if distance < compat_thresh

nodegenes & connectiongenes have an extra identifier
    - To reduce duplication in the gene_pools and recognise similarity between genomes containing duplicate nodes/connectiongenes
    I encorporated a guid which is used as the key in the gene_pool hashmaps
    - the key is built during the mutation stepp
    // Todo Explain this better

No more mutating connection genes to nodes connected via intermediaries
    - i.e. in case (0) --> (1) --> (2) original algorithm could evolve connections between node 0 and node 2
    - I have decided this is pointless as any influence an extra connection from node 0  to 2 would have, could be captured in the weights from 0 -> 1 and 1 -> 2

Only mutate new node/link when species stagnates
    - when a species stagnates give x% of the population a new node, and (100-x)% of the population a new connection





new algorithm? focused around topological speciation. 

Topologically speciated
    - speciating by topology only  (not weights)
    - genomes are only in same species if they have identical topology
    - only mutate topology of genomes in a species when their species becomes stagnant
    - can build a graph of all species discovered which traces their paths
    - mark a species extinct if it has made X topological mutations without significant fitness increase
    - all genomes in extinct species are reset and begin evolving from start again
    - 
