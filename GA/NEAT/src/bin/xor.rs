use neat_from_scratch::Neat;
use neat_from_scratch::test_environments::xor;



/*
numbers are node_ids, connections always from left to right.

(0)------(4)
    \ /  /  \   
     X (2)---(3)
    / \  \  /
(1)------(5)

*/



fn main() {
    let mut neat = Neat::new_fully_connected();
    let xor_genome = neat.train(xor, 15.9);
    println!("great success!");
    xor_genome.print();
}