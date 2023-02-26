# Training Deep Neural Nets Using Genetic Programming

## Description
This project is composed of 3 constituent parts:
    - Matrix implementation built ontop of Rust's vector system.
    - Neural network library using this matrix implementation.
    - Genetic algorithm using the crossover and mutation in a population.

The Algorithm trains the weights of a pre-specified neural net architecture for an XOR classification problem.
This project is a precurssor and proof of concept to my 'neat-from-scratch'.

reccomended hyperparameters:
    - population_size: 100
    - mutation_rate: 0.1

If population_size is too large, a random net will likely be able to classify XOR in the first generation.


## How To Run
1. Clone this repository:
```sh
$ git clone https://gitlab.com/joeagriffith/neat-rust.git
```
2. Compile:
```sh
$ cargo build --release
```

3. Run:
```sh
$ ./target/release/genetic-nn.exe {population_size[int]} {mutation_rate[float]}
```