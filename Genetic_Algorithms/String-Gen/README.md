# simple-genetic-algo

## Description
This command line program uses a genetic algorithm to evolve a population of random strings until a champion reaches optimal fitness. Fitness is defined by similarity to a user-specified string.

Reccomended Hyperparameters:
    - population_size: 100
    - mutation_rate: 0.1

## How To Run
1. Clone this repository:
```sh
$ git clone https://gitlab.com/joeagriffith/simple-genetic-algo.git
```
2. Compile:
```sh
$ cargo build --release
```

3. Run:
```sh
$ ./target/release/genetic-nn.exe {target_string} {population_size[Int]} {mutation_rate[float]}
```