# Training Deep Neural Nets Using Genetic Programming

## Description
This library is meant to provide tools for modelling concepts learnt from the book 'Theoretical Neuroscience' - Peter Dayan and L.F. Abbott

Currently this library implements both Inhomogeneous & Homogeneous Poisson spike generators.
These can both be used to generate spike train from the user specified arguments of firing rate, 'r' or 'r(t)', and period 'T'.

Some analysis of the distributions of model outputs can be found in the docs folder.

## How To Run
1. Clone this repository:
```sh
$ git clone https://gitlab.com/joeagriffith/neural-sim.git
```

2. Modify run.rs in binaries folder.

3. Compile:
```sh
$ cargo build --release
```

4. Run:
```sh
$ ./target/release/run.exe
```