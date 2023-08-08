use neural_sim::prelude::*;
use std::time::Duration;


pub fn main() {
    //let mut spike_generator = HomogeneousPoissonSpikeGenerator::new(2.0);
    let mut spike_generator = InhomogeneousPoissonSpikeGenerator::new(|t| (t*0.05).powf(1.1), Some(Duration::from_micros(100)));
    let mut spike_trains = Vec::new();

    for _ in 0..50000 {
        spike_trains.push(spike_generator.generate_spike_train(Duration::from_millis(20)));
    }
    let fano = calculate_fano_factor(spike_trains);
    println!("Fano Factor: {:.4}", fano);
}



// Builds 'iters' spike trains and finds the spike count for each, incrementing the relavent bin.
// Returns the distribution of spike counts from 'iters' iterations.
fn sample_num_spikes_generated(iters:usize) -> [usize;100] {
    let mut ns = [0;100];
    // let spike_generator = HomogeneousPoissonSpikeGenerator::new(2.0);
    let mut spike_generator = InhomogeneousPoissonSpikeGenerator::new(|x| x * 0.1, Some(Duration::from_micros(10)));
    for _ in 0..iters {
        let spike_train = spike_generator.generate_spike_train(Duration::from_millis(20));
        let length = spike_train.len();
        if length < 100 {
            ns[spike_train.len()] += 1;
        }
    }
    ns
}