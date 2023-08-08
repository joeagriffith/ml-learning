use std::time::Duration;
use crate::{Spike, SpikeTrain, SpikeGenerator};
use factorial::Factorial;
use rand::prelude::*;
use rand::distributions::Uniform;



pub struct HomogeneousPoissonSpikeGenerator {
    firing_rate: f64,
}

impl SpikeGenerator for HomogeneousPoissonSpikeGenerator {
    // Generates a spike train where each spike is independent of another, and firing rate is constant.
    fn generate_spike_train(&mut self, period:Duration) -> SpikeTrain {
        let mut spike_train = SpikeTrain::new();
        let range = Uniform::<f64>::from(0. .. 1.);
        let mut rng = thread_rng();

        let mut time_elapsed = Duration::ZERO;
        time_elapsed += self.float_to_duration(range.sample(&mut rng).ln() / -self.firing_rate);
        while time_elapsed < period {
            spike_train.add_spike(Spike::new(time_elapsed));
            time_elapsed += self.float_to_duration(range.sample(&mut rng).ln() / -self.firing_rate);
        }

        spike_train
    }
}

impl HomogeneousPoissonSpikeGenerator {
    // firing_rate given in terms of milliseconds
    pub fn new(firing_rate: f64) -> Self {
        Self {
            firing_rate,
        }
    }

    #[allow(dead_code)]
    // Converts Duration into milliseconds_f64
    fn duration_to_float(&self, dur:Duration) -> f64 {
        dur.as_nanos() as f64 / 1000000.0
    }

    // Converts milliseconds_f64 into Duration
    fn float_to_duration(&self, mut float:f64) -> Duration {
        float = float * 1000000.0;
        Duration::from_nanos(float as u64)
    }

    #[allow(dead_code, non_snake_case)]
    // calculates the probability that n spikes occur in a period of T
    fn p_of_n_spikes(&self, n:usize, period:Duration) -> f64 {
        let n_f64 = n as f64;
        let rT = self.firing_rate * self.duration_to_float(period);
        (rT.powf(n_f64) / n.factorial() as f64) * (-rT).exp()
    }

    
}



