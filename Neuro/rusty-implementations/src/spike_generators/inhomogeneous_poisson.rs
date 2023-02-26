use std::time::Duration;
use crate::{Spike, SpikeTrain, SpikeGenerator};
// use factorial::Factorial;
use rand::prelude::*;
use rand::distributions::Uniform;



pub struct InhomogeneousPoissonSpikeGenerator {
    firing_rate: fn(f64) -> f64,
    delta_time: Option<Duration>,
    max_firing_rate: f64,
    max_period: Duration,
}

impl SpikeGenerator for InhomogeneousPoissonSpikeGenerator {
    // Generates a spike train where each spike is independent of another, and firing rate is constant.
    fn generate_spike_train(&mut self, period:Duration) -> SpikeTrain {
        let mut spike_train = SpikeTrain::new();
        let range = Uniform::<f64>::from(0. .. 1.);
        let mut rng = thread_rng();

        // max_period is the period, 0 <= t < period, over which max_firing_rate has been maximised.
        if period > self.max_period {
            self.calculate_max_firing_rate(period);
        }

        let mut time_elapsed = Duration::ZERO;
        time_elapsed += self.float_to_duration(range.sample(&mut rng).ln() / -self.max_firing_rate);
        while time_elapsed < period {
            spike_train.add_spike(Spike::new(time_elapsed));
            time_elapsed += self.float_to_duration(range.sample(&mut rng).ln() / -self.max_firing_rate);
        }

        let mut i = 0;
        while i < spike_train.len() {
            let t_i = self.duration_to_float(spike_train.get_spike(i).unwrap().get_time().clone());
            let r_t_i = (self.firing_rate)(t_i);
            if r_t_i / self.max_firing_rate < range.sample(&mut rng) {
                spike_train.remove_spike(i);
            } else {
                i += 1;
            }
        }

        spike_train
    }
}

impl InhomogeneousPoissonSpikeGenerator {
    // firing_rate given in terms of milliseconds
    // delta_time is the increment by which we search for max firing rate. recommend 10 microseconds
    pub fn new(firing_rate: fn(f64) -> f64, delta_time:Option<Duration>) -> Self {
        Self {
            firing_rate,
            delta_time,
            max_firing_rate: 0.0,
            max_period: Duration::from_micros(0),
        }
    }

    // Calculated by binning the function into microseconds over the entire period
    fn calculate_max_firing_rate(&mut self, period:Duration) {
        let period_f64 = period.as_micros() as f64 / 1000.0;
        let mut d_time = 0.01;
        if self.delta_time.is_some() {
            d_time = self.delta_time.unwrap().as_micros() as f64 / 1000.0;
        }

        let mut time = self.max_period.as_micros() as f64 / 1000.0;
        while time < period_f64 {
            let firing_rate = (self.firing_rate)(time);
            if firing_rate > self.max_firing_rate {
                self.max_firing_rate = firing_rate;
            }
            time += d_time;
        }

        self.max_period = period;
    }

    // Converts Duration into milliseconds_f64
    fn duration_to_float(&self, dur:Duration) -> f64 {
        dur.as_nanos() as f64 / 1000000.0
    }

    // Converts milliseconds_f64 into Duration
    fn float_to_duration(&self, mut float:f64) -> Duration {
        float = float * 1000000.0;
        Duration::from_nanos(float as u64)
    }

    // calculates the probability that n spikes occur in a period of T
    // fn p_of_n_spikes(&self, n:usize, period:Duration) -> f64 {
    //     let n_f64 = n as f64;
    //     let rT = self.firing_rate * self.duration_to_float(period);
    //     (rT.powf(n_f64) / n.factorial() as f64) * (-rT).exp()
    // }

    
}



