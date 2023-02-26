use std::time::Duration;
use crate::SpikeTrain;

pub trait SpikeGenerator {
    fn generate_spike_train(&mut self, period:Duration) -> SpikeTrain;
}