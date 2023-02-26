use crate::SpikeTrain;

pub fn calculate_fano_factor(trains: Vec<SpikeTrain>) -> f64 {
    let mut mean:i32 = 0;
    for spike_train in &trains {
        mean += spike_train.len() as i32;
    }
    mean /= trains.len() as i32;

    let mut variance = 0;
    for spike_train in &trains {
        variance += (mean - spike_train.len() as i32).pow(2)
    }
    let mut variance = variance as f64;
    variance /= trains.len() as f64;

    variance / mean as f64
}