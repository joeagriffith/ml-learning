use super::spike::*;

pub struct SpikeTrain {
    spikes: Vec<Spike>,
}
impl SpikeTrain {
    pub fn new() -> Self {
        Self {
            spikes: Vec::new(),
        }
    }
    pub fn add_spike(&mut self, spike:Spike) {
        self.spikes.push(spike);
    }
    pub fn get_spike(&self, idx:usize) -> Option<&Spike> {
        self.spikes.get(idx)
    }
    pub fn remove_spike(&mut self, idx:usize) {
        self.spikes.remove(idx);
    }
    pub fn print(&self) {
        for spike in &self.spikes {
            println!("Spike at: {:?}", spike.get_time());
        }
    }
    pub fn len(&self) -> usize {
        self.spikes.len()
    }
}