use std::time::Duration;

pub struct Spike {
    time: Duration,
}
impl Spike {
    pub fn new(time:Duration) -> Self {
        Self {
            time,
        }
    }
    pub fn get_time(&self) -> &Duration {
        &self.time
    }
}