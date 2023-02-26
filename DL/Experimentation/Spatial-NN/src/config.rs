// pub const WIDTH:usize = 4;
// pub const HEIGHT:usize = 4;

use std::time::Duration;

pub const REFRACTORY_DURATION:Duration = Duration::from_millis(950);
pub const CONNECTION_ACTIVE_DURATION:Duration = Duration::from_millis(10);
pub const CONNECTION_MAX_CHECK:usize = 1;
pub const THRESHOLD_NODE_ACTIVATION:f32 = 0.75;
pub const MAX_WEIGHT:f32 = 1.0;
pub const NUM_THREADS:usize = 4;
