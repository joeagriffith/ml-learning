mod config;
mod utils;
mod structures;
mod tensor;

use std::thread;
use std::time::{SystemTime, Duration};


use structures::*;
use config::*;
pub use tensor::*;
// use utils::get_axon_coords;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
