mod structures;
pub use structures::*;
mod spike_generators;
pub use spike_generators::*;
mod utils;
pub use utils::*;
pub mod prelude;
pub use prelude::*;





#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
