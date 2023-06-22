//NEURAL NETWORKS FROM SCRATCH, from nd-arrays to transformers, ideally.

mod layers;
pub use layers::LayerDescriptor;

mod model;
pub use model::*;

mod activation_funcs;
pub use activation_funcs::*;

mod error_funcs;
pub use error_funcs::*;

mod optimisers;
pub use optimisers::*;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
