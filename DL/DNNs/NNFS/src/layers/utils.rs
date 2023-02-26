use ndarray::prelude::*;
use lazy_static::*;

use crate::ActivationFunction;


// lazy_static! {
//     pub static ref BIAS: Array1<f32> = arr1(&[-1.0]);
// }

pub trait LayerTrait {
    fn feedforward(&self, input:ArrayView1<f32>) -> Array1<f32>;
    fn get_zs(&mut self, input:ArrayView1<f32>) -> Array1<f32>;
    fn apply_activation(&self, input:ArrayView1<f32>) -> Array1<f32>;
    fn apply_derivative_inplace(&self, input:ArrayViewMut1<f32>);
    fn apply_derivative(&self, input:ArrayView1<f32>) -> Array1<f32>;
    fn shape(&self) -> &[usize];
    fn get_weights(&self) -> ArrayView2<f32>;
    fn get_weights_mut(&mut self) -> ArrayViewMut2<f32>;
    fn get_biases(&self) -> Option<ArrayView1<f32>>;
    fn get_biases_mut(&mut self) -> Option<ArrayViewMut1<f32>>;
    fn get_output_size(&self) -> usize;
    fn is_trainable(&self) -> bool;
    fn is_minimally_connected(&self) -> bool;
    fn has_biases(&self) -> bool;
}

#[derive(Clone, Copy)]
pub enum LayerDescriptor {
    Dense(usize, Option<ActivationFunction>), // nodes, Optional activation function
    DenseOnes(usize, Option<ActivationFunction>), // nodes, Optional activation function
    Dropout(f64), // p - probability of each node being dropped.
}
