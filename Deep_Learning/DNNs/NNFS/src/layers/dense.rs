use ndarray::{prelude::*, concatenate};
use ndarray_rand::{RandomExt, rand_distr::Uniform, rand_distr::StandardNormal};
use super::utils::{LayerTrait};
use crate::activation_funcs::*;


pub struct DenseLayer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    actv_func: Option<ActivationFunction>,
}

impl DenseLayer {
    pub fn new_uniform(inputs:usize, outputs:usize, max_weight:f32, actv_func: Option<ActivationFunction>) -> Self {
        Self {
            weights: Array2::random((outputs, inputs), Uniform::new(-max_weight, max_weight)),
            biases: Array1::random(outputs, Uniform::new(-max_weight, max_weight)),
            actv_func,
        }
    }

    pub fn new_standard_normal(inputs:usize, outputs:usize, actv_func: Option<ActivationFunction>) -> Self {
        Self {
            weights: Array2::random((outputs, inputs), StandardNormal),
            biases: Array1::random(outputs, StandardNormal),
            actv_func,
        }
    }

    pub fn new_ones(inputs:usize, outputs:usize, actv_func: Option<ActivationFunction>) -> Self {
        Self {
            weights: Array2::ones((outputs, inputs)),
            biases: Array1::ones(outputs),
            actv_func,
        }
    }
}

impl LayerTrait for DenseLayer {

    fn feedforward(&self, input:ArrayView1<f32>) -> Array1<f32>{
        // println!("Dense.feedforward(), input: {:?}", input);
        let mut output = self.weights.dot(&input) + self.biases.view();
        output = self.apply_activation(output.view()); 
        // println!("output: {:?}", output);
        output
    }

    fn get_zs(&mut self, input:ArrayView1<f32>) -> Array1<f32> {
        self.weights.dot(&input) + self.biases.view()
    }

    fn apply_activation(&self, input:ArrayView1<f32>) -> Array1<f32> {

        if let Some(func) = &self.actv_func {
            func.apply(input.view())
        } else {
            input.to_owned()
        }

    }
    
    fn is_trainable(&self) -> bool {
        true
    }
    fn is_minimally_connected(&self) -> bool {
        false
    }

    fn apply_derivative(&self, input:ArrayView1<f32>) -> Array1<f32> {
        if let Some(func) = &self.actv_func {
            func.apply_derivative(input.view())
        } else {
            Array1::<f32>::ones(input.shape()[0])
        }
    }

    fn apply_derivative_inplace(&self, mut input:ArrayViewMut1<f32>) {
        if let Some(func) = &self.actv_func {
            func.apply_derivative_inplace(input);
        } else {
            input.fill(1.0);
        }
    }

    fn get_output_size(&self) -> usize {
        self.weights.shape()[0]
    }


    fn shape(&self) -> &[usize] {
        self.weights.shape()
    }

    fn get_weights(&self) -> ArrayView2<f32> {
        self.weights.view()
    }
    fn get_biases(&self) -> Option<ArrayView1<f32>> {
        Some(self.biases.view())
    }
    fn get_weights_mut(&mut self) -> ArrayViewMut2<f32> {
        self.weights.view_mut()
    }
    fn get_biases_mut(&mut self) -> Option<ArrayViewMut1<f32>> {
        Some(self.biases.view_mut())
    }
    fn has_biases(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    // use crate::{DenseLayer, LayerTrait, ActivationFunction};
    use super::DenseLayer;
    use super::super::utils::LayerTrait;
    use crate::ActivationFunction;
    use ndarray::prelude::*;

    #[test]
    #[should_panic]
    fn input_too_small() {
        let mut layer = DenseLayer::new_standard_normal(4, 5, Some(ActivationFunction::Sigmoid));
        let input = arr1(&[1., 2., 3.]);
        layer.feedforward(input.view());
    }

    #[test]
    #[should_panic]
    fn input_too_large() {
        let mut layer = DenseLayer::new_standard_normal(2, 5, None);
        let input = arr1(&[1., 2., 3.]);
        layer.feedforward(input.view());
    }

    #[test]
    fn correct_activations() {
        let mut layer = DenseLayer::new_ones(3, 4, None);
        let input = arr1(&[1., -2., 3.]);
        let target = arr1(&[3., 3., 3., 3.]);

        assert_eq!(&target, layer.feedforward(input.view()));
    }

    #[test]
    fn correct_activations_w_sigmoid() {
        let mut layer = DenseLayer::new_ones(3, 2, Some(ActivationFunction::Sigmoid));
        let input = arr1(&[1., -2., 3.]);
        let target = arr1(&[0.9933, 0.9933]);
        let output = layer.feedforward(input.view());
        let result = output - target;

        assert!(result[0] < 0.00001 && result[1] < 0.00001);
    }
}