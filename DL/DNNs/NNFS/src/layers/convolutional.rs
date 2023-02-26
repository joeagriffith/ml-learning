use ndarray::{prelude::*, concatenate};
use ndarray_rand::{RandomExt, rand_distr::Uniform, rand_distr::StandardNormal};
use super::utils::{LayerTrait};
use crate::activation_funcs::*;

pub struct ConvolutionalLayer {
    weights: Array2<f32>, // sliders are concatenated rowwise, e.g. 4 3x3 sliders become 1 3x12 weight matrix. sliders are always square so can you num rows to iterate.
    num_filters: usize,
    filter_size: usize,
}

impl LayerTrait for ConvolutionalLayer {

    fn feedforward(&self, input:ArrayView1<f32>) -> Array1<f32>{
        if (input.len() as f32).sqrt().fract() != 0.0 {
            panic!("ConvLayer received non-squarable input with len: {}", input.len());
        }

        for i in 0..self.num_filters {
        
        }

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


fn convolve(input:Array1<f32>, filter:Array2<f32>)