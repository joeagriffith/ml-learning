use std::ops::MulAssign;

use ndarray::prelude::*;
use ndarray::Slice;
use ndarray_rand::{RandomExt, rand_distr::Bernoulli};
use super::utils::{LayerTrait};
use rand::prelude::*;

pub struct DropoutLayer {
    p: f64,
    weights: Array2<f32>,
    // biases: None,
}

impl DropoutLayer {
    pub fn new(inputs: usize, p:f64) -> Self{
        Self {
            p,
            weights: Array2::zeros((inputs, inputs)),
            // biases: Array1::ones(inputs),
        }
    }

}

impl LayerTrait for DropoutLayer {

    fn feedforward(&self, input:ArrayView1<f32>) -> Array1<f32> {
        //println!("Dropout.feedforward(), input: {:?}", input);
        let out = input.to_owned();
        // println!("output: {:?}", out);
        out * self.p as f32
    }


    fn get_zs(&mut self, input:ArrayView1<f32>) -> Array1<f32> {
        let bernoulli = Array1::random(input.dim(), Bernoulli::new(self.p).unwrap());
        // self.weights = Array2::from_shape_fn((input.dim(), input.dim()), |(i, j)| {
        //     if i == j && bernoulli[j] {
        //         1.0
        //     } else {
        //         0.0

        //     }
        // });

        for i in 0..bernoulli.len() {
            if bernoulli[i] {
                self.weights[(i,i)] = 1.0;
            }
        }

        self.weights.dot(&input)
    }

    fn apply_activation(&self, input:ArrayView1<f32>) -> Array1<f32> {
        input.to_owned()
    }

    fn is_trainable(&self) -> bool {
        false
    }
    fn is_minimally_connected(&self) -> bool {
        true
    }

    fn apply_derivative(&self, input:ArrayView1<f32>) -> Array1<f32> {
        // self.weights.slice(s![0,..]).to_owned()// * input
        Array1::ones(input.dim())
    }

    fn apply_derivative_inplace(&self, mut input:ArrayViewMut1<f32>) {
        
    }

    fn get_output_size(&self) -> usize {
        self.weights.shape()[1]
    }


    fn shape(&self) -> &[usize] {
        self.weights.shape()
    }

    fn get_weights(&self) -> ArrayView2<f32> {
        self.weights.view()
    }
    fn get_biases(&self) -> Option<ArrayView1<f32>> {
        None
    }
    fn get_weights_mut(&mut self) -> ArrayViewMut2<f32> {
        self.weights.view_mut()
    }
    fn get_biases_mut(&mut self) -> Option<ArrayViewMut1<f32>> {
        None
    }
    fn has_biases(&self) -> bool {
        false
    }
}