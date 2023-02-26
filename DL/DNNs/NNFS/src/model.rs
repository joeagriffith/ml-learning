use std::{ops::MulAssign};

use std::ops::{AddAssign, SubAssign};
use super::layers::{LayerTrait, LayerDescriptor, DenseLayer, DropoutLayer};
use crate::{activation_funcs::*, error_funcs::*};
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::SliceRandom;
use rand::prelude::*;

use super::optimisers::Adam;

use std::time::{SystemTime, Duration};


pub struct Model {
    layer_descriptors: Vec<LayerDescriptor>,
    layers: Vec<Box<dyn LayerTrait>>,
    shape: (usize,usize),
    actv_func: Option<ActivationFunction>,
    opt: Option<Adam>,
}
impl Model {
    pub fn new(inputs:usize, outputs:usize,  actv_func: Option<ActivationFunction>) -> Self {
        Self { 
            layer_descriptors: Vec::new(),
            layers: Vec::new(),
            shape: (inputs, outputs),
            actv_func,
            opt: None,
        }
    }

    pub fn add_layer(&mut self, layer_descriptor: LayerDescriptor) {
        self.layer_descriptors.push(layer_descriptor);
    }

    fn build_layer(&mut self, layer_descriptor: LayerDescriptor, inputs: usize) {
        match layer_descriptor {
            LayerDescriptor::Dense(outputs, actv_func) => {
                println!("  Dense layer, nodes: {}", outputs);
                self.layers.push(Box::new(DenseLayer::new_standard_normal(inputs, outputs, actv_func)));
            },
            LayerDescriptor::DenseOnes(outputs, actv_func) => {
                println!("  DenseOnes layer, nodes: {}", outputs);
                self.layers.push(Box::new(DenseLayer::new_ones(inputs, outputs, actv_func)));
            }
            LayerDescriptor::Dropout(p) => {
                // let len = self.layers.len();
                // let inputs = if len == 0 {

                // } else {
                //     self.layers[len-1].get_output_size()
                // };
                println!("  Dropout layer, p: {p}");
                self.layers.push(Box::new(DropoutLayer::new(inputs, p)));
            }
        }
    }

    pub fn compile(&mut self) {
        let len = self.layer_descriptors.len();
        println!("\n---Compiling Model---");
        println!("  Input layer, nodes: {}", self.shape.0);
        for i in 0..len {
            if i == 0 {
                self.build_layer(self.layer_descriptors[i], self.shape.0);
            } else {
                self.build_layer(self.layer_descriptors[i], self.layers[i-1].get_output_size());
            }
        }
        self.build_layer(LayerDescriptor::Dense(self.shape.1, self.actv_func), self.layers[len-1].get_output_size());

        self.opt = Some(Adam::new_default(&self.layers));
    }

    pub fn compile_with_ones(&mut self) {
        let len = self.layer_descriptors.len();
        for i in 0..len {
            if i == 0 {
                self.build_layer(self.layer_descriptors[i], self.shape.0);
            } else {
                self.build_layer(self.layer_descriptors[i], self.layers[i-1].get_output_size());
            }
        }
        self.build_layer(LayerDescriptor::DenseOnes(self.shape.1, self.actv_func), self.layers[len-1].get_output_size());
    }

    pub fn feedforward(&self, input:ArrayView1<f32>) -> Array1<f32> {
        // if self.shape.0 != input.shape()[0] {
        //     panic!("input is of invalid shape: {}, only accepting: {}", input.shape()[0], self.shape.0)
        // }

        let len = self.layers.len();
        // if self.shape.1 != self.layers[len-1].shape()[0] {
        //     panic!("model's last layer contradicts with target model output, model output size: {}, last layer output size: {}", self.shape.1, self.layers[len-1].shape()[1]);
        // }
        // println!("\n\n");


        let mut output:Array1<f32> = self.layers[0].feedforward(input);
        for i in 1..len {
            output = self.layers[i].feedforward(output.view());
        }

        if let Some(func) = &self.actv_func {
            func.apply_inplace(output.view_mut());
        }
        output
    }

    pub fn train_sgd(&mut self, mut training_data:Vec<(Array1<f32>, Array1<f32>)>, epochs:usize, batch_size:usize, learning_rate:f32, error_func:ErrorFunction, test_data:Option<Vec<(Array1<f32>, Array1<f32>)>>, l1_lambda: Option<f32>, l2_lambda: Option<f32>) {
        println!("\nTraining via stochastic gradient descent with options:");
        println!("  epochs: {epochs}\n  batch_size: {batch_size}\n  learning_rate: {learning_rate}\n  error_func: {:?}\n  l1_lambda: {:?}\n  l2_lambda: {:?}\n", error_func, l1_lambda, l2_lambda);
        self.evaluate(test_data.as_ref().unwrap());
        for j in 0..epochs {
            println!("epoch: {j}");
            training_data.shuffle(&mut thread_rng());
            let batches = training_data.chunks(batch_size);
            for batch in batches {
                self.update_batch(batch, learning_rate, &error_func, l1_lambda, l2_lambda);
            }
            println!("epoch: {}/{epochs} complete", j+1);

            if test_data.is_some() {
                println!("evaluating...");
                self.evaluate(test_data.as_ref().unwrap());
            }
        }
    }


    pub fn update_batch(&mut self, batch:&[(Array1<f32>, Array1<f32>)], learning_rate:f32, error_func:&ErrorFunction, l1_lambda: Option<f32>, l2_lambda: Option<f32>) {
        let mut nabla_w:Vec<Array2<f32>> = Vec::new();
        let mut nabla_b:Vec<Array1<f32>> = Vec::new();

        for i in 0..self.layers.len() {
            nabla_w.push(Array2::zeros(self.layers[i].get_weights().dim()));
            if self.layers[i].has_biases() {
                nabla_b.push(Array1::zeros(self.layers[i].get_biases().unwrap().dim()));
            } else {
                nabla_b.push(Array1::zeros(0));
            }
        }

        let layers_len = self.layers.len();
        for (input, target) in batch {
            let (delta_nabla_w, delta_nabla_b) = self.backprop(input.view(), target.view(), &error_func);
            for i in 0..layers_len {
                if !self.layers[i].is_trainable() {
                    continue;
                }
                nabla_w[i].add_assign(&delta_nabla_w[layers_len-i-1]); //reverse index as backprop returns in reverse order
                if self.layers[i].has_biases() {
                    nabla_b[i].add_assign(&delta_nabla_b[layers_len-i-1]);
                }
            }
        }    


        if let Some(lambda) = l1_lambda {
            for i in 0..self.layers.len() {
                if !self.layers[i].is_trainable() {
                    continue;
                }
                let w_reg = Array2::from_shape_fn(self.layers[i].get_weights().dim(), |(k,j)| {
                    let w = self.layers[i].get_weights()[(k,j)];
                    if w > lambda {
                        lambda
                    } else if w <= -lambda {
                        -lambda
                    } else {
                        w
                    }
                });
                self.layers[i].get_weights_mut().sub_assign(&w_reg);

                if self.layers[i].has_biases() {
                    let b_reg = Array1::from_shape_fn(self.layers[i].get_biases().unwrap().dim(), |(j)| {
                        let b = self.layers[i].get_biases().unwrap()[(j)];
                        if b >= lambda {
                            lambda
                        } else if b <= -lambda {
                            -lambda
                        } else {
                            b
                        }
                    });
                    self.layers[i].get_biases_mut().unwrap().sub_assign(&b_reg);
                }
            }
        }

        if let Some(lambda) = l2_lambda {
            for i in 0..self.layers.len() {
                if !self.layers[i].is_trainable() {
                    continue;
                }
                let w_reg = self.layers[i].get_weights().to_owned() * lambda;
                self.layers[i].get_weights_mut().sub_assign(&w_reg);

                if self.layers[i].has_biases() {
                    let b_reg = self.layers[i].get_biases().unwrap().to_owned() * lambda;
                    self.layers[i].get_biases_mut().unwrap().sub_assign(&b_reg);
                }
            }
        }

        for i in 0..layers_len {
            if !self.layers[i].is_trainable() {
                continue;
            }
            if self.opt.is_some() {
                let (dw, db) = self.opt.as_mut().unwrap().calculate_layer(i, nabla_w[i].view(), nabla_b[i].view());
                self.layers[i].get_weights_mut().sub_assign(&(&dw * learning_rate / batch.len() as f32));
                if self.layers[i].has_biases() {
                    self.layers[i].get_biases_mut().unwrap().sub_assign(&(&db * learning_rate / batch.len() as f32));
                }
            } else {
                self.layers[i].get_weights_mut().sub_assign(&(&nabla_w[i] * learning_rate / batch.len() as f32));
                if self.layers[i].has_biases() {
                    self.layers[i].get_biases_mut().unwrap().sub_assign(&(&nabla_b[i] * learning_rate / batch.len() as f32));
                }
            }
        }

    }

    pub fn backprop(&mut self, input:ArrayView1<f32>, target:ArrayView1<f32>, error_func:&ErrorFunction) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {

        let mut nabla_w:Vec<Array2<f32>> = Vec::new();
        let mut nabla_b:Vec<Array1<f32>> = Vec::new();

        let mut activations:Vec<Array1<f32>> = vec![input.to_owned()];
        let mut zs = Vec::new();
        for i in 0..self.layers.len() {
            zs.push(self.layers[i].get_zs(activations[i].view()));
            activations.push(self.layers[i].apply_activation(zs[i].view()));
        }

        
        let mut delta = error_func.apply_derivative(activations[activations.len()-1].view(), target.view());
        // let mut delta = &activations[activations.len()-1] - &target;
        // let mut delta = -(&target - &activations[activations.len()-1]);
        if let Some(func) = &self.actv_func {
            delta.mul_assign(&func.apply_derivative(zs[zs.len()-1].view()));
        }
        nabla_w.push(dot_1d_arrays(delta.view(), activations[activations.len()-2].view()));
        nabla_b.push(delta.to_owned());

        for i in (0..self.layers.len()-1).rev() {

            let z = zs[i].view();
            let sp = self.layers[i].apply_derivative(z);
            // println!("delta before: {:?}", delta);
            // if self.layers[i].is_trainable() {}
            // if self.layers[i+1].is_minimally_connected() {
            //     println!("diagmult");
            //     delta.mul_assign(&self.layers[i+1].get_weights().diag());
            //     delta = delta * sp;
            // } else {
            if self.layers[i].is_trainable()  || i > 1{
                delta = &self.layers[i+1].get_weights().t().dot(&delta.view()) * sp;
            }
            if !self.layers[i].is_trainable() {
                nabla_w.push(Array2::zeros((0,0)));
                nabla_b.push(Array1::zeros(0));
            } else {
                nabla_w.push(dot_1d_arrays(delta.view(), activations[i].view()));
                nabla_b.push(delta.to_owned());
            }
            // }
            // println!("delta after: {:?}", delta);
        }

        (nabla_w, nabla_b)
    }

    pub fn evaluate(&self, test_data:&Vec<(Array1<f32>, Array1<f32>)>) {
        let mut test_results = Vec::new();
        for (input, target) in test_data {
            let mut output = self.feedforward(input.view());
            argmax_inplace(output.view_mut());
            test_results.push((output, target));
        }
        let mut num_correct = 0;
        for (output, target) in &test_results {
            if output == target{
                num_correct += 1;
            }
        }

        let len = test_results.len();
        let test_accuracy = (num_correct as f32 / len as f32) * 100.0;
        println!("correct: {num_correct}/{len}, accuracy: {test_accuracy}%");
    }

}

fn dot_1d_arrays(a:ArrayView1<f32>, b:ArrayView1<f32>) -> Array2<f32> {
    Array2::from_shape_fn((a.shape()[0], b.shape()[0]), |(i, j)| a[i] * b[j])
}



#[cfg(test)]
mod tests {
    use crate::{Model, LayerDescriptor, ActivationFunction};
    use ndarray::prelude::*;
    use super::dot_1d_arrays;

    #[test]
    fn test_dot_1d_arrays() {
        let a = Array1::from_shape_vec(3, vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array1::from_shape_vec(3, vec![4.0, 5.0, 6.0]).unwrap();
        let c = dot_1d_arrays(a.view(), b.view());
        assert_eq!(Array2::from_shape_fn((3, 3), |(i,j)| (i+1) as f32 * (j+4) as f32), c);
    }

    #[test]
    fn basic() {
        let mut model = Model::new(3, 1, None);
        model.add_layer(LayerDescriptor::DenseOnes(2, None));
        model.compile_with_ones();
        
        let input = arr1(&[1., 2., 3.]);
        let output = model.feedforward(input.view());

        let target = arr1(&[15.]);
        assert_eq!(target, output);
    }

    #[test]
    fn with_softmax() {
        let mut model = Model::new(3, 1, Some(ActivationFunction::Softmax));
        model.add_layer(LayerDescriptor::DenseOnes(2, None));
        model.compile_with_ones();
        
        let input = arr1(&[1., 2., 3.]);
        let output = model.feedforward(input.view());

        let target = arr1(&[1.0]);
        assert_eq!(target, output);
    }
}