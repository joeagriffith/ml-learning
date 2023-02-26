use ndarray::{ArrayViewMut1, ArrayView1, Array1, Array2};

#[derive(PartialEq, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Softmax,
    Argmax,
}

impl ActivationFunction {
    pub fn apply(&self, input: ArrayView1<f32>) -> Array1<f32> {
        match self {
            ActivationFunction::Sigmoid => sigmoid(input),
            ActivationFunction::Tanh => tanh(input),
            ActivationFunction::ReLU => relu(input),
            ActivationFunction::LeakyReLU => leaky_relu(input),
            ActivationFunction::Softmax => softmax(input),
            ActivationFunction::Argmax => argmax(input),
        }
    }
    pub fn apply_inplace(&self, input: ArrayViewMut1<f32>) {
        match self {
            ActivationFunction::Sigmoid => sigmoid_inplace(input),
            ActivationFunction::Tanh => tanh_inplace(input),
            ActivationFunction::ReLU => relu_inplace(input),
            ActivationFunction::LeakyReLU => leaky_relu_inplace(input),
            ActivationFunction::Softmax => softmax_inplace(input),
            ActivationFunction::Argmax => argmax_inplace(input),
        }
    }
    pub fn apply_derivative(&self, input: ArrayView1<f32>) -> Array1<f32> {
        match self {
            ActivationFunction::Sigmoid => d_sigmoid(input),
            ActivationFunction::Tanh => d_tanh(input),
            ActivationFunction::ReLU => d_relu(input),
            ActivationFunction::LeakyReLU => d_leaky_relu(input),
            ActivationFunction::Softmax => d_softmax(input),
            ActivationFunction::Argmax => d_argmax(input),
        }
    }
    pub fn apply_derivative_inplace(&self, input: ArrayViewMut1<f32>) {
        match self {
            ActivationFunction::Sigmoid => d_sigmoid_inplace(input),
            ActivationFunction::Tanh => d_tanh_inplace(input),
            ActivationFunction::ReLU => d_relu_inplace(input),
            ActivationFunction::LeakyReLU => d_leaky_relu_inplace(input),
            ActivationFunction::Softmax => d_softmax_inplace(input),
            ActivationFunction::Argmax => d_argmax_inplace(input),
        }
    }
}


fn sigmoid(input:ArrayView1<f32>) -> Array1<f32> {
    input.map(|val| 1./(1. + (-*val).exp()))
}
fn sigmoid_inplace(mut array:ArrayViewMut1<f32>) {
    array.map_inplace(|val| *val = 1./(1. + (-*val).exp()))
}
fn d_sigmoid(array:ArrayView1<f32>) -> Array1<f32>{
    array.map(|val| {
        let sig = 1./(1. + (-*val).exp());
        sig * (1. - sig)
    })
}
fn d_sigmoid_inplace(mut array:ArrayViewMut1<f32>) {
    array.map_inplace(|val| {
        let sig = 1./(1. + (-*val).exp());
        *val = sig * (1. - sig)
    })
}


fn tanh(input:ArrayView1<f32>) -> Array1<f32> {
    input.map(|val| val.tanh())
}
fn tanh_inplace(mut array:ArrayViewMut1<f32>) {
    array.map_inplace(|val| *val = val.tanh())
}
fn d_tanh(array:ArrayView1<f32>) -> Array1<f32>{
    array.map(|val| 1. - val.tanh().powi(2))
}
fn d_tanh_inplace(mut array:ArrayViewMut1<f32>) {
    array.map_inplace(|val| {
        *val = 1. - val.tanh().powi(2)
    })
}


fn relu(input:ArrayView1<f32>) -> Array1<f32> {
    input.map(|val| if *val < 0.0 { 0.0 } else { *val })
}
fn relu_inplace(mut array:ArrayViewMut1<f32>) {
    array.map_inplace(|val| if *val < 0.0 { *val = 0.0;})
}
fn d_relu(array:ArrayView1<f32>) -> Array1<f32>{
    array.map(|val| if *val < 0.0 { 0.0 } else { 1.0 })
}
fn d_relu_inplace(mut array:ArrayViewMut1<f32>) {
    array.map_inplace(|val| {
        if *val < 0.0 {
            *val = 0.0
        } else {
            *val = 1.0
        }
    })
}

// a = 0.0001
fn leaky_relu(input:ArrayView1<f32>) -> Array1<f32> {
    input.map(|val| if *val < 0.0 { *val * 0.0001 } else { *val })
}
fn leaky_relu_inplace(mut array:ArrayViewMut1<f32>) {
    array.map_inplace(|val| if *val < 0.0 { *val *= 0.0001;})
}
fn d_leaky_relu(array:ArrayView1<f32>) -> Array1<f32>{
    array.map(|val| if *val < 0.0 { 0.0001 } else { 1.0 })
}
fn d_leaky_relu_inplace(mut array:ArrayViewMut1<f32>) {
    array.map_inplace(|val| {
        if *val < 0.0 {
            *val = 0.0001
        } else {
            *val = 1.0
        }
    })
}

fn softmax(input:ArrayView1<f32>) -> Array1<f32> {
    let mut e_sum = 0.0;
    for val in &input {
        e_sum += val.exp();
    }
    input.map(|val| val.exp() / e_sum)
}
fn softmax_inplace(mut array:ArrayViewMut1<f32>) {
    let mut e_sum = 0.0;
    for val in &array {
        e_sum += val.exp();
    }
    array.map_inplace(|val| *val = val.exp() / e_sum);
}
fn d_softmax(array:ArrayView1<f32>) -> Array1<f32>{
    panic!("Not implemented")
}
fn d_softmax_inplace(mut array:ArrayViewMut1<f32>) {
    panic!("Not implemented")
}


fn argmax(input:ArrayView1<f32>) -> Array1<f32> {
    let mut max = f32::MIN;
    for val in &input {
        if max > *val {
            max = *val;
        }
    }
    input.map(|val| if *val < max { 0.0 } else if *val == max { 1.0 } else { panic!("argmax should failed.") })
}
pub fn argmax_inplace(mut array:ArrayViewMut1<f32>) {
    let mut max = f32::MIN;
    let mut max_index = 0;

    for (i, val) in array.iter().enumerate() {
        if *val > max {
            max = *val;
            max_index = i;
        }
    }
    array.fill(0.0);
    array[max_index] = 1.0;
}
fn d_argmax(array:ArrayView1<f32>) -> Array1<f32>{
    panic!("Not implemented");
    let mut max = f32::MIN;
    for val in &array {
        if max > *val {
            max = *val;
        }
    }
    array.map(|val| if *val < max { 0.0 } else { 1.0 })
}
fn d_argmax_inplace(mut array:ArrayViewMut1<f32>) {
    panic!("Not implemented");
    let mut max = f32::MIN;
    for val in &array {
        if max > *val {
            max = *val;
        }
    }
    array.map_inplace(|val| {
        if *val < max {
            *val = 0.0
        } else {
            *val = 1.0
        }
    })
}