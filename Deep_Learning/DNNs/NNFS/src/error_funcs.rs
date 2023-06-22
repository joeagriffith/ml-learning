use ndarray::{ArrayView1, Array1};
use core::fmt::Debug;
use std::fmt::Formatter;

pub enum ErrorFunction {
    MeanSquaredError,
    CategoricalCrossEntropyError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
}
impl ErrorFunction {
    pub fn apply(&self, output:ArrayView1<f32>, target:ArrayView1<f32>) -> f32 {
        match self {
            ErrorFunction::MeanSquaredError => mean_squared_error(output, target),
            ErrorFunction::CategoricalCrossEntropyError => categorical_cross_entropy_error(output, target),
            ErrorFunction::MeanAbsoluteError => mean_absolute_error(output, target),
            ErrorFunction::MeanAbsolutePercentageError => mean_absolute_percentage_error(output, target),
        }
    }
    pub fn apply_derivative(&self, output:ArrayView1<f32>, target: ArrayView1<f32>) -> Array1<f32> {
        match self {
            ErrorFunction::MeanSquaredError => d_mse(output, target),
            ErrorFunction::CategoricalCrossEntropyError => d_ccee(output, target),
            ErrorFunction::MeanAbsoluteError => d_mae(output, target),
            ErrorFunction::MeanAbsolutePercentageError => d_mape(output, target),
        }
    }
}

impl Debug for ErrorFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            ErrorFunction::MeanSquaredError => write!(f, "MeanSquaredError"),
            ErrorFunction::CategoricalCrossEntropyError => write!(f, "CategoricalCrossEntropyError"),
            ErrorFunction::MeanAbsoluteError => write!(f, "MeanAbsoluteError"),
            ErrorFunction::MeanAbsolutePercentageError => write!(f, "MeanAbsolutePercentageError"),
        }

    }
}

fn mean_squared_error(output: ArrayView1<f32>, target: ArrayView1<f32>) -> f32 {
    let mut error = 0.0;
    for (o, t) in output.iter().zip(target.iter()) {
        error += (o - t).powi(2);
    }
    error / output.len() as f32
}
fn d_mse(output:ArrayView1<f32>, target:ArrayView1<f32>) -> Array1<f32> {
    let n = target.len();
    Array1::from_shape_fn(target.shape()[0], |i| -2.0 * (target[i] - output[i]) / n as f32)
}

fn categorical_cross_entropy_error(output: ArrayView1<f32>, target: ArrayView1<f32>) -> f32 {
    let mut error = 0.0;
    for (o, t) in output.iter().zip(target.iter()) {
        error += t * o.ln() + (1.0-t) * o.ln();
    }
    error
}
fn d_ccee(output: ArrayView1<f32>, target: ArrayView1<f32>) -> Array1<f32> {
    let len = output.len() as f32;
    Array1::from_shape_fn(target.shape()[0], |i| {
        ((target[i]/output[i]) - ((1.0 - target[i])/(1.0 - output[i]))) / -len
    })
}


fn mean_absolute_error(output: ArrayView1<f32>, target: ArrayView1<f32>) -> f32 {
    let mut error = 0.0;
    for (o, t) in output.iter().zip(target.iter()) {
        error += (t - o).abs();
    }
    error / output.len() as f32
}

fn mean_absolute_percentage_error(output: ArrayView1<f32>, target: ArrayView1<f32>) -> f32 {
    let mut error = 0.0;
    for (o, t) in output.iter().zip(target.iter()) {
        error += ((t - o).abs() / t) * 100.0;
    }
    error / output.len() as f32
}


fn d_mae(output: ArrayView1<f32>, target: ArrayView1<f32>) -> Array1<f32> {
    Array1::from_shape_fn(target.shape()[0], |i| {
        1.0
    })
}

fn d_mape(output: ArrayView1<f32>, target: ArrayView1<f32>) -> Array1<f32> {
    Array1::from_shape_fn(target.shape()[0], |i| {
        100.0/target[i]
    })
}





#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn correct_error_mse() {
        let output = arr1(&[1., 3., 4., 2., 2.]);
        let target = arr1(&[1., 1., 1., 1., 1.]);
        assert_eq!(mean_squared_error(output.view(), target.view()), 3.0);
    }

    #[test]
    fn correct_error_ccee() {
        let output = arr1(&[0.1, 0.9, 0.1, 0.8, 0.3]);
        let target = arr1(&[0., 0., 1., 0., 0.]);
        assert_eq!(categorical_cross_entropy_error(output.view(), target.view()), -0.1_f32.ln());
    }

    #[test]
    fn correct_error_mae() {
        let output = arr1(&[1., 3., 4., 2., 2.]);
        let target = arr1(&[1., 1., 1., 1., 1.]);
        assert_eq!(mean_absolute_error(output.view(), target.view()), 1.4);
    }

    #[test]
    fn correct_error_mape() {
        let output = arr1(&[1., 3., 4., 2., 2.]);
        let target = arr1(&[1., 1., 1., 1., 1.]);
        assert_eq!(mean_absolute_percentage_error(output.view(), target.view()), 140.0);
    }
}