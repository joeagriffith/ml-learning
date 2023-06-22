// A file which tests a model on the mnist dataset.
use nnfs::*;
use mnist::*;
use ndarray::prelude::*;
use std::env::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub fn main() {

    let trn_len = 50_000;
    let val_len = 10_000;
    let tst_len = 10_000;

    // println!("current dir: {:?}", current_dir().unwrap());
    set_current_dir(Path::new("c:\\Users\\joegr\\Documents\\git-projects\\nnfs\\src\\bin")).unwrap();
    let (training_data, validation_data, test_data) = load_data(trn_len, val_len, tst_len);

    
    let mut model = Model::new(784, 10, Some(ActivationFunction::Sigmoid));
    // model.add_layer(LayerDescriptor::Dropout(0.8));
    model.add_layer(LayerDescriptor::Dense(16, Some(ActivationFunction::Sigmoid)));
    // model.add_layer(LayerDescriptor::Dropout(0.5));
    model.add_layer(LayerDescriptor::Dense(16, Some(ActivationFunction::Sigmoid)));
    // model.add_layer(LayerDescriptor::Dropout(0.5));
    model.compile();


    let epochs = 100;
    let batch_size = 10;
    let learning_rate = 0.01;
    let error_func = ErrorFunction::CategoricalCrossEntropyError;
    //model.evaluate(&test_data);
    model.train_sgd(training_data, epochs, batch_size, learning_rate, error_func, Some(test_data), None, None);

}

fn load_data(trn_len:usize, val_len:usize, tst_len:usize) -> (Vec<(Array1<f32>, Array1<f32>)>, Vec<(Array1<f32>, Array1<f32>)>, Vec<(Array1<f32>, Array1<f32>)>) {
    println!("Loading data...");
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(50000)
        .validation_set_length(10000)
        .test_set_length(10000)
        .finalize();

    let mut training_data = Vec::new();
    for i in 0..trn_len {
        training_data.push((
            Array1::from_shape_vec((784,), trn_img[i * 784..(i + 1) * 784].to_vec())
                .expect("Error converting image to Array1")
                .map(|x| *x as f32 / 256.0),
            Array1::from_shape_vec((10,), trn_lbl[i * 10..(i + 1) * 10].to_vec())
                .expect("Error converting label to Array1")
                .map(|x| *x as f32)
        ));
    }

    let mut validation_data = Vec::new();
    for i in 0..val_len {
        validation_data.push((
            Array1::from_shape_vec((784,), tst_img[i * 784..(i + 1) * 784].to_vec())
                .expect("Error converting image to Array1")
                .map(|x| *x as f32 / 256.0),
            Array1::from_shape_vec((10,), tst_lbl[i * 10..(i + 1) * 10].to_vec())
                .expect("Error converting label to Array1")
                .map(|x| *x as f32)
        ));
    }

    let mut test_data = Vec::new();
    for i in 0..tst_len {
        test_data.push((
            Array1::from_shape_vec((784,), tst_img[i * 784..(i + 1) * 784].to_vec())
                .expect("Error converting image to Array1")
                .map(|x| *x as f32 / 256.0),
            Array1::from_shape_vec((10,), tst_lbl[i * 10..(i + 1) * 10].to_vec())
                .expect("Error converting label to Array1")
                .map(|x| *x as f32)
        ));
    } 

    println!("finished loading data.");
    (training_data, validation_data, test_data)
}