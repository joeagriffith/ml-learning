use ndarray::prelude::*;
use ndarray::Zip;
use super::super::layers::LayerTrait;


pub struct Adam {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,

    v_dw: Vec<Array2<f32>>,
    v_db: Vec<Array1<f32>>,
    s_dw: Vec<Array2<f32>>,
    s_db: Vec<Array1<f32>>,

    iters: usize,
}

impl Adam {
    pub fn new_default(layers: &Vec<Box<dyn LayerTrait>>) -> Self {
        let mut optimiser = Adam {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 10e-8, 

            v_dw: Vec::new(),
            v_db: Vec::new(),
            s_dw: Vec::new(),
            s_db: Vec::new(),

            iters: 0,
        };

       for i in 0..layers.len() {
            optimiser.v_dw.push(Array2::zeros(layers[i].get_weights().dim()));
            optimiser.s_dw.push(Array2::zeros(layers[i].get_weights().dim()));
            if layers[i].has_biases() {
                optimiser.v_db.push(Array1::zeros(layers[i].get_biases().unwrap().dim()));
                optimiser.s_db.push(Array1::zeros(layers[i].get_biases().unwrap().dim()));
            } else {
                optimiser.v_db.push(Array1::zeros(0));
                optimiser.s_db.push(Array1::zeros(0));
            }
        }

        optimiser
    }

    fn update_layer(&mut self, layer: usize, dw: ArrayView2<f32>, db: ArrayView1<f32>) {
        self.iters += 1;

        self.v_dw[layer] = self.beta_1 * self.v_dw[layer].to_owned() + (1.0 - self.beta_1) * dw.to_owned();
        self.v_db[layer] = self.beta_1 * self.v_db[layer].to_owned() + (1.0 - self.beta_1) * db.to_owned();

        self.s_dw[layer] = self.beta_2 * self.s_dw[layer].to_owned() + (1.0 - self.beta_2) * dw.to_owned() * dw;
        self.s_db[layer] = self.beta_2 * self.s_db[layer].to_owned() + (1.0 - self.beta_2) * db.to_owned() * db;
    }

    fn get_corrected(&self, layer: usize) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
        let v_dw_corrected = self.v_dw[layer].to_owned() / (1.0 - self.beta_1.powi(self.iters as i32));
        let v_db_corrected = self.v_db[layer].to_owned() / (1.0 - self.beta_1.powi(self.iters as i32));

        let s_dw_corrected = self.s_dw[layer].to_owned() / (1.0 - self.beta_2.powi(self.iters as i32));
        let s_db_corrected = self.s_db[layer].to_owned() / (1.0 - self.beta_2.powi(self.iters as i32));

        (v_dw_corrected, v_db_corrected, s_dw_corrected, s_db_corrected)
    }

    pub fn calculate_layer(&mut self, layer:usize, dw:ArrayView2<f32>, db: ArrayView1<f32>) -> (Array2<f32>, Array1<f32>) {
        self.update_layer(layer, dw, db);
        let (mut v_dw_c, mut v_db_c, s_dw_c, s_db_c) = self.get_corrected(layer);

        Zip::from(&mut v_dw_c).and(&s_dw_c).for_each(|v, &s| {
            *v = *v / (s.sqrt() + self.epsilon);
        });

        Zip::from(&mut v_db_c).and(&s_db_c).for_each(|v, &s|  {
            *v = *v / (s.sqrt() + self.epsilon);
        });

        (v_dw_c, v_db_c)
    }
}