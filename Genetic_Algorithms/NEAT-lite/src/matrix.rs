use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    pub values: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows:usize, cols:usize, init_value:f64) -> Matrix {
        Matrix {
            rows,
            cols,
            values: vec![vec![init_value;cols];rows],
        }
    }

    pub fn new_rand(rows:usize, cols:usize) -> Matrix {
        let mut m = Matrix::new(rows, cols, 0.0);
        let mut rng = rand::thread_rng();
        for i in 0..rows {
            for j in 0..cols {
                m.values[i][j] = rng.gen_range(-30.0..30.0);
            }
        }
        m
    }

    
    #[allow(dead_code)]
    pub fn new_rand_simple(rows:usize, cols:usize) -> Matrix {
        let mut m = Matrix::new(rows, cols, 0.0);
        let mut rng = rand::thread_rng();
        for i in 0..rows {
            for j in 0..cols {
                m.values[i][j] = rng.gen_range(-10..10) as f64 / 10.0;
            }
        }
        m
    }

    #[allow(dead_code)]
    pub fn from(vec:&Vec<Vec<f64>>) -> Matrix {
        let mut res = Matrix::new(vec.len(), vec[0].len(), 0.0);
        res.values = vec.clone();
        res
    }
    
    pub fn from_vec(vec:&Vec<f64>) -> Matrix {
        let mut res = Matrix::new(vec.len(), 1, 0.0);
        for i in 0..vec.len() {
            res.values[i][0] = vec[i];
        }
        res
    }

    pub fn to_vec(&self) -> Vec<f64> {
        let mut res:Vec<f64> = Vec::new();
        for row in 0..self.rows {
            for col in 0..self.cols {
                res.push(self.values[row][col]);
            }
        }
        res
    }

    #[allow(dead_code)]
    pub fn scalar_multiply(&mut self, scalar:f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.values[i][j] *= scalar;
            }
        }
    }

    #[allow(dead_code)]
    pub fn scalar_add (&mut self, scalar:f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.values[i][j] += scalar;
            }
        }
    }

    #[allow(dead_code)]
    pub fn element_wise_multiply(&mut self, m2:&Matrix){
        if !is_same_shape(self, m2) { 
            panic!("element-wise-multiplication failed on matrices of shapes {:?} and {:?}.", self.shape(), m2.shape());
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.values[i][j] *= m2.values[i][j];
            }
        }
    }

    pub fn element_wise_add(&mut self, m2:&Matrix){
        if !is_same_shape(self, m2) { 
            panic!("element-wise-addition failed on matrices of shapes {:?} and {:?}.", self.shape(), m2.shape());
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.values[i][j] += m2.values[i][j];
            }
        }
    }

    pub fn element_wise_sigmoid(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.values[i][j] = 1.0 / (1.0 + (-self.values[i][j]).exp());
            }
        }
    }

    #[allow(dead_code)]
    pub fn map<F>(&mut self, mut funct: F) where
        F: FnMut(&mut f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                funct(&mut self.values[i][j]);
            }
        }
    }

    pub fn crossover(&mut self, m2:&Matrix, mutation_rate:f64) {
        let mut rng = rand::thread_rng();
        for i in 0..self.rows {
            for j in 0..self.cols {
                if rng.gen_range(0.0..1.0) < mutation_rate {
                    self.values[i][j] = rng.gen_range(-30.0..30.0);
                } else {
                    if rng.gen::<bool>() {
                        self.values[i][j] = m2.values[i][j];
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn transpose(&mut self) {
        *self = transpose(self);
    }

    #[allow(dead_code)]
    pub fn dot(&mut self, m2:&Matrix) {
        *self = dot(self, m2);
    }

    pub fn dbg(&self) {
        println!("");
        for i in 0..self.rows {
            println!("{:?}", self.values[i]);
        }
        println!("");
    }
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols)}
}

fn is_same_shape(m1:&Matrix, m2:&Matrix) -> bool {
    m1.shape() == m2.shape()
}
fn is_dottable(m1:&Matrix, m2:&Matrix) -> bool {
    m1.shape().1 == m2.shape().0
}

#[allow(dead_code)]
pub fn transpose(m1:&Matrix) -> Matrix {
    let mut res = Matrix::new(m1.cols, m1.rows, 0.0);
    for i in 0..m1.rows {
        for j in 0..m1.cols {
            res.values[j][i] = m1.values[i][j];
        }
    }
    res
}

pub fn dot(m1:&Matrix, m2:&Matrix) -> Matrix {
    if !is_dottable(m1, m2) {
        panic!("dot multiplication failed on matrices of shapes {:?} and {:?}.", m1.shape(), m2.shape());
    }
    let mut res = Matrix::new(m1.shape().0, m2.shape().1, 0.0);
    for i in 0..m1.rows() {
        for j in 0..m2.cols() {
            for x in 0..m1.shape().1 {
                res.values[i][j] += m1.values[i][x] * m2.values[x][j];
            }
        }
    }
    res
}