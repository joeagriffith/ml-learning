use crate::OUTPUTS;

pub fn softmax(input: [f64;OUTPUTS]) -> [usize;OUTPUTS] {
    let mut idx = 0;
    let mut max = input[0];
    for i in 1..input.len() {
        if input[i] > max {
            idx = i;
            max = input[i];
        }
    }
    let mut output = [0;OUTPUTS];
    output[idx] = 1;
    output
}