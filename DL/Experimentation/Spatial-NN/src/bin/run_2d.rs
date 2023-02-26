use std::time::{SystemTime, Duration};
use snn::Tensor2D;



// TSSNN - TIME STEPPED SPATIAL NEURAL NETWORK
fn main() {
    let mut tensor = Tensor2D::new(4);
    tensor.try_activate_random_node();
    tensor.try_activate_random_node();
    tensor.try_activate_random_node();
    let mut total = Duration::ZERO;
    for _ in 0..10000 {
        let start_time = SystemTime::now();
        tensor.update();
        total += start_time.elapsed().unwrap();
    }
    total /= 10000;

    println!("mean update() time: {:?}", total);
}
