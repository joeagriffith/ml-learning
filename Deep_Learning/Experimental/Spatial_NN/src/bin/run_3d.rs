use std::time::{SystemTime, Duration};
use snn::Tensor3D;



// TSSNN - TIME STEPPED SPATIAL NEURAL NETWORK
fn main() {
    for size in 3..=60 {
        let start = (0,0,0);
        let end = (size-1, size-1, size-1);
        let mut tensor = Tensor3D::new_ones(size);
        tensor.try_activate_node(start, 1.0);

        let start_time = SystemTime::now();
        while !tensor.node_is_active(end).unwrap() {
            tensor.update();
        }
        let duration = start_time.elapsed().unwrap();
        println!("{:?}", duration.as_millis());
    }
    
}
