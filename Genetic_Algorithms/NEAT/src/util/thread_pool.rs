use std::thread;
use std::sync::{mpsc, Arc, Mutex, RwLock};
use crate::config::POPULATION_SIZE;
use crate::genetics::Genome;



pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Message>,
}

impl ThreadPool {
    pub fn new(size: usize, fitnesses: &Arc<Mutex<[f64;POPULATION_SIZE]>>, organisms: &Arc<RwLock<Vec<Genome>>>) -> ThreadPool {
        assert!(size != 0);

        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(size);

        for _ in 0..size {
            workers.push(Worker::new(Arc::clone(&receiver), Arc::clone(fitnesses), Arc::clone(organisms)));
        }

        ThreadPool { workers, sender, }
    }

    pub fn execute(&self, i:usize, env:for<'r> fn(&'r Genome) -> f64) {
        self.sender.send(Message::CalcFit(i, env)).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            self.sender.send(Message::Terminate).unwrap();
        }

        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

struct Worker {
    thread: Option<thread::JoinHandle<()>>,
    // fitnesses: Arc<Mutex<[f64;POPULATION_SIZE]>>,
    // organisms: Arc<RwLock<Vec<Genome>>>,
}
impl Worker {
    pub fn new(receiver: Arc<Mutex<mpsc::Receiver<Message>>>, fitnesses:Arc<Mutex<[f64;POPULATION_SIZE]>>, organisms: Arc<RwLock<Vec<Genome>>>) -> Self {
        let thread = Some(thread::spawn(move || loop {
            let message = receiver.lock().unwrap().recv().unwrap();
            match message {
                Message::CalcFit(i ,env) => {
                    let fitness = env(&organisms.read().unwrap()[i]);
                    fitnesses.lock().unwrap()[i] = fitness;
                }
                Message::Terminate => {
                    break;
                }
            }
        }));
        Self { thread }
    }
}

enum Message {
    CalcFit(usize, for<'r> fn(&'r Genome) -> f64),
    Terminate,
}