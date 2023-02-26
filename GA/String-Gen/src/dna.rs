use rand::Rng;

#[derive(Clone, Debug)]
pub struct DNA {
    pub dna: String,
    pub fitness:usize,
    pub prob:f32,
}
impl DNA {
    pub fn new(dna:String) -> DNA {
        DNA {
            dna,
            fitness: 0,
            prob: 0.0,
        }
    }
    
    pub fn calc_fitness(&mut self, target:&String) -> usize {
        let mut fitness:usize = 0;
        let mut chars = self.dna.chars();
        let mut target_chars = target.chars();

        while let Some(c) = chars.next() {
            if c == target_chars.next().unwrap() {
                fitness += 1;
            }
        }
        self.fitness = fitness.pow(2);
        self.fitness
    }

}

pub fn crossover(parent1:&DNA, parent2:&DNA, mutation_rate:f32) -> DNA {
    let mut child_dna = String::new();
    let length = parent1.dna.len();

    let mut parent1 = parent1.dna.chars();
    let mut parent2 = parent2.dna.chars();
    let mut p1char:char;
    let mut p2char:char;
    let mut rand_x:f32;
    for _i in 0..length {
        p1char = parent1.next().unwrap();
        p2char = parent2.next().unwrap();
        rand_x = rand::random::<f32>();
        if rand_x <= mutation_rate {
            child_dna.push(random_char());
        }
        else if rand::random::<bool>() == true {
            child_dna.push(p1char);
        } else {
            child_dna.push(p2char);
        }
    }
    DNA::new(child_dna)
} 

pub fn random_char() -> char {
    let mut res = rand::thread_rng().gen_range(33..127) as u8 as char;
    if res == '{' { res = ' '; }
    res
}

