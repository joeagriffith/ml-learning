use std::collections::HashSet;
use rand::prelude::*;

use crate::genetics::Gene;

//Isn't this essentially a HashMap??
#[derive(Clone)]
pub struct VecSet<T> {
    vec: Vec<T>, // Store NodeGenes & ConnectionGenes
    hashset: HashSet<usize>, // Store Innovation Numbers, to prevent duplicates in vec
}

impl<T> VecSet<T> where T: Gene + Clone {

    pub fn new() -> Self {
        Self {
            hashset: HashSet::new(),
            vec: Vec::new(),
        }
    }

    pub fn push(&mut self, gene:T) -> bool {
        if self.hashset.insert(gene.get_innov()) {
            self.vec.push(gene);
            return true
        } 
        return false
    }

    pub fn insert_sorted(&mut self, gene:T) {

        let innov = gene.get_innov();
        if self.hashset.insert(innov) {

            let mut index = self.vec.len() as isize - 1;

            while index >= 0 && self.vec[index as usize].get_innov() > innov {
                index -= 1;
            }

            self.vec.insert((index+1) as usize, gene);
        } 
    }

    pub fn remove(&mut self, innov:usize) {
        if self.hashset.remove(&innov) {
            let index = self.find_innov_index(innov);
            self.vec.remove(index);
        } else {
            panic!("Attempted to remove item which does not exist in VecSet.");
        }
    }

    // Binary search vector for item with innovation_number = innov
    fn find_innov_index(&self, innov:usize) -> usize {
        let mut i_upper = self.vec.len();
        let mut i_lower:usize = 0;
        let mut index = i_upper / 2;
        while self.vec[index].get_innov() != innov {
            if self.vec[index].get_innov() > innov {
                i_upper = index;
            } else {
                i_lower = index;
            }
            let new_index = (i_upper + i_lower) / 2;
            if new_index == index {
                panic!("Binary search for item in VecSet Failed.");
            } else {
                index = new_index;
            }
        }
        index
    }

    pub fn get_by_innov(&self, innov:usize) -> &T {
        let index = self.find_innov_index(innov);
        &self.vec[index]
    }

    pub fn get(&self, index:usize) -> &T {
        &self.vec[index]
    }

    // pub fn at(&self, index:usize) -> T {
    //     self.vec[index].clone()
    // }

    pub fn rand_innov(&self) -> usize {
        let index = rand::thread_rng().gen_range(0..self.vec.len());
        self.vec[index].get_innov()
    }

    // pub fn rand_element(&self) -> &T {
    //     let index = rand::thread_rng().gen_range(0..self.vec.len());
    //     self.vec.get(index).unwrap()
    // }

    pub fn rand_element_mut(&mut self) -> &mut T {
        let index = rand::thread_rng().gen_range(0..self.vec.len());
        self.vec.get_mut(index).unwrap()
    }
    
    pub fn contains_innov(&self, inno:usize) -> bool {
        self.hashset.contains(&inno)
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.vec.iter()
    }

    pub fn max_innov(&self) -> Option<usize> {
        if self.vec.len() == 0 {
            None
        } else {
            Some(self.vec[self.vec.len() - 1].get_innov())
        }
    }


}
