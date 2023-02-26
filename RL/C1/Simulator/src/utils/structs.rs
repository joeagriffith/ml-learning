use std::collections::HashMap;
use crate::units::MobileUnit;

#[derive(Debug)]
pub struct MobList {
    hashmap: HashMap<usize, MobileUnit>,
    innov: usize,
}

impl MobList {
    pub fn new() -> Self {
        Self {
            hashmap: HashMap::new(),
            innov: 0,
        }
    }

    pub fn insert(&mut self, mob:MobileUnit) -> usize {
        self.hashmap.insert(self.innov, mob);
        self.innov += 1;
        self.innov - 1
    }

    pub fn get(&self, innov:&usize) -> &MobileUnit {
        self.hashmap.get(innov).unwrap()
    }

    pub fn get_mut(&mut self, innov:&usize) -> &mut MobileUnit {
        self.hashmap.get_mut(innov).unwrap()
    }

    pub fn remove(&mut self, innov:&usize) {
        self.hashmap.remove(innov);
    }

    pub fn get_mobs(&self) -> std::collections::hash_map::Values<usize, MobileUnit> {
        self.hashmap.values()
    }

    pub fn get_mobs_mut(&mut self) -> std::collections::hash_map::ValuesMut<usize, MobileUnit> {
        self.hashmap.values_mut()
    }

    pub fn get_guids(&self) -> std::collections::hash_map::Keys<usize, MobileUnit> {
        self.hashmap.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.hashmap.is_empty()
    }

    pub fn clear(&mut self) {
        self.hashmap.clear();
        self.innov = 0;
    }

    pub fn iter_mut(&mut self) -> std::collections::hash_map::IterMut<usize, MobileUnit> {
        self.hashmap.iter_mut()
    }

}
