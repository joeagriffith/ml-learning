// Unit configs for different unit types
pub enum UnitType {
    Structure,
    Mobile,
}


pub struct Guid {
    id: u32,
    unit_type: UnitType,
}

impl Guid {
    // pub fn new()
}