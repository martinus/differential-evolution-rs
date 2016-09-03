#[derive(Debug)]
pub struct Settings {
    pub min_pos: Vec<f32>,
    pub max_pos: Vec<f32>,

    pub cr_min: f32,
    pub cr_max: f32,
    pub cr_change_probability: f32,

    pub f_min: f32,
    pub f_max: f32,
    pub f_change_probability: f32,

    pub pop_size: usize,
}

#[derive(Clone,Debug)]
pub struct Individual {
    pos: Vec<f32>,
    fitness: f32,
}

#[derive(Debug)]
pub struct Population {
    settings: Settings,
    pop_curr: Vec<Individual>,
    pop_best: Vec<Individual>,
}

impl Population {
    pub fn new(s: Settings) -> Population {
        let empty_individual = Individual {
            pos: Vec::new(),
            fitness: 0.0,
        };
        let pop = Population {
            pop_curr: vec![empty_individual.clone(); s.pop_size],
            pop_best: vec![empty_individual; s.pop_size],
            settings: s,
        };
        pop
    }
}
