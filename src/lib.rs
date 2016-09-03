extern crate rand;

pub struct Settings<'a> {
    pub min_pos: Vec<f32>,
    pub max_pos: Vec<f32>,

    pub cr_min: f32,
    pub cr_max: f32,
    pub cr_change_probability: f32,

    pub f_min: f32,
    pub f_max: f32,
    pub f_change_probability: f32,

    pub pop_size: usize,
    pub rng: &'a mut rand::Rng,
}

#[derive(Clone)]
pub struct Individual {
    pos: Vec<f32>,
    fitness: f32,
}

pub struct Population<'a> {
    pop_curr: Vec<Individual>,
    pop_best: Vec<Individual>,
    settings: Settings<'a>,
}

impl<'a> Population<'a> {
    pub fn new(s: Settings) -> Population {

        let empty_individual = Individual {
            pos: Vec::new(),
            fitness: 0.0,
        };

        // creates all the empty individuals
        let pop = Population {
            pop_curr: vec![empty_individual.clone(); s.pop_size],
            pop_best: vec![empty_individual; s.pop_size],
            settings: s,
        };

        pop
    }
}
