extern crate rand;

pub struct Settings<R>
    where R: rand::Rng
{
    pub min_pos: Vec<f32>,
    pub max_pos: Vec<f32>,

    pub cr_min: f32,
    pub cr_max: f32,
    pub cr_change_probability: f32,

    pub f_min: f32,
    pub f_max: f32,
    pub f_change_probability: f32,

    pub pop_size: usize,
    pub rng: R,
}

#[derive(Clone)]
pub struct Individual {
    pos: Vec<f32>,
    fitness: f32,
}

pub struct Population<R>
    where R: rand::Rng
{
    pop_curr: Vec<Individual>,
    pop_best: Vec<Individual>,
    settings: Settings<R>,
}

impl<R> Population<R>
    where R: rand::Rng
{
    pub fn new(s: Settings<R>) -> Population<R> {

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
