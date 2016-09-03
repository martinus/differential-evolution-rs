extern crate rand;

use rand::distributions::{IndependentSample, Range};

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
    pub pos: Vec<f32>,
    pub fitness: f32,
}

pub struct Population<R>
    where R: rand::Rng
{
    pub curr: Vec<Individual>,
    pub best: Vec<Individual>,
    settings: Settings<R>,
}


impl<R> Population<R>
    where R: rand::Rng
{
    pub fn new(s: Settings<R>) -> Population<R> {
        assert_eq!(s.min_pos.len(),
                   s.max_pos.len(),
                   "min_pos and max_pos need to have the same number of elements");
        assert!(s.min_pos.len() >= 1,
                "need at least one element to optimize");


        // create a vector of randomly initialized individuals for current.
        let dim = s.min_pos.len();

        let empty_individual = Individual {
            pos: vec![0.0; dim],
            fitness: 0.0,
        };

        // creates all the empty individuals
        let mut pop = Population {
            curr: vec![empty_individual.clone(); s.pop_size],
            best: vec![empty_individual; s.pop_size],
            settings: s,
        };

        // random range for each dimension
        for d in 0..dim {
            let between = Range::new(pop.settings.min_pos[d], pop.settings.max_pos[d]);

            // initialize each individual's dimension
            for ind in &mut pop.curr {
                ind.pos[d] = between.ind_sample(&mut pop.settings.rng);
            }
        }

        pop
    }
}
