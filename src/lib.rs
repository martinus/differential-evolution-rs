#![feature(plugin)]
#![plugin(clippy)]

extern crate rand;

use rand::distributions::{IndependentSample, Range};
// use rand::distributions::range::SampleRange;

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

#[derive(Clone,Debug)]
pub struct Individual {
    pub pos: Vec<f32>,
    // the lower, the better.
    pub cost: Option<f32>,

    // control parameters
    cr: f32,
    f: f32,
}

pub struct Population<R>
    where R: rand::Rng
{
    // TODO use a single vector for curr and best and controlparameters?
    pub curr: Vec<Individual>,
    best: Vec<Individual>,

    settings: Settings<R>,
    best_idx: Option<usize>,
    dim: usize,
    between_popsize: Range<usize>,
    between_01: Range<f32>,
    between_dim: Range<usize>,
    between_cr: Range<f32>,
    between_f: Range<f32>,
}

impl<R> Population<R>
    where R: rand::Rng
{
    // Creates a new population based on the given settings.
    pub fn new(s: Settings<R>) -> Population<R> {
        assert_eq!(s.min_pos.len(),
                   s.max_pos.len(),
                   "min_pos and max_pos need to have the same number of elements");
        assert!(s.min_pos.len() >= 1,
                "need at least one element to optimize");


        // create a vector of randomly initialized individuals for current.
        let dim = s.min_pos.len();

        // Empty individual, with no cost value (yet)
        let dummy_individual = Individual {
            pos: vec![0.0; dim],
            cost: None,
            cr: 0.0,
            f: 0.0,
        };

        // creates all the empty individuals
        let mut pop = Population {
            curr: vec![dummy_individual.clone(); s.pop_size],
            best: vec![dummy_individual; s.pop_size],
            best_idx: None,
            dim: dim,
            between_popsize: Range::new(0, s.pop_size),
            between_01: Range::new(0.0, 1.0),
            between_dim: Range::new(0, dim),
            between_cr: Range::new(s.cr_min, s.cr_max),
            between_f: Range::new(s.f_min, s.f_max),
            settings: s,
        };

        for ind in &mut pop.curr {
            // init control parameters
            ind.cr = pop.between_cr.ind_sample(&mut pop.settings.rng);
            ind.f = pop.between_f.ind_sample(&mut pop.settings.rng);

            // random range for each dimension
            for d in 0..dim {
                let between_min_max = Range::new(pop.settings.min_pos[d], pop.settings.max_pos[d]);
                ind.pos[d] = between_min_max.ind_sample(&mut pop.settings.rng);
            }
        }

        pop
    }

    fn update_best(&mut self) -> bool {
        let last_best_cost: Option<f32>;
        if let Some(last_best_idx) = self.best_idx {
            last_best_cost = self.best[last_best_idx].cost;
        } else {
            last_best_cost = None;
        }

        let mut new_best_idx = 0;
        for i in 0..self.curr.len() {
            let cost_curr = self.curr[i].cost.unwrap();
            if let Some(cost_best) = self.best[i].cost {
                // if we already have a best, check if current is better.
                if cost_curr <= cost_best {
                    // TODO don't clone but swap?
                    self.best[i] = self.curr[i].clone();
                }
            } else {
                // no best yet, overwrite with current
                self.best[i] = self.curr[i].clone();
            }

            // min best cost index
            if self.best[i].cost.unwrap() < self.best[new_best_idx].cost.unwrap() {
                new_best_idx = i;
            }
        }
        self.best_idx = Some(new_best_idx);

        // got a new best?
        last_best_cost.is_none() || self.best[new_best_idx].cost.unwrap() < last_best_cost.unwrap()
    }

    fn update_positions(&mut self) {
        let best_idx = self.best_idx.unwrap();
        for i in 0..self.curr.len() {
            let mut id1 = i;
            while id1 == i {
                id1 = self.between_popsize.ind_sample(&mut self.settings.rng);
            }

            let mut id2 = i;
            while id2 == i || id2 == id1 {
                id2 = self.between_popsize.ind_sample(&mut self.settings.rng);
            }

            let curr = &mut self.curr[i];
            let best = &self.best[i];

            // see "Self-Adapting Control Parameters in Differential Evolution:
            // A Comparative Study on Numerical Benchmark Problems"
            curr.cr = best.cr;
            curr.f = best.f;
            if self.between_01.ind_sample(&mut self.settings.rng) <
               self.settings.cr_change_probability {
                curr.cr = self.between_cr.ind_sample(&mut self.settings.rng);
            }
            if self.between_01.ind_sample(&mut self.settings.rng) <
               self.settings.f_change_probability {
                curr.f = self.between_f.ind_sample(&mut self.settings.rng);
            }

            let forced_mutation_dim = self.between_dim.ind_sample(&mut self.settings.rng);
            for d in 0..self.dim {
                if d == forced_mutation_dim ||
                   self.between_01.ind_sample(&mut self.settings.rng) < curr.cr {

                    curr.pos[d] = self.best[best_idx].pos[d] +
                                  curr.f * (self.best[id1].pos[d] - self.best[id2].pos[d]);
                } else {
                    curr.pos[d] = self.best[i].pos[d];
                }
            }
        }
    }

    // Uses updated cost values to update positions of individuals.
    pub fn evolve(&mut self) -> Option<&Individual> {
        let found_new_best = self.update_best();
        self.update_positions();

        if found_new_best {
            Some(&self.best[self.best_idx.unwrap()])
        } else {
            None
        }
    }
}
