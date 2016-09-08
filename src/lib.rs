#![feature(test)]

#![feature(plugin)]
#![plugin(clippy)]

extern crate test;
extern crate rand;

use rand::distributions::{IndependentSample, Range};

pub struct Settings<F, R>
    where F: Fn(&[f32]) -> f32,
          R: rand::Rng
{
    pub min_max_pos: Vec<(f32, f32)>,

    pub cr_min: f32,
    pub cr_max: f32,
    pub cr_change_probability: f32,

    pub f_min: f32,
    pub f_max: f32,
    pub f_change_probability: f32,

    pub pop_size: usize,
    pub rng: R,

    pub cost_function: F,
}

impl<F, R> Settings<F, R>
    where F: Fn(&[f32]) -> f32,
          R: rand::Rng
{
    pub fn min_max_rng(min_max_pos: Vec<(f32, f32)>, cost_function: F, rng: R) -> Settings<F, R> {
        // create settings for the algorithm
        Settings {
            min_max_pos: min_max_pos,

            cr_min: 0.0,
            cr_max: 1.0,
            cr_change_probability: 0.1,

            f_min: 0.1,
            f_max: 1.0,
            f_change_probability: 0.1,

            pop_size: 50,
            rng: rng,

            cost_function: cost_function,
        }
    }
}

impl<F> Settings<F, rand::XorShiftRng>
    where F: Fn(&[f32]) -> f32
{
    pub fn new(min_max_pos: Vec<(f32, f32)>, cost_function: F) -> Settings<F, rand::XorShiftRng> {
        Settings::min_max_rng(min_max_pos, cost_function, rand::weak_rng())
    }
}

#[derive(Clone,Debug)]
pub struct Individual {
    pos: Vec<f32>,
    // the lower, the better.
    cost: Option<f32>,

    // control parameters
    cr: f32,
    f: f32,
}

pub struct Population<F, R>
    where F: Fn(&[f32]) -> f32,
          R: rand::Rng
{
    // TODO use a single vector for curr and best and controlparameters?
    curr: Vec<Individual>,
    best: Vec<Individual>,

    settings: Settings<F, R>,

    // index of global best individual. Might be in best or in curr.
    best_idx: Option<usize>,

    // cost value of the global best individual, for quick access
    best_cost_cache: Option<f32>,
    dim: usize,
    between_popsize: Range<usize>,
    between_dim: Range<usize>,
    between_cr: Range<f32>,
    between_f: Range<f32>,

    pop_countdown: usize,
}

impl<F, R> Iterator for Population<F, R>
    where F: Fn(&[f32]) -> f32,
          R: rand::Rng
{
    type Item = f32;

    /// Returns the cost value of the current best
    fn next(&mut self) -> Option<f32> {
        if 0 == self.pop_countdown {
            // if the whole pop has been evaluated, evolve it to update positions.
            // this also copies curr to best, if better.
            self.update_best();
            self.update_positions();
            self.pop_countdown = self.curr.len();
        }

        // perform a single fitness evaluation
        self.pop_countdown -= 1;
        let curr = &mut self.curr[self.pop_countdown];

        let cost = (self.settings.cost_function)(&curr.pos);
        curr.cost = Some(cost);

        // see if we have improved the global best
        if self.best_cost_cache.is_none() || cost < self.best_cost_cache.unwrap() {
            self.best_cost_cache = Some(cost);
            self.best_idx = Some(self.pop_countdown);
        }

        self.best_cost_cache
    }
}

impl<F> Population<F, rand::XorShiftRng>
    where F: Fn(&[f32]) -> f32
{
    pub fn new(min_max_pos: Vec<(f32, f32)>, cost_function: F) -> Population<F, rand::XorShiftRng> {
        Population::from_settings(Settings::new(min_max_pos, cost_function))
    }
}

impl<F, R> Population<F, R>
    where F: Fn(&[f32]) -> f32,
          R: rand::Rng
{
    // Creates a new population based on the given settings.
    pub fn from_settings(s: Settings<F, R>) -> Population<F, R> {
        assert!(s.min_max_pos.len() >= 1,
                "need at least one element to optimize");

        // create a vector of randomly initialized individuals for current.
        let dim = s.min_max_pos.len();

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
            best_cost_cache: None,
            dim: dim,
            pop_countdown: s.pop_size,
            between_popsize: Range::new(0, s.pop_size),
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
                let between_min_max = Range::new(pop.settings.min_max_pos[d].0,
                                                 pop.settings.min_max_pos[d].1);
                ind.pos[d] = between_min_max.ind_sample(&mut pop.settings.rng);
            }
        }

        pop
    }

    fn update_best(&mut self) {
        for i in 0..self.curr.len() {
            let curr = &mut self.curr[i];
            let best = &mut self.best[i];

            // we use <= here, so that the individual moves even if the cost
            // stays the same.
            if best.cost.is_none() || curr.cost.unwrap() <= best.cost.unwrap() {
                // replace individual's best. swap is *much* faster than clone.
                std::mem::swap(curr, best);
            }
        }
    }

    fn update_positions(&mut self) {
        let global_best_pos = &self.best[self.best_idx.unwrap()].pos;
        let rng = &mut self.settings.rng;
        for i in 0..self.curr.len() {
            let mut id1 = self.between_popsize.ind_sample(rng);
            while id1 == i {
                id1 = self.between_popsize.ind_sample(rng);
            }

            let mut id2 = self.between_popsize.ind_sample(rng);
            while id2 == i || id2 == id1 {
                id2 = self.between_popsize.ind_sample(rng);
            }

            let curr = &mut self.curr[i];
            let best = &self.best[i];

            // see "Self-Adapting Control Parameters in Differential Evolution:
            // A Comparative Study on Numerical Benchmark Problems"
            if rng.gen::<f32>() < self.settings.cr_change_probability {
                curr.cr = self.between_cr.ind_sample(rng);
            } else {
                curr.cr = best.cr;
            }
            if rng.gen::<f32>() < self.settings.f_change_probability {
                curr.f = self.between_f.ind_sample(rng);
            } else {
                curr.f = best.f;
            }

            let curr_pos = &mut curr.pos;
            let best_pos = &best.pos;
            let best1_pos = &self.best[id1].pos;
            let best2_pos = &self.best[id2].pos;

            let forced_mutation_dim = self.between_dim.ind_sample(rng);

            // This implements the DE/1/best/bin algorithm.
            for d in 0..self.dim {
                if d == forced_mutation_dim || rng.gen::<f32>() < curr.cr {
                    curr_pos[d] = global_best_pos[d] + curr.f * (best1_pos[d] - best2_pos[d]);
                } else {
                    curr_pos[d] = best_pos[d];
                }
            }

            // reset cost, has to be updated by the user.
            curr.cost = None;
        }
    }

    pub fn best(&self) -> Option<(f32, &[f32])> {
        if let Some(bi) = self.best_idx {
            let curr = &self.curr[bi];
            let best = &self.best[bi];

            if curr.cost.is_none() {
                return Some((best.cost.unwrap(), &best.pos));
            }
            if best.cost.is_none() {
                return Some((curr.cost.unwrap(), &curr.pos));
            }
            if curr.cost.unwrap() < best.cost.unwrap() {
                return Some((curr.cost.unwrap(), &curr.pos));
            }
            return Some((best.cost.unwrap(), &best.pos));
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate rand;
    use super::*;
    use test::Bencher;
    use rand::{XorShiftRng, StdRng, IsaacRng, Isaac64Rng, Rng, ChaChaRng};
    use rand::{OsRng, weak_rng};


    fn setup<F: Fn(&[f32]) -> f32, R: rand::Rng>(dim: usize, cost_fn: F, rng: R) -> Population<F, R> {
        let s = Settings::min_max_rng(vec![(-100.0, 100.0); dim], cost_fn, rng);
        Population::from_settings(s)
    }

    #[bench]
    fn bench_square_fitness_opt(b: &mut Bencher) {
        let rng: XorShiftRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }


    #[bench]
    fn rand_thread_rng(b: &mut Bencher) {
        let mut pop = setup(5, |_| 1.234, rand::thread_rng());
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_xor_shift(b: &mut Bencher) {
        let rng: XorShiftRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_chacha(b: &mut Bencher) {
        let rng: ChaChaRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_isaac(b: &mut Bencher) {
        let rng: IsaacRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_isaac64(b: &mut Bencher) {
        let rng: Isaac64Rng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_std(b: &mut Bencher) {
        let rng: StdRng = StdRng::new().unwrap();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_osrng(b: &mut Bencher) {
        let rng: OsRng = OsRng::new().unwrap();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_weak_rng(b: &mut Bencher) {
        let rng = weak_rng();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }
}
