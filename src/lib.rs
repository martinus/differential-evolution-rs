// Copyright 2016 Martin Ankerl.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Differential Evolution optimizer for rust.
//!
//! Simple and powerful global optimization using a
//! [Self-Adapting Differential Evolution](http://bit.ly/2cMPiMj)
//! for Rust. See Wikipedia's article on
//! [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution)
//! for more information.
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! differential-evolution = "*"
//! ```
//!
//! and this to your crate root:
//!
//! ```rust
//! extern crate differential_evolution;
//! ```
//!
//! ## Examples
//!
//! Differential Evolution is a global optimization algorithm that
//! tries to iteratively improve candidate solutions with regards to
//! a user-defined cost function.
//!
//! ### Sum of Squares
//! This example finds the minimum of a simple 5-dimensional function.
//!
//! ```
//! extern crate differential_evolution;
//!
//! use differential_evolution::self_adaptive_de;
//!
//! fn main() {
//!     // create a self adaptive DE with an inital search area
//!     // from -10 to 10 in 5 dimensions.
//!     let mut de = self_adaptive_de(vec![(-10.0, 10.0); 5], |pos| {
//!         // cost function to minimize: sum of squares
//!         pos.iter().fold(0.0, |sum, x| sum + x*x)
//!     });
//!
//!     // perform 10000 cost evaluations
//!     de.nth(10000);
//!
//!     // show the result
//!     let (cost, pos) = de.best().unwrap();
//!     println!("cost: {}", cost);
//!     println!("pos: {:?}", pos);
//! }
//! ```
//!
//! # Similar Crates
//!
//! - [darwin-rs](https://github.com/willi-kappler/darwin-rs)
//! - [RsGenetic](https://github.com/m-decoster/RsGenetic)
//!

extern crate rand;

use rand::distributions::{IndependentSample, Range};

/// Holds all settings for the self adaptive differential evolution
/// algorithm.
pub struct Settings<F, R>
    where F: Fn(&[f32]) -> f32,
          R: rand::Rng
{
    /// The population is initialized with uniform random
    /// for each dimension between the tuple's size.
    /// Beware that this is only the initial state, the DE
    /// will search outside of this initial search space.
    pub min_max_pos: Vec<(f32, f32)>,

    /// Minimum and maximum value for `cr`, the crossover control parameter.
    /// a good value is (0, 1) so cr is randomly choosen between in the full
    /// range of usable CR's from `[0, 1)`.
    pub cr_min_max: (f32, f32),

    /// Probability to change the `cr` value of an individual. Tests with
    /// 0.05, 0.1, 0.2 and 0.3 did not show any significant different
    /// results. So 0.1 seems to be a reasonable choice.
    pub cr_change_probability: f32,

    /// Minimum and maximum value for `f`, the amplification factor of the
    /// difference vector. DE is more sensitive to `F` than it is to `CR`.
    /// In literature, `F` is rarely greater than 1. If `F=0`, the evolution
    /// degenerates to a crossover but no mutation, so a reasonable choise
    /// for f_min_max seems to be (0.1, 1.0).
    pub f_min_max: (f32, f32),

    /// Probability to change the `f` value of an individual. See
    /// `cr_change_probability`, 0.1 is a reasonable choice.
    pub f_change_probability: f32,

    /// Number of individuals for the DE. In many benchmarks, a size of
    /// 100 is used. The choice somewhat depends on the difficulty and the
    /// dimensionality of the  problem to solve. Reasonable choices seem
    /// between 20 and 200.
    pub pop_size: usize,

    /// Random number generator used to generate mutations. If the fitness
    /// function is fairly fast, the random number generator should be
    /// very fast as well. Since it is not necessary to use a cryptographic
    /// secure RNG, the best (fastest) choice is to use `rand::weak_rng()`.
    pub rng: R,

    /// The cost function to minimize. This takes an `&[f32]` and returns
    /// the calculated cost for this position as `f32`. This should be
    /// fast to evaluate, and always produce the same result for the same
    /// input.
    pub cost_function: F,
}

impl<F> Settings<F, rand::XorShiftRng>
    where F: Fn(&[f32]) -> f32
{
    /// Creates default settings for the differential evolution. It uses the default
    /// parameters as defined in the paper "Self-Adapting Control Parameters in Differential
    /// Evolution: A Comparative Study on Numerical Benchmark Problems", with a population
    /// size of 100. It also uses This uses `rand::weak_rng()` for the fastest random number
    /// generator available.
    ///
    /// For most problems this should be a fairly good parameter set.
    pub fn default(min_max_pos: Vec<(f32, f32)>,
                   cost_function: F)
                   -> Settings<F, rand::XorShiftRng> {
        Settings {
            min_max_pos: min_max_pos,

            cr_min_max: (0.0, 1.0),
            cr_change_probability: 0.1,

            f_min_max: (0.1, 1.0),
            f_change_probability: 0.1,

            pop_size: 100,
            rng: rand::weak_rng(),

            cost_function: cost_function,
        }
    }
}

#[derive(Clone)]
struct Individual {
    pos: Vec<f32>,
    // the lower, the better.
    cost: Option<f32>,

    // control parameters
    cr: f32,
    f: f32,
}

/// Holds the population for the differential evolution based on the given settings.
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
    num_cost_evaluations: usize,

    dim: usize,
    between_popsize: Range<usize>,
    between_dim: Range<usize>,
    between_cr: Range<f32>,
    between_f: Range<f32>,

    pop_countdown: usize,
}

/// The population inplements the `Iterator` trait, and for each call
/// of `next()` the cost function is evaluated once and returns the
/// fitness value of the current global best. This way it is possible
/// to use all the iterator's features for optimizig. Here are a few
/// examples.
///
/// Let's say we have a simple cost function that calculates sum
/// of squares:
///
/// ```
/// fn sum_of_squares(pos: &[f32]) -> f32 {
///     pos.iter().fold(0.0, |sum, x| sum + x*x)
/// }
/// ```
/// 
/// We'd like to search for the minimum in the range -5 to 5, for
/// 10 dimensions:
/// 
/// ```
/// let initial_min_max = vec![(-5.0, 5.0); 10];
/// ```
////
/// We can create a self adaptive DE, and search until the cost
/// reaches a given minimum: 
/// 
/// ```
/// # use differential_evolution::self_adaptive_de;
/// # fn sum_of_squares(pos: &[f32]) -> f32 { pos.iter().fold(0.0, |sum, x| sum + x*x) }
/// # let initial_min_max = vec![(-5.0, 5.0); 10];
/// let mut de = self_adaptive_de(initial_min_max, sum_of_squares);
/// de.find(|&cost| cost < 0.1);    
/// ```
/// 
/// This is a bit dangerous though, because the optimizer might never reach that minimum.
/// It is safer to just let it run for a given number of evaluations:
///
/// ```
/// # use differential_evolution::self_adaptive_de;
/// # fn sum_of_squares(pos: &[f32]) -> f32 { pos.iter().fold(0.0, |sum, x| sum + x*x) }
/// # let initial_min_max = vec![(-5.0, 5.0); 10];
/// let mut de = self_adaptive_de(initial_min_max, sum_of_squares);
/// de.nth(10000);
/// ```
/// 
/// Of course it is possible to combine both: run until cost is below a threshold, or until
/// the maximum number of iterations have been reached:
///
/// # use differential_evolution::self_adaptive_de;
/// # fn sum_of_squares(pos: &[f32]) -> f32 { pos.iter().fold(0.0, |sum, x| sum + x*x) }
/// # let initial_min_max = vec![(-5.0, 5.0); 10];
/// let mut de = self_adaptive_de(initial_min_max, sum_of_squares);
/// de.nth(10000);
/// 
impl<F, R> Iterator for Population<F, R>
    where F: Fn(&[f32]) -> f32,
          R: rand::Rng
{
    /// A tuple of current best cost and number of cost function evaluations so far.
    type Item = (f32, usize);

    /// Returns the cost value of the current best solution found.
    fn next(&mut self) -> Option<Self::Item> {
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
        self.num_cost_evaluations += 1;

        // see if we have improved the global best
        if self.best_cost_cache.is_none() || cost < self.best_cost_cache.unwrap() {
            self.best_cost_cache = Some(cost);
            self.best_idx = Some(self.pop_countdown);
        }

        Some((self.best_cost_cache.unwrap(), self.num_cost_evaluations))
    }
}


pub fn self_adaptive_de<F>(min_max_pos: Vec<(f32, f32)>,
                           cost_function: F)
                           -> Population<F, rand::XorShiftRng>
    where F: Fn(&[f32]) -> f32
{
    Population::new(Settings::default(min_max_pos, cost_function))
}

impl<F, R> Population<F, R>
    where F: Fn(&[f32]) -> f32,
          R: rand::Rng
{
    // Creates a new population based on the given settings.
    pub fn new(s: Settings<F, R>) -> Population<F, R> {
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
            num_cost_evaluations: 0,
            dim: dim,
            pop_countdown: s.pop_size,
            between_popsize: Range::new(0, s.pop_size),
            between_dim: Range::new(0, dim),
            between_cr: Range::new(s.cr_min_max.0, s.cr_min_max.1),
            between_f: Range::new(s.f_min_max.0, s.f_min_max.1),
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

    // Modifies all the curr positions. This needs a lot of random numbers, so
    // for a fast cost function it is important to use a fast random number
    // generator.
    fn update_positions(&mut self) {
        let rng = &mut self.settings.rng;
        for i in 0..self.curr.len() {
            // sample 3 different individuals
            let id1 = self.between_popsize.ind_sample(rng);

            let mut id2 = self.between_popsize.ind_sample(rng);
            while id2 == id1 {
                id2 = self.between_popsize.ind_sample(rng);
            }

            let mut id3 = self.between_popsize.ind_sample(rng);
            while id3 == id1 || id3 == id2 {
                id3 = self.between_popsize.ind_sample(rng);
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
            let best3_pos = &self.best[id3].pos;

            let forced_mutation_dim = self.between_dim.ind_sample(rng);

            // This implements the DE/rand/1/bin, the most widely used algorithm.
            // See "A Comparative Study of Differential Evolution Variants for
            // Global Optimization (2006)".
            for d in 0..self.dim {
                if d == forced_mutation_dim || rng.gen::<f32>() < curr.cr {
                    curr_pos[d] = best3_pos[d] + curr.f * (best1_pos[d] - best2_pos[d]);
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

    /// Gets the total number of times the cost function has been evaluated.
    pub fn num_cost_evaluations(&self) -> usize {
        self.num_cost_evaluations
    }
}

#[cfg(test)]
mod tests {
    // TODO
}
