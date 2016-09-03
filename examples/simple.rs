// Simple example how to use the API.
extern crate differential_evolution;
extern crate rand;

use differential_evolution::{Settings, Population};
use rand::thread_rng;

fn square_fitness(pos: &Vec<f32>) -> f32 {
    let mut f = 0.0;
    for x in pos {
        f += x * x;
    }
    f
}

fn main() {
    println!("Hello from differential evolution!");

    // problem dimension
    let dim = 5;

    // create settings for the algorithm
    let settings = Settings {
        min_pos: vec![-20.0; dim],
        max_pos: vec![20.0; dim],

        cr_min: 0.0,
        cr_max: 1.0,
        cr_change_probability: 0.1,

        f_min: 0.1,
        f_max: 1.0,
        f_change_probability: 0.1,

        pop_size: 50,
        rng: &mut rand::thread_rng(),
    };

    // create population
    let pop = Population::new(settings);
}