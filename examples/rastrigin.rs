// Simple example how to use the API.
extern crate differential_evolution;

use differential_evolution::self_adaptive_de;
use std::f32::consts::PI;
use std::env;

// The Rastrigin function is a non-convex function used as a
// performance test problem for optimization algorithms.
// see https://en.wikipedia.org/wiki/Rastrigin_function 
fn rastrigin(pos: &[f32]) -> f32 {
    pos.iter().fold(0.0, |sum, x| 
        sum + x * x - 10.0 * (2.0 * PI * x).cos() + 10.0)
}

fn main() {
    // command line args: dimension, number of evaluations
    let args: Vec<String> = env::args().collect();
    let dim = args[1].parse::<usize>().unwrap();
    let num_cost_evaluations = args[2].parse::<usize>().unwrap();

    // initial search space for each dimension
    let initial_min_max = vec![(-5.12, 5.12); dim];

    // perform optimization 
    let mut de = self_adaptive_de(initial_min_max, rastrigin);
    de.nth(num_cost_evaluations);

    // see what we've found
    println!("{} evaluations done", num_cost_evaluations);
    
    let (cost, pos) = de.best().unwrap();
    println!("{} best cost", cost);
    println!("{:?} best position", pos);
}