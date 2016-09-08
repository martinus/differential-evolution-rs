// Simple example how to use the API.
extern crate differential_evolution;

use differential_evolution::Population;
use std::f32::consts::PI;
use std::env;

fn rastrigin(pos: &[f32]) -> f32 {
    10.0 * (pos.len() as f32) +
    pos.iter().fold(0.0, |sum, x| sum + x * x - 10.0 * (2.0 * PI * x).cos())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // problem dimension
    let dim = args[1].parse::<usize>().unwrap();

    // number of cost evaluations
    let num_cost_evaluations = args[2].parse::<usize>().unwrap();

    // initial search space for each dimension
    let initial_min_max = vec![(-5.12, 5.12); dim];

    // perform optimization 
    let mut pop = Population::new(initial_min_max, rastrigin);
    pop.nth(num_cost_evaluations);

    // see what we've found
    println!("{} evaluations done", num_cost_evaluations);
    
    let (cost, pos) = pop.best().unwrap();
    println!("{} best cost", cost);
    println!("{:?} best position", pos);
}