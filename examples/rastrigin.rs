// Copyright 2016 Martin Ankerl. 
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

    // initial search space for each dimension
    let initial_min_max = vec![(-5.12, 5.12); dim];

    // perform optimization until best cost is below a threshold
    let mut de = self_adaptive_de(initial_min_max, rastrigin);
    for (i, item) in de.iter().enumerate() {
        println!("{:?} {:?}", i, item);
    }
    //de.find(|&(cost, iters)| cost < 0.1 || iters >= 100000);

    // see what we've found
    println!("{} evaluations done", de.num_cost_evaluations());
    
    let (cost, pos) = de.best().unwrap();
    println!("{} best cost", cost);
    println!("{:?} best position", pos);
}