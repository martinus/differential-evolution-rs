// Simple example how to use the API.
extern crate differential_evolution;

use differential_evolution::Population;

fn main() {
    // problem dimension
    let dim = 10;

    // initial search space for each dimension
    let initial_min_max = vec![(-100.0, 100.0); dim];

    // create population with default settings:
    let mut pop = Population::new(initial_min_max, |pos| {
        // cost function to minimize: sum of squares
        pos.iter().fold(0.0, |sum, x| sum + x*x)
    });

    // perform 10000 cost evaluations
    pop.nth(10000);

    // see what we've found
    println!("best: {:?}", pop.best());
}