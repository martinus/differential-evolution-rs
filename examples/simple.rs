// Simple example how to use the API.
extern crate differential_evolution;

use differential_evolution::Population;

fn main() {
    // problem dimension
    let dim = 10;

    // create population with default settings:
    let mut pop = Population::new(vec![(-100.0, 100.0); dim], |pos| {
        // sum of squares
        let mut f = 0.0;
        for x in pos {
            f += x*x;
        }
        f
    });

    // perform 10000 cost evaluations
    pop.nth(10000);

    // see what we've found
    println!("best: {:?}", pop.best());
}