// Simple example how to use the API.
extern crate differential_evolution;
extern crate rand;

use differential_evolution::{Settings, Population};

fn square_fitness(pos: &Vec<f32>) -> f32 {
    let mut f = 0.0;
    for x in pos {
        f += x*x;
    }
    f
}

fn main() {
    println!("Hello from differential evolution!");

    // problem dimension
    let dim = 5;
    let settings = Settings::new(vec![-20.0; dim], vec![20.0; dim]);

    // create population
    let mut pop = Population::new(settings);
    
    for iter in 0..10000 {
        // evaluate individual
        // TODO make pos immutable somehow?
        for ind in &mut pop.curr {
            ind.cost = Some(square_fitness(&ind.pos));
        }

        if let Some(best) = pop.evolve() {
            println!("new best in iteration {}: {:?}", iter, best);
        }
    }

    println!("best: {:?}", pop.best());
}