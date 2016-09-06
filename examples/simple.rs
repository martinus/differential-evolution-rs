// Simple example how to use the API.
extern crate differential_evolution;

use differential_evolution::Population;

// function that we try to minimize
fn sum_of_squares(pos: &[f32]) -> f32 {
    let mut f = 0.0;
    for x in pos {
        f += x*x;
    }
    f
}

fn main() {
    // problem dimension
    let dim = 10;

    // create population with default settings:
    let mut pop = Population::new(vec![-100.0; dim], vec![100.0; dim]);
    
    // performs 10000 fitness evaluations. 
    //pop.iter(|pos| { sum_of_squares(pos) });

    //println!("best cost: {:?}", pop.best().cost);
    //println!("best position: {:?}", pop.best().pos);


    for iter in 0..10000 {
        // evaluate individual
        // TODO make pos immutable somehow?
        pop.iter(|pos| { sum_of_squares(pos) });
        /*
        for ind in &mut pop.curr {
            ind.cost = Some(sum_of_squares(&ind.pos));
        }
        */
        if let Some(best) = pop.evolve() {
            println!("new best in iteration {}: {:?}", iter, best);
        }
    }

    println!("best: {:?}", pop.best());
}