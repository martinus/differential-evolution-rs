// Copyright 2016 Martin Ankerl. 
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate differential_evolution;

use differential_evolution::self_adaptive_de;

fn main() {
    // create a self adaptive DE with an inital search area
    // from -10 to 10 in 5 dimensions.
    let mut de = self_adaptive_de(vec![(-10.0, 10.0); 5], |pos| {
        // cost function to minimize: sum of squares
        pos.iter().fold(0.0, |sum, x| sum + x*x)
    });

    // perform 10000 cost evaluations
    de.iter().nth(10000);
    
    // show the result
    let (cost, pos) = de.best().unwrap();
    println!("cost: {}", cost);
    println!("pos: {:?}", pos);
}