# Differential Evolution [![Crates Version](https://img.shields.io/crates/v/differential-evolution.svg)](https://crates.io/crates/differential-evolution) [![Build Status](https://travis-ci.org/martinus/differential-evolution-rs.svg?branch=master)](https://travis-ci.org/martinus/differential-evolution-rs)

Simple and powerful global optimization using a [Self-Adapting Differential Evolution](https://www.researchgate.net/publication/3418914_Self-Adapting_Control_Parameters_in_Differential_Evolution_A_Comparative_Study_on_Numerical_Benchmark_Problems) for Rust. See Wikipedia's article on [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) for more information.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
differential-evolution = "*"
```

and this to your crate root:

```rust
extern crate differential_evolution;
```

## Examples

Differential Evolution is a global optimization algorithm that tries to iteratively improve candidate solutions with regards to a user-defined cost function. 

This example finds the minimum of a simple 5-dimensional function.

```rust
// Simple example how to use the API.
extern crate differential_evolution;

use differential_evolution::Population;

fn main() {
    // problem dimension
    let dim = 5;

    // initial search space for each dimension
    let initial_min_max = vec![(-10.0, 10.0); dim];

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
```

# Similar Crates

- [darwin-rs](https://github.com/willi-kappler/darwin-rs)
- [RsGenetic](https://github.com/m-decoster/RsGenetic)

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
