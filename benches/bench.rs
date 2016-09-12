// Copyright 2016 Martin Ankerl. 
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(test)]

extern crate test;
extern crate differential_evolution;
extern crate rand;

#[cfg(test)]
mod tests {
    use test::Bencher;
    use rand::{XorShiftRng, StdRng, IsaacRng, Isaac64Rng, Rng, ChaChaRng};
    use rand::{OsRng, weak_rng, thread_rng};
    use differential_evolution::{Population, Settings};


    fn setup<F: Fn(&[f32]) -> C, R: Rng, C: PartialOrd + Clone>(dim: usize,
                                                 cost_fn: F,
                                                 rng: R)
                                                 -> Population<F, R, C> {
        let s = Settings {
            min_max_pos: vec![(-100.0, 100.0); dim],
            cr_min_max: (0.0, 1.0),
            cr_change_probability: 0.1,
            f_min_max: (0.1, 1.0),
            f_change_probability: 0.1,
            pop_size: 100,
            rng: rng,
            cost_function: cost_fn,
        };
        Population::new(s)
    }

    #[bench]
    fn bench_square_fitness_opt(b: &mut Bencher) {
        let rng: XorShiftRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.eval());
    }


    #[bench]
    fn rand_thread_rng(b: &mut Bencher) {
        let mut pop = setup(5, |_| 1.234, thread_rng());
        b.iter(|| pop.eval());
    }

    #[bench]
    fn rand_xor_shift(b: &mut Bencher) {
        let rng: XorShiftRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.eval());
    }

    #[bench]
    fn rand_chacha(b: &mut Bencher) {
        let rng: ChaChaRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.eval());
    }

    #[bench]
    fn rand_isaac(b: &mut Bencher) {
        let rng: IsaacRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.eval());
    }

    #[bench]
    fn rand_isaac64(b: &mut Bencher) {
        let rng: Isaac64Rng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.eval());
    }

    #[bench]
    fn rand_std(b: &mut Bencher) {
        let rng: StdRng = StdRng::new().unwrap();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.eval());
    }

    #[bench]
    fn rand_osrng(b: &mut Bencher) {
        let rng: OsRng = OsRng::new().unwrap();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.eval());
    }

    #[bench]
    fn rand_weak_rng(b: &mut Bencher) {
        let rng = weak_rng();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.eval());
    }
}
