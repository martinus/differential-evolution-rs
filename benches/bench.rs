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


    fn setup<F: Fn(&[f32]) -> f32, R: Rng>(dim: usize,
                                                 cost_fn: F,
                                                 rng: R)
                                                 -> Population<F, R> {
        let s = Settings::min_max_rng(vec![(-100.0, 100.0); dim], cost_fn, rng);
        Population::from_settings(s)
    }

    #[bench]
    fn bench_square_fitness_opt(b: &mut Bencher) {
        let rng: XorShiftRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }


    #[bench]
    fn rand_thread_rng(b: &mut Bencher) {
        let mut pop = setup(5, |_| 1.234, thread_rng());
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_xor_shift(b: &mut Bencher) {
        let rng: XorShiftRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_chacha(b: &mut Bencher) {
        let rng: ChaChaRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_isaac(b: &mut Bencher) {
        let rng: IsaacRng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_isaac64(b: &mut Bencher) {
        let rng: Isaac64Rng = OsRng::new().unwrap().gen();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_std(b: &mut Bencher) {
        let rng: StdRng = StdRng::new().unwrap();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_osrng(b: &mut Bencher) {
        let rng: OsRng = OsRng::new().unwrap();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }

    #[bench]
    fn rand_weak_rng(b: &mut Bencher) {
        let rng = weak_rng();
        let mut pop = setup(5, |_| 1.234, rng);
        b.iter(|| pop.next());
    }
}
