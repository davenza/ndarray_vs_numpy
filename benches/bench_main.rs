#[macro_use]
extern crate criterion;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use criterion::Criterion;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::RandomExt;
use rand::{Rand, distributions::Range};

fn product(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "product",
        |b, &size| {
            // Number of axes to add on each matrix.
            let new_axes = size / 2;
            let mut rng = rand::thread_rng();
            let left = {
                let mut shape = vec![2; size];
                shape.extend_from_slice(&vec![1; new_axes]);
                ArrayD::<f64>::random_using(shape, Range::new(0., 1.), &mut rng)
            };
            let right = {
                let mut shape = vec![1; new_axes];
                shape.extend_from_slice(&vec![2; size]);
                ArrayD::<f64>::random_using(shape, Range::new(0., 1.), &mut rng)
            };
            // Expected cardinality of the product of the two arrays.
            let expected_card = vec![2; size + new_axes];

            b.iter_with_setup(
                || (left.view(), right.view(), expected_card.clone()),
                |(left_view, right_view, expected_card)| {
                    let mut out = unsafe { ArrayD::<f64>::uninitialized(expected_card) };
                    Zip::from(&mut out)
                        .and_broadcast(&left_view)
                        .and_broadcast(&right_view)
                        .apply(|out, a, b| *out = a * b);
                },
            )
        },
        vec![2usize, 4, 6, 8, 10, 12],
    );
}

criterion_group!(product_group, product);
criterion_main!(product_group);
