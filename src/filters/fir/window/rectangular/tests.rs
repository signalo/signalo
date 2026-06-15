// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;

use crate::window_behavior_tests;

/// Numeric test fixture for smoke tests.
fn numeric_fixture() -> Vec<f32> {
    std::vec![
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0, 106.0, 5.0,
        26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0,
        11.0, 24.0, 24.0,
    ]
}

#[test]
fn smoke() {
    const N: usize = 8;
    let mut window = Rectangular::<f32, N>::default();
    let input = numeric_fixture();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
    assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-5);
}

window_behavior_tests!(
    Rectangular::<f32, N>::default(),
    Rectangular::<f32, 0>::with_config(Config(core::marker::PhantomData)),
    "window size N must be > 0"
);

#[test]
fn endpoints() {
    const N: usize = 8;
    let mut window = Rectangular::<f32, N>::default();

    let input: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

    assert_abs_diff_eq!(output[0], input[0], epsilon = 1e-5);
    assert_abs_diff_eq!(output[N - 1], input[N - 1], epsilon = 1e-5);
}
