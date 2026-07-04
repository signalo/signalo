// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(any(feature = "libm", feature = "std"))]
use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;

use crate::window_behavior_tests;

/// Numeric test fixture for smoke tests.
#[cfg(any(feature = "libm", feature = "std"))]
fn numeric_fixture() -> Vec<f32> {
    alloc::vec![
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0, 106.0, 5.0,
        26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0,
        11.0, 24.0, 24.0,
    ]
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn smoke() {
    const N: usize = 8;
    let mut window = TriangularArray::<f32, N>::default();
    let input = numeric_fixture();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

    // w[k] = [0, 2/7, 4/7, 6/7, 6/7, 4/7, 2/7, 0]
    let expected: Vec<f32> = alloc::vec![
        0.0,
        2.0 / 7.0,
        4.0,
        12.0 / 7.0,
        30.0 / 7.0,
        32.0 / 7.0,
        32.0 / 7.0,
        0.0,
        0.0,
        12.0 / 7.0,
        8.0,
        54.0 / 7.0,
        54.0 / 7.0,
        68.0 / 7.0,
        34.0 / 7.0,
        0.0,
        0.0,
        40.0 / 7.0,
        80.0 / 7.0,
        6.0,
        6.0,
        60.0 / 7.0,
        30.0 / 7.0,
        0.0,
        0.0,
        20.0 / 7.0,
        444.0 / 7.0,
        108.0 / 7.0,
        108.0 / 7.0,
        72.0 / 7.0,
        212.0 / 7.0,
        0.0,
        0.0,
        26.0 / 7.0,
        52.0 / 7.0,
        18.0,
        18.0,
        12.0,
        68.0 / 7.0,
        0.0,
        0.0,
        16.0 / 7.0,
        116.0 / 7.0,
        96.0 / 7.0,
        96.0 / 7.0,
        64.0 / 7.0,
        208.0 / 7.0,
        0.0,
        0.0,
        48.0 / 7.0,
    ];
    assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
}

window_behavior_tests!(
    TriangularArray::<f32, N>::default(),
    TriangularArray::<f32, 0>::with_config(Config {
        weights: [0.0f32; 0],
    }),
    "window size N must be > 0"
);

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn endpoints() {
    const N: usize = 8;
    let mut window = TriangularArray::<f32, N>::default();

    let input: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

    assert_abs_diff_eq!(output[0], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[N - 1], 0.0, epsilon = 1e-5);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn degenerate_n1() {
    let mut window = TriangularArray::<f32, 1>::default();
    assert_abs_diff_eq!(window.filter(42.0), 42.0, epsilon = 1e-5);
    assert_abs_diff_eq!(window.filter(7.0), 7.0, epsilon = 1e-5);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "window size N must be > 0")]
fn config_new_n0_panics() {
    let _ = Config::<[f64; 0]>::new();
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "window size N must be > 0")]
fn default_zero_window_panics() {
    let _ = TriangularArray::<f32, 0>::default();
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn config_new_weights_match_formula() {
    const N: usize = 8;
    let config = Config::<[f32; N]>::new();
    for k in 0..N {
        let k_f = k as f32;
        let n_f = N as f32;
        let expected = 1.0 - (2.0 * k_f / (n_f - 1.0) - 1.0).abs();
        assert_abs_diff_eq!(config.weights[k], expected, epsilon = 1e-5);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn triangular_parity_with_windowed_sinc() {
    use crate::filters::util::window::triangular;

    const N: usize = 33;
    let mut window = TriangularArray::<f64, N>::default();

    for k in 0..N {
        let expected = triangular::<f64>(k, N);
        let got = window.filter(1.0);
        assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
    }
}
