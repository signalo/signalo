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
    let config = Config::<f32, N>::new();
    let mut window = Hann::<f32, N>::with_config(config.clone());
    let input = numeric_fixture();
    let expected: Vec<f32> = input
        .iter()
        .enumerate()
        .map(|(i, &x)| x * config.weights[i % N])
        .collect();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
    assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
}

window_behavior_tests!(
    Hann::<f32, N>::with_config(Config::<f32, N>::new()),
    Hann::<f32, 0>::with_config(Config {
        weights: [0.0f32; 0],
    }),
    "window size N must be > 0"
);

#[test]
fn endpoints() {
    const N: usize = 8;
    let config = Config::<f32, N>::new();
    let mut window = Hann::<f32, N>::with_config(config.clone());

    let input: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

    assert_abs_diff_eq!(output[0], input[0] * config.weights[0], epsilon = 1e-5);
    assert_abs_diff_eq!(
        output[N - 1],
        input[N - 1] * config.weights[N - 1],
        epsilon = 1e-5
    );
}

#[test]
#[should_panic(expected = "Hann: window size N must be > 0")]
fn from_guts_zero_window_panics() {
    let _: Hann<f32, 0> = FromGuts::from_guts((
        Config {
            weights: [0.0f32; 0],
        },
        State::default(),
    ));
}

#[test]
fn config_new() {
    const N: usize = 8;
    let config = Config::<f32, N>::new();
    let n_minus_1 = (N - 1) as f32;
    let two_pi = 2.0 * core::f32::consts::PI;
    for k in 0..N {
        let alpha = two_pi * k as f32 / n_minus_1;
        let expected = 0.5 * (1.0 - alpha.cos());
        assert_abs_diff_eq!(config.weights[k], expected, epsilon = 1e-5);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn sidelobe_attenuation() {
    const N: usize = 64;
    const ZP: usize = 8;
    const L: usize = N * ZP;

    let config = Config::<f64, N>::new();
    let weights = &config.weights;

    let two_pi = core::f64::consts::PI * 2.0;
    let mut magnitude = [0.0_f64; L / 2];
    for bin in 0..(L / 2) {
        let mut re = 0.0;
        let mut im = 0.0;
        for k in 0..N {
            let theta = two_pi * bin as f64 * k as f64 / L as f64;
            re += weights[k] * theta.cos();
            im += weights[k] * -(theta.sin());
        }
        magnitude[bin] = (re * re + im * im).sqrt();
    }

    let main_peak = magnitude[0];

    let first_null = 2 * L / N;
    let mut side_peak = 0.0_f64;
    for bin in (first_null + 1)..magnitude.len() {
        if magnitude[bin] > side_peak {
            side_peak = magnitude[bin];
        }
    }

    let sidelobe_db = 20.0 * (side_peak / main_peak).log10();
    let documented = -31.5;

    assert!(
        (sidelobe_db - documented).abs() < 3.0,
        "Sidelobe {sidelobe_db} dB not within ±3 dB of documented {documented} dB"
    );
}

#[test]
fn n_eq_1() {
    let config = Config::<f32, 1>::new();
    assert_abs_diff_eq!(config.weights[0], 1.0, epsilon = 1e-7);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn hann_parity_with_windowed_sinc() {
    use crate::filters::fir::convolve::windowed_sinc::hann_window;

    const N: usize = 33;
    let config = Config::<f64, N>::new();
    let win_fn = hann_window::<f64>;

    for k in 0..N {
        let expected = win_fn(k, N);
        let got = config.weights[k];
        assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn hann_parity_with_windowed_sinc_n2() {
    use crate::filters::fir::convolve::windowed_sinc::hann_window;

    const N: usize = 2;
    let config = Config::<f64, N>::new();
    let win_fn = hann_window::<f64>;

    for k in 0..N {
        let expected = win_fn(k, N);
        let got = config.weights[k];
        assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "window size N must be > 0")]
fn config_new_n0_panics() {
    let _ = Config::<f64, 0>::new();
}
