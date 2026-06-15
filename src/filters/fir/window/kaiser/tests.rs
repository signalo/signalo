// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;

fn numeric_fixture() -> Vec<f32> {
    std::vec![
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0, 106.0, 5.0,
        26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0,
        11.0, 24.0, 24.0,
    ]
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn bessel_i0_zero() {
    assert_abs_diff_eq!(bessel_i0(0.0f64), 1.0, epsilon = 1e-15);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn bessel_i0_one() {
    assert_abs_diff_eq!(bessel_i0(1.0f64), 1.266065877752008, epsilon = 1e-12);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn bessel_i0_ten() {
    assert_abs_diff_eq!(bessel_i0(10.0f64), 2815.716628466254, epsilon = 1e-10);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kaiser_n_eq_1() {
    let config = Config::<f32, 1>::new(6.0);
    assert_abs_diff_eq!(config.weights[0], 1.0, epsilon = 1e-7);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "window size N must be > 0")]
fn kaiser_n_eq_0_panics() {
    let _ = Config::<f64, 0>::new(6.0);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "window size N must be > 0")]
fn config_new_n0_panics() {
    let _ = Config::<f64, 0>::new(6.0);
}

#[cfg(feature = "std")]
#[test]
fn smoke() {
    const N: usize = 8;
    let beta = 1.0f32;
    let config = Config::<f32, N>::new(beta);
    let mut window = Kaiser::<f32, N>::with_config(config.clone());
    let input = numeric_fixture();
    let expected: Vec<f32> = input
        .iter()
        .enumerate()
        .map(|(i, &x)| x * config.weights[i % N])
        .collect();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
    assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
}

#[cfg(feature = "std")]
#[test]
fn periodicity() {
    const N: usize = 8;
    let beta = 1.0f32;
    let config = Config::<f32, N>::new(beta);
    let mut window = Kaiser::<f32, N>::with_config(config);
    let input: Vec<f32> = std::iter::repeat(1.0).take(3 * N).collect();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
    for block in 0..3 {
        let start = block * N;
        let end = start + N;
        assert_eq!(&output[start..end], &output[0..N]);
    }
}

#[cfg(feature = "std")]
#[test]
fn reset_restarts_counter() {
    const N: usize = 8;
    let beta = 1.0f32;
    let config = Config::<f32, N>::new(beta);
    let mut window = Kaiser::<f32, N>::with_config(config);

    let first_half: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
    let output_a: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

    window = window.reset();
    let output_b: Vec<_> = first_half.iter().map(|&x| window.filter(x)).collect();

    assert_eq!(output_a, output_b);
}

#[cfg(feature = "std")]
#[test]
fn endpoints() {
    const N: usize = 8;
    let beta = 1.0f32;
    let config = Config::<f32, N>::new(beta);
    let mut window = Kaiser::<f32, N>::with_config(config.clone());

    let input: Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
    let output: Vec<_> = input.iter().map(|&x| window.filter(x)).collect();

    assert_abs_diff_eq!(output[0], input[0] * config.weights[0], epsilon = 1e-5);
    assert_abs_diff_eq!(
        output[N - 1],
        input[N - 1] * config.weights[N - 1],
        epsilon = 1e-5
    );
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn beta_for_attenuation_40() {
    let beta = Config::<f64, 8>::beta_for_attenuation(40.0);
    assert_abs_diff_eq!(beta, 3.3953, epsilon = 1e-4);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn beta_for_attenuation_60() {
    let beta = Config::<f64, 8>::beta_for_attenuation(60.0);
    assert_abs_diff_eq!(beta, 5.65326, epsilon = 1e-4);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn beta_for_attenuation_boundary_21() {
    // Attenuation = 21 dB is the boundary of the zero-return region.
    let beta = Config::<f64, 8>::beta_for_attenuation(21.0);
    assert_abs_diff_eq!(beta, 0.0, epsilon = 1e-12);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn beta_for_attenuation_boundary_50() {
    // Attenuation = 50 dB is the boundary between mid- and high-attenuation formulas.
    let beta = Config::<f64, 8>::beta_for_attenuation(50.0);
    // Expected from formula: 0.1102 * (50.0 - 8.7) = 0.1102 * 41.3 = 4.55126
    assert_abs_diff_eq!(beta, 4.55126, epsilon = 1e-3);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn beta_for_attenuation_below_21_returns_zero() {
    let beta = Config::<f64, 8>::beta_for_attenuation(10.0);
    assert_abs_diff_eq!(beta, 0.0, epsilon = 1e-12);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn sidelobe_attenuation() {
    const N: usize = 64;
    const ZP: usize = 8;
    const L: usize = N * ZP;

    let config = Config::<f64, N>::new(6.0);
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

    let first_null = 5 * L / N;
    let mut side_peak = 0.0_f64;
    for bin in (first_null + 1)..magnitude.len() {
        if magnitude[bin] > side_peak {
            side_peak = magnitude[bin];
        }
    }

    let sidelobe_db = 20.0 * (side_peak / main_peak).log10();
    let documented = -57.0;

    assert!(
        (sidelobe_db - documented).abs() < 3.0,
        "Sidelobe {sidelobe_db} dB not within ±3 dB of documented {documented} dB"
    );
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kaiser_weights_consistency() {
    const N: usize = 33;
    let beta = 6.0_f64;
    let config = Config::<f64, N>::new(beta);
    let weights = config.weights;

    // Center tap must be 1.0 by definition
    let m = (N - 1) / 2;
    assert_abs_diff_eq!(weights[m], 1.0, epsilon = 1e-12);

    // Endpoint: w[0] = I₀(0) / I₀(β) = 1 / I₀(β)
    let i0_beta = bessel_i0(beta);
    assert_abs_diff_eq!(weights[0], 1.0 / i0_beta, epsilon = 1e-12);
    assert_abs_diff_eq!(weights[N - 1], 1.0 / i0_beta, epsilon = 1e-12);

    // Symmetry
    for k in 0..N / 2 {
        assert_abs_diff_eq!(weights[k], weights[N - 1 - k], epsilon = 1e-12);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kaiser_parity_with_windowed_sinc() {
    use crate::filters::fir::convolve::windowed_sinc::kaiser_window;

    const N: usize = 33;
    let beta = 6.0_f64;

    let config = Config::<f64, N>::new(beta);
    let win_fn = kaiser_window::<f64>(beta);

    for k in 0..N {
        let expected = win_fn(k, N);
        let got = config.weights[k];
        assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn all_weights_are_finite() {
    let config = Config::<f32, 16>::new(6.0);
    for weight in &config.weights {
        assert!(weight.is_finite());
        assert!(*weight >= 0.0);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "Kaiser beta must be non-negative")]
fn negative_beta_panics() {
    let _ = Config::<f64, 8>::new(-1.0);
}

#[test]
#[should_panic(expected = "window size N must be > 0")]
fn zero_window_panics() {
    let _ = Kaiser::<f32, 0>::with_config(Config {
        beta: 0.0f32,
        weights: [0.0f32; 0],
    });
}
