// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec;
use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use crate::traits::{ConfigRef, Filter};
use crate::util::test_fixtures::collatz;

use super::*;

// MARK: Integer-delta identity tests

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn integer_delta_identity_m3() {
    for delta in 0..=2 {
        let filter = ConvolveArray::<f32, 3>::lagrange(delta as f32);
        let c = filter.config_ref().coefficients;

        for (k, &val) in c.iter().enumerate() {
            if k == delta {
                assert_abs_diff_eq!(val, 1.0, epsilon = 1e-5);
            } else {
                assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
            }
        }
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn integer_delta_identity_m5() {
    for delta in 0..=4 {
        let filter = ConvolveArray::<f32, 5>::lagrange(delta as f32);
        let c = filter.config_ref().coefficients;

        for (k, &val) in c.iter().enumerate() {
            if k == delta {
                assert_abs_diff_eq!(val, 1.0, epsilon = 1e-5);
            } else {
                assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
            }
        }
    }
}

// MARK: Sum-to-one tests

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn sum_to_one_m4() {
    for delta_int in 0..=10 {
        let delta = delta_int as f32 * 0.3;
        let filter = ConvolveArray::<f32, 4>::lagrange(delta);
        let coeffs = filter.config_ref().coefficients;
        let sum: f32 = coeffs.iter().sum();

        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn sum_to_one_m6() {
    for delta_int in 0..=16 {
        let delta = delta_int as f32 * 0.3;
        let filter = ConvolveArray::<f32, 6>::lagrange(delta);
        let coeffs = filter.config_ref().coefficients;
        let sum: f32 = coeffs.iter().sum();

        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn sum_to_one_f64_edge() {
    // Test near edges where ill-conditioning is worst.
    let filter1 = ConvolveArray::<f64, 8>::lagrange(0.001);
    let sum1: f64 = filter1.config_ref().coefficients.iter().sum();
    assert_abs_diff_eq!(sum1, 1.0, epsilon = 1e-10);

    let filter2 = ConvolveArray::<f64, 8>::lagrange(6.999);
    let sum2: f64 = filter2.config_ref().coefficients.iter().sum();
    assert_abs_diff_eq!(sum2, 1.0, epsilon = 1e-10);
}

// MARK: Linear-signal preservation

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn linear_signal_preservation() {
    // x[n] = a + b*n, output should be a + b*(n - delta) after warm-up.
    let a = 3.0f32;
    let b = 2.5f32;
    let delta = 1.5f32;
    let m: usize = 4;
    let mut filter = ConvolveArray::<f32, 4>::lagrange(delta);

    for n in 0..=20 {
        let x = a + b * (n as f32);
        let out = filter.filter(x);

        if n >= m {
            let expected = a + b * ((n as f32) - delta);
            assert_abs_diff_eq!(out, expected, epsilon = 1e-4);
        }
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn lagrange_delta_0p3_matches_x_minus_0p3() {
    let mut f = ConvolveArray::<f64, 4>::lagrange(0.3);
    for n in 0..40 {
        let x = f64::from(n);
        let y = f.filter(x);
        if n >= 16 {
            let expected = f64::from(n) - 0.3;
            assert!(
                (y - expected).abs() < 1e-12,
                "n={n}: y={y} expected={expected}"
            );
        }
    }
}

// MARK: Phase-lag test (slow sinusoid)

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn phase_lag_slow_sinusoid() {
    let f = 0.05f64;
    let delta = 2.5f64;
    let m: usize = 6;
    let n_samples = 10_000;

    let mut filter = ConvolveArray::<f64, 6>::lagrange(delta);

    // Warm up
    for n in 0..(m * 4) {
        let _ = filter.filter(f64::sin(2.0 * core::f64::consts::PI * f * (n as f64)));
    }

    // Measure phase in steady state via IQ demodulation.
    let mut i_acc = 0.0f64;
    let mut q_acc = 0.0f64;

    for n in 0..n_samples {
        let idx = (m * 4 + n) as f64;
        let phase = 2.0 * core::f64::consts::PI * f * idx;
        let x = f64::sin(phase);
        let y = filter.filter(x);

        i_acc += y * f64::sin(phase);
        q_acc += y * f64::cos(phase);
    }

    let measured_phase = f64::atan2(-q_acc, i_acc);

    // Expected phase: 2π · f · δ
    let expected_phase = 2.0 * core::f64::consts::PI * f * delta;

    // 1 mrad tolerance
    assert_abs_diff_eq!(measured_phase, expected_phase, epsilon = 1e-3);
}

// MARK: Smoke test

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn smoke() {
    let filter = ConvolveArray::<f32, 4>::lagrange(1.5);
    let input = collatz();
    let output: Vec<_> = input
        .iter()
        .scan(filter, |f, &x| Some(f.filter(x)))
        .collect();

    #[rustfmt::skip]
    let expected = vec![
        0.0, -0.0625, 0.125, 4.375, 4.6875, 3.0, 6.1875, 12.375, 14.625, 16.625, 12.375, 9.5,
        12.0, 8.1875, 13.0, 18.3125, 10.0, 6.6875, 16.5, 21.3125, 13.5, 5.6875, 11.0, 15.8125,
        11.6875, 17.0, 11.0, 55.375, 156.3125, 153.9375, 53.0, 62.6875, 59.6875, 10.0, 20.8125,
        11.6875, 17.0, 21.5, 20.1875, 29.125, 15.5, 63.1875, 63.5, 13.0, 23.8125, 15.1875,
        10.5, 65.8125, 62.1875, 11.6875,
    ];

    assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
}

// MARK: f64 precision

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn f64_half_sample_matches_f32() {
    let filter_f32 = ConvolveArray::<f32, 4>::lagrange(1.5);
    let filter_f64 = ConvolveArray::<f64, 4>::lagrange(1.5);

    let c32 = filter_f32.config_ref().coefficients;
    let c64 = filter_f64.config_ref().coefficients;

    for (a, b) in c32.iter().zip(c64.iter()) {
        assert_abs_diff_eq!(f64::from(*a), *b, epsilon = 1e-7);
    }
}

// MARK: Half-sample-delay table tests

#[test]
fn half_sample_m4_golden() {
    let filter = ConvolveArray::<f32, 4>::half_sample_delay();
    let c = filter.config_ref().coefficients;

    assert_abs_diff_eq!(c[0], -0.0625, epsilon = f32::EPSILON);
    assert_abs_diff_eq!(c[1], 0.5625, epsilon = f32::EPSILON);
    assert_abs_diff_eq!(c[2], 0.5625, epsilon = f32::EPSILON);
    assert_abs_diff_eq!(c[3], -0.0625, epsilon = f32::EPSILON);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn half_sample_m4_matches_runtime() {
    let table = ConvolveArray::<f32, 4>::half_sample_delay();
    let runtime = ConvolveArray::<f32, 4>::lagrange(1.5);

    let ct = table.config_ref().coefficients;
    let cr = runtime.config_ref().coefficients;

    for (a, b) in ct.iter().zip(cr.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-7);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn half_sample_m6_matches_runtime() {
    let table = ConvolveArray::<f64, 6>::half_sample_delay();
    let runtime = ConvolveArray::<f64, 6>::lagrange(2.5);

    let ct = table.config_ref().coefficients;
    let cr = runtime.config_ref().coefficients;

    for (a, b) in ct.iter().zip(cr.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn half_sample_m10_f32_bit_exact_with_runtime() {
    let table = ConvolveArray::<f32, 10>::half_sample_delay();
    let runtime = ConvolveArray::<f32, 10>::lagrange(4.5);
    for (a, b) in table
        .config_ref()
        .coefficients
        .iter()
        .zip(runtime.config_ref().coefficients.iter())
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit mismatch: table={a:e} runtime={b:e}"
        );
    }
}

// MARK: Out-of-range delta panic tests

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "delta must be non-negative")]
fn lagrange_delta_negative_panics() {
    let _ = ConvolveArray::<f64, 3>::lagrange(-0.1);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "delta must be <= M-1")]
fn lagrange_delta_too_large_panics() {
    let _ = ConvolveArray::<f64, 3>::lagrange(3.0);
}
