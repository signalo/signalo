// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec;
use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use crate::traits::{ConfigRef, Filter};
use crate::util::test_fixtures::collatz;

use super::*;

// MARK: Coefficient identity tests

macro_rules! coefficients_test {
    ($name:ident, $n:literal, $expected:expr) => {
        #[test]
        fn $name() {
            let filter = ConvolveArray::<f32, $n>::central_difference();
            let c = filter.config_ref().coefficients;
            // 1e-6 accounts for f32 division rounding.
            let expected: [f32; $n] = $expected;
            for (a, e) in c.iter().zip(expected.iter()) {
                assert_abs_diff_eq!(a, e, epsilon = 1e-6);
            }
        }
    };
}

coefficients_test!(coefficients_n3, 3, [0.5, 0.0, -0.5]);
coefficients_test!(
    coefficients_n5,
    5,
    [-1.0 / 12.0, 8.0 / 12.0, 0.0, -8.0 / 12.0, 1.0 / 12.0]
);
coefficients_test!(
    coefficients_n7,
    7,
    [
        1.0 / 60.0,
        -9.0 / 60.0,
        45.0 / 60.0,
        0.0,
        -45.0 / 60.0,
        9.0 / 60.0,
        -1.0 / 60.0
    ]
);
coefficients_test!(
    coefficients_n9,
    9,
    [
        -3.0 / 840.0,
        32.0 / 840.0,
        -168.0 / 840.0,
        672.0 / 840.0,
        0.0,
        -672.0 / 840.0,
        168.0 / 840.0,
        -32.0 / 840.0,
        3.0 / 840.0
    ]
);

// MARK: DC rejection tests

macro_rules! dc_rejection_test {
    ($name:ident, $n:literal) => {
        #[test]
        fn $name() {
            let filter = ConvolveArray::<f32, $n>::central_difference();
            let coeffs = filter.config_ref().coefficients;
            let sum: f32 = coeffs.iter().sum();
            // 1e-6 accounts for f32 dot-product rounding.
            assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-6);
        }
    };
}

dc_rejection_test!(dc_rejection_n3, 3);
dc_rejection_test!(dc_rejection_n5, 5);
dc_rejection_test!(dc_rejection_n7, 7);
dc_rejection_test!(dc_rejection_n9, 9);

// MARK: Ramp response tests (output → 1 after warm-up)

macro_rules! ramp_response_test {
    ($name:ident, $n:literal) => {
        #[test]
        fn $name() {
            let mut filter = ConvolveArray::<f32, $n>::central_difference();
            let warm_up = $n - 1;

            for n in 0..=(warm_up + 8) {
                let out = filter.filter(n as f32);
                if n >= warm_up {
                    // 1e-5 accounts for f32 dot-product rounding.
                    assert_abs_diff_eq!(out, 1.0, epsilon = 1e-5);
                }
            }
        }
    };
}

ramp_response_test!(ramp_response_n3, 3);
ramp_response_test!(ramp_response_n5, 5);
ramp_response_test!(ramp_response_n7, 7);
ramp_response_test!(ramp_response_n9, 9);

// MARK: Quadratic input tests

macro_rules! quadratic_input_test {
    ($name:ident, $n:literal) => {
        #[test]
        fn $name() {
            // f(x) = x², f'(x) = 2x. Centre tap at M = (N-1)/2, output = 2*(n-M).
            let mut filter = ConvolveArray::<f32, $n>::central_difference();
            let m = ($n - 1) / 2;
            let warm_up = $n - 1;

            for n in 0..=(warm_up + 8) {
                let out = filter.filter((n * n) as f32);
                if n >= warm_up {
                    let expected = 2.0 * (n as f32 - m as f32);
                    // 1e-5 accounts for accumulated f32 dot-product and squaring error.
                    assert_abs_diff_eq!(out, expected, epsilon = 1e-5);
                }
            }
        }
    };
}

quadratic_input_test!(quadratic_input_n3, 3);
quadratic_input_test!(quadratic_input_n5, 5);
quadratic_input_test!(quadratic_input_n7, 7);
quadratic_input_test!(quadratic_input_n9, 9);

// MARK: Group delay test

#[test]
fn central_difference_group_delay_n5() {
    // N=5 → M = (5-1)/2 = 2.
    // The impulse response is antisymmetric around index M,
    // confirming the group delay of M samples.
    let mut filter = ConvolveArray::<f32, 5>::central_difference();
    let y: Vec<f32> = [1.0_f32]
        .into_iter()
        .chain(core::iter::repeat(0.0).take(5))
        .map(|x| filter.filter(x))
        .collect();
    // Antisymmetry about M=2: y[n] = -y[2M - n] for 0 ≤ n ≤ 2M.
    assert_abs_diff_eq!(y[0], -y[4], epsilon = 1e-6);
    assert_abs_diff_eq!(y[1], -y[3], epsilon = 1e-6);
    assert_abs_diff_eq!(y[2], 0.0, epsilon = 1e-6);
}

// MARK: Smoke test

#[test]
fn smoke() {
    let filter = ConvolveArray::<f32, 3>::central_difference();
    let input = collatz();
    let output: Vec<_> = input
        .iter()
        .scan(filter, |f, &x| Some(f.filter(x)))
        .collect();

    #[rustfmt::skip]
    let expected = vec![
        0.0, 0.5, 3.5, 0.5, -1.0, 3.0, 5.5, 2.5, 1.5, -3.5, -2.5, 1.5, -2.5, 4.0, 4.0, -6.5,
        -2.5, 8.0, 4.0, -6.5, -6.5, 4.0, 4.0, -2.5, 4.0, 0.0, 44.0, 85.0, -1.5, -81.0, -1.0,
        -6.5, -40.0, 4.0, -6.5, 4.0, 4.0, 0.0, 6.5, -6.5, 37.5, 0.0, -40.0, 4.0, -6.5, 0.0,
        44.0, -2.5, -40.0, 6.5,
    ];

    assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
}

// MARK: f64 precision

#[test]
fn f64_coefficients_match_f32() {
    let filter_f32 = ConvolveArray::<f32, 5>::central_difference();
    let filter_f64 = ConvolveArray::<f64, 5>::central_difference();

    let c32 = filter_f32.config_ref().coefficients;
    let c64 = filter_f64.config_ref().coefficients;

    // f32 has ~7 decimal digits of precision; comparing across precision
    // levels requires an epsilon that accounts for f32 rounding.
    for (a, b) in c32.iter().zip(c64.iter()) {
        assert_abs_diff_eq!(f64::from(*a), *b, epsilon = 1e-7);
    }
}

// MARK: Runtime constructor tests

macro_rules! runtime_matches_table_test {
    ($name:ident, $n:literal) => {
        #[test]
        fn $name() {
            let table = <ConvolveArray<f32, $n> as FirDifferentiator>::central_difference();
            let runtime = ConvolveArray::<f32, $n>::central_difference_runtime();

            let ct = table.config_ref().coefficients;
            let cr = runtime.config_ref().coefficients;

            for (a, b) in ct.iter().zip(cr.iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-6);
            }
        }
    };
}

runtime_matches_table_test!(runtime_matches_table_n3, 3);
runtime_matches_table_test!(runtime_matches_table_n5, 5);
runtime_matches_table_test!(runtime_matches_table_n7, 7);
runtime_matches_table_test!(runtime_matches_table_n9, 9);

#[test]
fn runtime_n11_f64_dc_rejection() {
    let filter = ConvolveArray::<f64, 11>::central_difference_runtime();
    let coeffs = filter.config_ref().coefficients;
    let sum: f64 = coeffs.iter().sum();

    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-12);
}

#[test]
fn runtime_n11_f64_ramp_response() {
    let mut filter = ConvolveArray::<f64, 11>::central_difference_runtime();

    for n in 0..=20 {
        let out = filter.filter(f64::from(n));
        if n >= 10 {
            assert_abs_diff_eq!(out, 1.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn runtime_n19_f64_dc_rejection_and_ramp() {
    let filter = ConvolveArray::<f64, 19>::central_difference_runtime();
    let coeffs = filter.config_ref().coefficients;
    let sum: f64 = coeffs.iter().sum();
    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-12);

    let mut filter = ConvolveArray::<f64, 19>::central_difference_runtime();
    for n in 0..=30 {
        let out = filter.filter(f64::from(n));
        if n >= 18 {
            assert_abs_diff_eq!(out, 1.0, epsilon = 1e-10);
        }
    }
}

#[test]
#[should_panic(expected = "N <= 19")]
fn runtime_n21_panics() {
    let _ = ConvolveArray::<f64, 21>::central_difference_runtime();
}

#[test]
fn runtime_f64_matches_table_n5() {
    let table = ConvolveArray::<f64, 5>::central_difference();
    let runtime = ConvolveArray::<f64, 5>::central_difference_runtime();

    let ct = table.config_ref().coefficients;
    let cr = runtime.config_ref().coefficients;

    for (a, b) in ct.iter().zip(cr.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }
}

#[test]
fn runtime_n9_f32_does_not_panic() {
    let _ = ConvolveArray::<f32, 9>::central_difference_runtime();
}

#[test]
#[should_panic(expected = "N must be <= 9")]
fn runtime_n11_f32_panics() {
    let _ = ConvolveArray::<f32, 11>::central_difference_runtime();
}

// MARK: Laplacian tests

#[test]
fn laplacian_coefficients_n3() {
    let filter = ConvolveArray::<f32, 3>::laplacian();
    let c = filter.config_ref().coefficients;

    // 1e-6 accounts for f32 literal conversion rounding.
    assert_abs_diff_eq!(c[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[1], -2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[2], 1.0, epsilon = 1e-6);
}

#[test]
fn laplacian_n3_dc_rejection() {
    let filter = ConvolveArray::<f32, 3>::laplacian();
    let coeffs = filter.config_ref().coefficients;
    let sum: f32 = coeffs.iter().sum();

    // Integer-coefficient sum is exact in f32.
    assert_abs_diff_eq!(sum, 0.0, epsilon = f32::EPSILON);
}

#[test]
fn laplacian_n3_quadratic() {
    // f(x) = x², f″(x) = 2. After warm-up, output should be 2.
    let mut filter = ConvolveArray::<f32, 3>::laplacian();

    for n in 0..=10 {
        let out = filter.filter((n * n) as f32);
        if n >= 2 {
            assert_abs_diff_eq!(out, 2.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn laplacian_n3_ramp_rejection() {
    // f(x) = x, after warm-up, output should be 0 (zero second derivative).
    let mut filter = ConvolveArray::<f32, 3>::laplacian();

    for n in 0..=10 {
        let out = filter.filter(n as f32);
        if n >= 2 {
            assert_abs_diff_eq!(out, 0.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn laplacian_n3_constant_rejection() {
    let mut filter = ConvolveArray::<f32, 3>::laplacian();

    for n in 0..=5 {
        let out = filter.filter(7.0);
        // After warm-up, should be 0 for constant input
        if n >= 2 {
            assert_abs_diff_eq!(out, 0.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn second_central_difference_n5_coefficients() {
    let filter = ConvolveArray::<f32, 5>::second_central_difference();
    let c = filter.config_ref().coefficients;

    // 1e-6 accounts for f32 division rounding.
    assert_abs_diff_eq!(c[0], -1.0 / 12.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[1], 4.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[2], -5.0 / 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[3], 4.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[4], -1.0 / 12.0, epsilon = 1e-6);
}

#[test]
fn second_central_difference_n5_dc_rejection() {
    let filter = ConvolveArray::<f32, 5>::second_central_difference();
    let coeffs = filter.config_ref().coefficients;
    let sum: f32 = coeffs.iter().sum();

    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-6);
}

#[test]
fn second_central_difference_n5_quadratic() {
    let mut filter = ConvolveArray::<f32, 5>::second_central_difference();

    for n in 0..=12 {
        let out = filter.filter((n * n) as f32);
        if n >= 4 {
            assert_abs_diff_eq!(out, 2.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn second_central_difference_n5_ramp_rejection() {
    let mut filter = ConvolveArray::<f32, 5>::second_central_difference();

    for n in 0..=12 {
        let out = filter.filter(n as f32);
        if n >= 4 {
            assert_abs_diff_eq!(out, 0.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn laplacian_smoke() {
    let filter = ConvolveArray::<f32, 3>::laplacian();
    let input = collatz();
    let output: Vec<_> = input
        .iter()
        .scan(filter, |f, &x| Some(f.filter(x)))
        .collect();

    #[rustfmt::skip]
    let expected = vec![
        0.0, 1.0, 5.0, -11.0, 8.0, 0.0, 5.0, -11.0, 9.0, -19.0, 21.0, -13.0, 5.0, 8.0, -8.0,
        -13.0, 21.0, 0.0, -8.0, -13.0, 13.0, 8.0, -8.0, -5.0, 18.0, -26.0, 114.0, -32.0,
        -141.0, -18.0, 178.0, -189.0, 122.0, -34.0, 13.0, 8.0, -8.0, 0.0, 13.0, -39.0, 127.0,
        -202.0, 122.0, -34.0, 13.0, 0.0, 88.0, -181.0, 106.0, -13.0,
    ];

    assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
}
