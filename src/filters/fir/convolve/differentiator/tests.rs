// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec;
use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use crate::traits::{ConfigRef, Filter};

use super::*;

fn collatz() -> Vec<f32> {
    vec![
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0, 106.0, 5.0,
        26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0,
        11.0, 24.0, 24.0,
    ]
}

// MARK: Coefficient identity tests

#[test]
fn coefficients_n3() {
    let filter = Convolve::<f32, 3>::central_difference();
    let c = filter.config_ref().coefficients;

    // 1e-6 accounts for f32 division rounding.
    assert_abs_diff_eq!(c[0], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(c[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[2], -0.5, epsilon = 1e-6);
}

#[test]
fn coefficients_n5() {
    let filter = Convolve::<f32, 5>::central_difference();
    let c = filter.config_ref().coefficients;

    // 1e-6 accounts for f32 division rounding.
    assert_abs_diff_eq!(c[0], -1.0 / 12.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[1], 8.0 / 12.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[2], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[3], -8.0 / 12.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[4], 1.0 / 12.0, epsilon = 1e-6);
}

#[test]
fn coefficients_n7() {
    let filter = Convolve::<f32, 7>::central_difference();
    let c = filter.config_ref().coefficients;

    // 1e-6 accounts for f32 division rounding.
    assert_abs_diff_eq!(c[0], 1.0 / 60.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[1], -9.0 / 60.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[2], 45.0 / 60.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[3], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[4], -45.0 / 60.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[5], 9.0 / 60.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[6], -1.0 / 60.0, epsilon = 1e-6);
}

#[test]
fn coefficients_n9() {
    let filter = Convolve::<f32, 9>::central_difference();
    let c = filter.config_ref().coefficients;

    // 1e-6 accounts for f32 division rounding.
    assert_abs_diff_eq!(c[0], -3.0 / 840.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[1], 32.0 / 840.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[2], -168.0 / 840.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[3], 672.0 / 840.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[4], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[5], -672.0 / 840.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[6], 168.0 / 840.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[7], -32.0 / 840.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[8], 3.0 / 840.0, epsilon = 1e-6);
}

// MARK: DC rejection tests

#[test]
fn dc_rejection_n3() {
    let filter = Convolve::<f32, 3>::central_difference();
    let coeffs = filter.config_ref().coefficients;
    let sum: f32 = coeffs.iter().sum();

    // 1e-6 accounts for f32 dot-product rounding.
    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-6);
}

#[test]
fn dc_rejection_n5() {
    let filter = Convolve::<f32, 5>::central_difference();
    let coeffs = filter.config_ref().coefficients;
    let sum: f32 = coeffs.iter().sum();

    // 1e-6 accounts for f32 dot-product rounding.
    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-6);
}

#[test]
fn dc_rejection_n7() {
    let filter = Convolve::<f32, 7>::central_difference();
    let coeffs = filter.config_ref().coefficients;
    let sum: f32 = coeffs.iter().sum();

    // 1e-6 accounts for f32 dot-product rounding.
    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-6);
}

#[test]
fn dc_rejection_n9() {
    let filter = Convolve::<f32, 9>::central_difference();
    let coeffs = filter.config_ref().coefficients;
    let sum: f32 = coeffs.iter().sum();

    // 1e-6 accounts for f32 dot-product rounding.
    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-6);
}

// MARK: Ramp response tests (output → 1 after warm-up)

#[test]
fn ramp_response_n3() {
    let mut filter = Convolve::<f32, 3>::central_difference();

    for n in 0..=10 {
        let out = filter.filter(n as f32);
        // After warm-up (N samples fed), output should be 1.
        if n >= 2 {
            // 1e-6 accounts for f32 dot-product rounding.
            assert_abs_diff_eq!(out, 1.0, epsilon = 1e-6);
        }
    }
}

#[test]
fn ramp_response_n5() {
    let mut filter = Convolve::<f32, 5>::central_difference();

    for n in 0..=12 {
        let out = filter.filter(n as f32);
        if n >= 4 {
            assert_abs_diff_eq!(out, 1.0, epsilon = 1e-6);
        }
    }
}

#[test]
fn ramp_response_n7() {
    let mut filter = Convolve::<f32, 7>::central_difference();

    for n in 0..=14 {
        let out = filter.filter(n as f32);
        if n >= 6 {
            assert_abs_diff_eq!(out, 1.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn ramp_response_n9() {
    let mut filter = Convolve::<f32, 9>::central_difference();

    for n in 0..=16 {
        let out = filter.filter(n as f32);
        if n >= 8 {
            assert_abs_diff_eq!(out, 1.0, epsilon = 1e-5);
        }
    }
}

// MARK: Quadratic input tests

#[test]
fn quadratic_input_n3() {
    // f(x) = x², f'(x) = 2x. N=3 centre tap at M=1, so output = 2*(n-1).
    let mut filter = Convolve::<f32, 3>::central_difference();

    for n in 0..=10 {
        let out = filter.filter((n * n) as f32);
        if n >= 2 {
            let expected = 2.0 * (n as f32 - 1.0);
            // 1e-5 accounts for accumulated f32 dot-product and squaring error.
            assert_abs_diff_eq!(out, expected, epsilon = 1e-5);
        }
    }
}

#[test]
fn quadratic_input_n5() {
    // N=5 centre tap at M=2, output = 2*(n-2).
    let mut filter = Convolve::<f32, 5>::central_difference();

    for n in 0..=12 {
        let out = filter.filter((n * n) as f32);
        if n >= 4 {
            let expected = 2.0 * (n as f32 - 2.0);
            assert_abs_diff_eq!(out, expected, epsilon = 1e-5);
        }
    }
}

#[test]
fn quadratic_input_n7() {
    // N=7 centre tap at M=3, output = 2*(n-3).
    let mut filter = Convolve::<f32, 7>::central_difference();

    for n in 0..=14 {
        let out = filter.filter((n * n) as f32);
        if n >= 6 {
            let expected = 2.0 * (n as f32 - 3.0);
            assert_abs_diff_eq!(out, expected, epsilon = 1e-5);
        }
    }
}

#[test]
fn quadratic_input_n9() {
    // N=9 centre tap at M=4, output = 2*(n-4).
    let mut filter = Convolve::<f32, 9>::central_difference();

    for n in 0..=16 {
        let out = filter.filter((n * n) as f32);
        if n >= 8 {
            let expected = 2.0 * (n as f32 - 4.0);
            assert_abs_diff_eq!(out, expected, epsilon = 1e-5);
        }
    }
}

// MARK: Group delay test

#[test]
fn central_difference_group_delay_n5() {
    // N=5 → M = (5-1)/2 = 2.
    // The impulse response is antisymmetric around index M,
    // confirming the group delay of M samples.
    let mut filter = Convolve::<f32, 5>::central_difference();
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
    let filter = Convolve::<f32, 3>::central_difference();
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
    let filter_f32 = Convolve::<f32, 5>::central_difference();
    let filter_f64 = Convolve::<f64, 5>::central_difference();

    let c32 = filter_f32.config_ref().coefficients;
    let c64 = filter_f64.config_ref().coefficients;

    // f32 has ~7 decimal digits of precision; comparing across precision
    // levels requires an epsilon that accounts for f32 rounding.
    for (a, b) in c32.iter().zip(c64.iter()) {
        assert_abs_diff_eq!(f64::from(*a), *b, epsilon = 1e-7);
    }
}

// MARK: Runtime constructor tests

#[test]
fn runtime_matches_table_n3() {
    let table = <Convolve<f32, 3> as FirDifferentiator>::central_difference();
    let runtime = Convolve::<f32, 3>::central_difference_runtime();

    let ct = table.config_ref().coefficients;
    let cr = runtime.config_ref().coefficients;

    for (a, b) in ct.iter().zip(cr.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

#[test]
fn runtime_matches_table_n5() {
    let table = <Convolve<f32, 5> as FirDifferentiator>::central_difference();
    let runtime = Convolve::<f32, 5>::central_difference_runtime();

    let ct = table.config_ref().coefficients;
    let cr = runtime.config_ref().coefficients;

    for (a, b) in ct.iter().zip(cr.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

#[test]
fn runtime_matches_table_n7() {
    let table = <Convolve<f32, 7> as FirDifferentiator>::central_difference();
    let runtime = Convolve::<f32, 7>::central_difference_runtime();

    let ct = table.config_ref().coefficients;
    let cr = runtime.config_ref().coefficients;

    for (a, b) in ct.iter().zip(cr.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

#[test]
fn runtime_matches_table_n9() {
    let table = <Convolve<f32, 9> as FirDifferentiator>::central_difference();
    let runtime = Convolve::<f32, 9>::central_difference_runtime();

    let ct = table.config_ref().coefficients;
    let cr = runtime.config_ref().coefficients;

    for (a, b) in ct.iter().zip(cr.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    }
}

#[test]
fn runtime_n11_f64_dc_rejection() {
    let filter = Convolve::<f64, 11>::central_difference_runtime();
    let coeffs = filter.config_ref().coefficients;
    let sum: f64 = coeffs.iter().sum();

    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-12);
}

#[test]
fn runtime_n11_f64_ramp_response() {
    let mut filter = Convolve::<f64, 11>::central_difference_runtime();

    for n in 0..=20 {
        let out = filter.filter(n as f64);
        if n >= 10 {
            assert_abs_diff_eq!(out, 1.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn runtime_n19_f64_dc_rejection_and_ramp() {
    let filter = Convolve::<f64, 19>::central_difference_runtime();
    let coeffs = filter.config_ref().coefficients;
    let sum: f64 = coeffs.iter().sum();
    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-12);

    let mut filter = Convolve::<f64, 19>::central_difference_runtime();
    for n in 0..=30 {
        let out = filter.filter(n as f64);
        if n >= 18 {
            assert_abs_diff_eq!(out, 1.0, epsilon = 1e-10);
        }
    }
}

#[test]
#[should_panic(expected = "N <= 19")]
fn runtime_n21_panics() {
    let _ = Convolve::<f64, 21>::central_difference_runtime();
}

#[test]
fn runtime_f64_matches_table_n5() {
    let table = Convolve::<f64, 5>::central_difference();
    let runtime = Convolve::<f64, 5>::central_difference_runtime();

    let ct = table.config_ref().coefficients;
    let cr = runtime.config_ref().coefficients;

    for (a, b) in ct.iter().zip(cr.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }
}

#[test]
fn runtime_n9_f32_does_not_panic() {
    let _ = Convolve::<f32, 9>::central_difference_runtime();
}

#[test]
#[should_panic(expected = "N must be <= 9")]
fn runtime_n11_f32_panics() {
    let _ = Convolve::<f32, 11>::central_difference_runtime();
}

// MARK: Laplacian tests

#[test]
fn laplacian_coefficients_n3() {
    let filter = Convolve::<f32, 3>::laplacian();
    let c = filter.config_ref().coefficients;

    // 1e-6 accounts for f32 literal conversion rounding.
    assert_abs_diff_eq!(c[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[1], -2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(c[2], 1.0, epsilon = 1e-6);
}

#[test]
fn laplacian_n3_dc_rejection() {
    let filter = Convolve::<f32, 3>::laplacian();
    let coeffs = filter.config_ref().coefficients;
    let sum: f32 = coeffs.iter().sum();

    // Integer-coefficient sum is exact in f32.
    assert_abs_diff_eq!(sum, 0.0, epsilon = f32::EPSILON);
}

#[test]
fn laplacian_n3_quadratic() {
    // f(x) = x², f″(x) = 2. After warm-up, output should be 2.
    let mut filter = Convolve::<f32, 3>::laplacian();

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
    let mut filter = Convolve::<f32, 3>::laplacian();

    for n in 0..=10 {
        let out = filter.filter(n as f32);
        if n >= 2 {
            assert_abs_diff_eq!(out, 0.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn laplacian_n3_constant_rejection() {
    let mut filter = Convolve::<f32, 3>::laplacian();

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
    let filter = Convolve::<f32, 5>::second_central_difference();
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
    let filter = Convolve::<f32, 5>::second_central_difference();
    let coeffs = filter.config_ref().coefficients;
    let sum: f32 = coeffs.iter().sum();

    assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-6);
}

#[test]
fn second_central_difference_n5_quadratic() {
    let mut filter = Convolve::<f32, 5>::second_central_difference();

    for n in 0..=12 {
        let out = filter.filter((n * n) as f32);
        if n >= 4 {
            assert_abs_diff_eq!(out, 2.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn second_central_difference_n5_ramp_rejection() {
    let mut filter = Convolve::<f32, 5>::second_central_difference();

    for n in 0..=12 {
        let out = filter.filter(n as f32);
        if n >= 4 {
            assert_abs_diff_eq!(out, 0.0, epsilon = 1e-5);
        }
    }
}

#[test]
fn laplacian_smoke() {
    let filter = Convolve::<f32, 3>::laplacian();
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
