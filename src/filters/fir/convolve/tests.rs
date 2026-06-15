// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec;
use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;
use crate::util::test_fixtures::collatz as get_input;

#[test]
#[should_panic(expected = "window size N must be > 0")]
fn zero_window_panics() {
    let _ = Convolve::<f32, 0>::with_config(Config { coefficients: [] });
}

fn get_output() -> Vec<f32> {
    vec![
        0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -3.0, 6.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0, -13.0, 8.0,
        8.0, 0.0, -13.0, 0.0, 8.0, 0.0, -5.0, 13.0, -13.0, 101.0, 69.0, -72.0, -90.0, 88.0, -101.0,
        21.0, -13.0, 0.0, 8.0, 0.0, 0.0, 13.0, -26.0, 101.0, -101.0, 21.0, -13.0, 0.0, 0.0, 88.0,
        -93.0, 13.0, 0.0,
    ]
}

#[test]
fn test() {
    // Effectively calculates the derivative:
    let filter = Convolve::with_config(Config {
        coefficients: [1.000, -1.000],
    });
    let input = get_input();
    let output: Vec<_> = input
        .iter()
        .scan(filter, |filter, &input| Some(filter.filter(input)))
        .collect();
    assert_abs_diff_eq!(output.as_slice(), get_output().as_slice(), epsilon = 0.001);
}

#[test]
fn test_filter_buffer_filling() {
    // With zero-padded cold-start (tap buffer pre-filled with zeros),
    // the first N outputs are partial convolutions.
    let config = Config {
        coefficients: [0.25, 0.25, 0.25, 0.25],
    };
    let mut filter = Convolve::<f32, 4>::with_config(config);

    // taps=[0,0,0,0] → push(4.0) → taps=[0,0,0,4.0] → sum=1.0
    let output1 = filter.filter(4.0);
    assert_abs_diff_eq!(output1, 1.0, epsilon = 0.0001);

    // taps=[0,0,0,4.0] → push(8.0) → taps=[0,0,4.0,8.0] → sum=3.0
    let output2 = filter.filter(8.0);
    assert_abs_diff_eq!(output2, 3.0, epsilon = 0.0001);
}

#[test]
fn cold_start_is_zero_padded_partial_convolution() {
    let h = [0.5_f32, -0.25, 0.125];
    let mut filter = Convolve::<f32, 3>::with_config(Config { coefficients: h });
    assert!((filter.filter(4.0) - 0.5 * 4.0).abs() < 1e-7);
    assert!((filter.filter(8.0) - (0.5 * 8.0 + -0.25 * 4.0)).abs() < 1e-7);
    assert!((filter.filter(2.0) - (0.5 * 2.0 + -0.25 * 8.0 + 0.125 * 4.0)).abs() < 1e-7);
}

#[test]
fn impulse_response() {
    // Verify the canonical FIR convolution contract y[n] = Σ h[k]·x[n−k]
    // with zero-padding (x[n] = 0 for n < 0).
    // The impulse response must reproduce h[0], h[1], …, h[N−1] exactly.
    let h = [0.1, 0.2, 0.3, 0.4, 0.5_f32];
    let mut filter = Convolve::<f32, 5>::with_config(Config { coefficients: h });
    let response: Vec<f32> = [1.0_f32]
        .into_iter()
        .chain(core::iter::repeat(0.0).take(h.len()))
        .map(|x| filter.filter(x))
        .collect();
    assert_abs_diff_eq!(response[0], h[0], epsilon = 1e-7);
    assert_abs_diff_eq!(response[1], h[1], epsilon = 1e-7);
    assert_abs_diff_eq!(response[2], h[2], epsilon = 1e-7);
    assert_abs_diff_eq!(response[3], h[3], epsilon = 1e-7);
    assert_abs_diff_eq!(response[4], h[4], epsilon = 1e-7);
    // After N+1 samples the buffer is fully zero again
    assert_abs_diff_eq!(response[5], 0.0, epsilon = 1e-7);
}

#[test]
fn integer_convolution() {
    // Integer convolution must work without overflow surprises.
    // 2-tap moving sum: output(n) = x[n] + x[n-1]
    let mut filter = Convolve::<i32, 2>::with_config(Config {
        coefficients: [1, 1],
    });
    // taps=[0,0], push(4): taps=[0,4], output = 0*1 + 4*1 = 4
    assert_eq!(filter.filter(4), 4);
    // taps=[0,4], push(6): taps=[4,6], output = 4*1 + 6*1 = 10
    assert_eq!(filter.filter(6), 10);
    // taps=[4,6], push(8): taps=[6,8], output = 6*1 + 8*1 = 14
    assert_eq!(filter.filter(8), 14);
    // Reset and check cold-start zero-padding
    let mut filter2 = filter.reset();
    assert_eq!(filter2.filter(1), 1);
    assert_eq!(filter2.filter(2), 3);
}

#[test]
#[should_panic(expected = "denominator magnitude")]
fn tiny_sum_rejected() {
    let _ = Convolve::<f32, 3>::normalized(Config {
        coefficients: [1.0, -1.0, f32::from_bits(1)],
    });
}

macro_rules! normalized_case {
    ($name:ident, $n:literal, $input:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let filter = Convolve::<f32, $n>::normalized(Config {
                coefficients: $input,
            });
            let c = filter.config_ref().coefficients;
            let expected: [f32; $n] = $expected;
            for (a, e) in c.iter().zip(expected.iter()) {
                assert_abs_diff_eq!(a, e, epsilon = 1e-4);
            }
        }
    };
}
normalized_case!(
    normalized_positive,
    3,
    [2.0, 4.0, 6.0],
    [1.0 / 6.0, 1.0 / 3.0, 0.5]
);
normalized_case!(normalized_negative, 2, [-1.0, 0.5], [2.0, -1.0]);
normalized_case!(normalized_zero_sum, 3, [1.0, -1.0, 0.0], [1.0, -1.0, 0.0]);
