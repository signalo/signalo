// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn autocorrelation_constant_signal() {
    // Input: [(1.0, 1.0)] repeated, N=1
    // Expected: dot_product = 1.0 * 1.0 = 1.0, divided by 1 = 1.0
    let mut sink: CorrelationArray<f32, 1> = CorrelationArray::default();
    sink.sink((1.0, 1.0));
    let result = sink.finalize();
    assert_eq!(result, Some(1.0));
}

#[test]
fn uncorrelated_signals() {
    // Input: [(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)] with N=3
    // dot_product = 1.0*0.0 + (-1.0)*0.0 + 1.0*0.0 = 0.0
    // correlation = 0.0 / 3 = 0.0
    let mut sink: CorrelationArray<f32, 3> = CorrelationArray::default();
    sink.sink((1.0, 0.0));
    sink.sink((-1.0, 0.0));
    sink.sink((1.0, 0.0));
    let result = sink.finalize();
    assert_eq!(result, Some(0.0));
}

#[test]
fn perfectly_correlated() {
    // Input: [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)] with N=3
    // dot_product = 2*2 + 3*3 + 4*4 = 4 + 9 + 16 = 29
    // correlation = 29 / 3 ≈ 9.667
    let mut sink: CorrelationArray<f32, 3> = CorrelationArray::default();
    sink.sink((2.0, 2.0));
    sink.sink((3.0, 3.0));
    sink.sink((4.0, 4.0));
    let result = sink.finalize();
    let val = result.unwrap();
    assert_abs_diff_eq!(val, 29.0 / 3.0, epsilon = 0.0001);
}

#[test]
fn filter_interface() {
    // Test Filter interface returns same result as Finalize
    // Input: [(1.0, 1.0), (2.0, 2.0)] with N=2
    let mut sink: CorrelationArray<f32, 2> = CorrelationArray::default();
    let out1 = sink.filter((1.0, 1.0));
    let out2 = sink.filter((2.0, 2.0));

    // After first input: dot = 1*1 = 1, count = 1, out = 1.0
    assert_abs_diff_eq!(out1, 1.0, epsilon = 1e-6);
    // After second input: dot = 1*1 + 2*2 = 5, count = 2, out = 2.5
    assert_abs_diff_eq!(out2, 2.5, epsilon = 1e-6);

    let finalize_result = sink.finalize();
    assert_eq!(finalize_result, Some(2.5));
}

#[test]
fn test_nan_propagation() {
    let mut sink: CorrelationArray<f32, 4> = CorrelationArray::default();
    let result = sink.filter((f32::NAN, 1.0));
    assert!(result.is_nan());
}

#[test]
fn test_large_values() {
    let mut sink: CorrelationArray<f32, 2> = CorrelationArray::default();
    let result = sink.filter((1e10, 1e10));
    assert!(result.is_finite());
}

#[test]
fn test_inf_propagation() {
    let mut sink: CorrelationArray<f32, 4> = CorrelationArray::default();
    let result = sink.filter((f32::INFINITY, 1.0));
    assert!(result.is_infinite());
}

#[test]
fn test_reset() {
    let mut sink: CorrelationArray<f32, 2> = CorrelationArray::default();
    sink.sink((1.0, 1.0));
    sink.sink((2.0, 2.0));
    let sink = sink.reset();
    let sink = sink;
    assert_eq!(sink.finalize(), None);
}

#[test]
fn test_n1_window() {
    let mut sink: CorrelationArray<f32, 1> = CorrelationArray::default();
    let out1 = sink.filter((3.0, 4.0));
    assert_eq!(out1, 12.0);
    let out2 = sink.filter((5.0, 6.0));
    assert_eq!(out2, 30.0);
}

#[test]
fn test_integer_type() {
    let mut sink: CorrelationArray<i32, 2> = CorrelationArray::default();
    let _out1 = sink.filter((1, 2));
    let out2 = sink.filter((3, 4));
    // dot = 1*2 + 3*4 = 14, count = 2, result = 7
    assert_eq!(out2, 7);
}

#[test]
fn test_state_mut() {
    let mut sink: CorrelationArray<f32, 4> = CorrelationArray::default();
    let state = sink.state_mut();
    state.buffer_x.push_back(1.0);
    state.buffer_y.push_back(1.0);
    state.len = 1;
    let result = sink.filter((2.0, 2.0));
    assert_eq!(result, (1.0 * 1.0 + 2.0 * 2.0) / 2.0);
}

#[test]
fn empty_sink() {
    // Empty sink should return None
    let sink: CorrelationArray<f32, 4> = CorrelationArray::default();
    let result = sink.finalize();
    assert_eq!(result, None);
}

#[test]
#[should_panic(expected = "Correlation: window size must be > 0")]
fn from_parts_zero_capacity_panics() {
    let bx = circular_buffer::FixedCircularBuffer::<f32, 0>::new();
    let by = circular_buffer::FixedCircularBuffer::<f32, 0>::new();
    let _ = Correlation::<f32, _, _>::from_parts(bx, by);
}

#[test]
#[should_panic(expected = "must equal buffer_y capacity")]
fn from_parts_capacity_mismatch_panics() {
    let bx = circular_buffer::FixedCircularBuffer::<f32, 3>::new();
    let by = circular_buffer::FixedCircularBuffer::<f32, 2>::new();
    let _ = Correlation::<f32, _, _>::from_parts(bx, by);
}

#[test]
fn test() {
    let input = [
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0,
    ];
    // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
    let mut sink: CorrelationArray<f32, 20> = CorrelationArray::default();
    for value in input {
        sink.sink((value, value));
    }
    let result = sink.finalize();
    assert_eq!(result, Some(137.5));
}
