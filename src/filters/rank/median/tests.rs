// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec;
use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;

macro_rules! test_filter {
    ($size:expr, $input:expr, $output:expr) => {
        let filter: MedianArray<_, $size> = MedianArray::default();
        let output: Vec<_> = $input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(output, $output);
    };
}

#[test]
fn single_peak_4() {
    let input = [10, 20, 30, 100, 30, 20, 10];
    let output = vec![10, 10, 20, 20, 30, 30, 20];

    test_filter!(4, input, output);
}

#[test]
fn single_peak_5() {
    let input = [10, 20, 30, 100, 30, 20, 10];
    let output = vec![10, 10, 20, 20, 30, 30, 30];
    test_filter!(5, input, output);
}

#[test]
fn single_valley_4() {
    let input = [90, 80, 70, 10, 70, 80, 90];
    let output = vec![90, 80, 80, 70, 70, 70, 70];
    test_filter!(4, input, output);
}

#[test]
fn single_valley_5() {
    let input = [90, 80, 70, 10, 70, 80, 90];
    let output = vec![90, 80, 80, 70, 70, 70, 70];
    test_filter!(5, input, output);
}

#[test]
fn single_outlier_4() {
    let input = [10, 10, 10, 100, 10, 10, 10];
    let output = vec![10, 10, 10, 10, 10, 10, 10];
    test_filter!(4, input, output);
}

#[test]
fn single_outlier_5() {
    let input = [10, 10, 10, 100, 10, 10, 10];
    let output = vec![10, 10, 10, 10, 10, 10, 10];
    test_filter!(5, input, output);
}

#[test]
fn triple_outlier_4() {
    let input = [10, 10, 100, 100, 100, 10, 10];
    let output = vec![10, 10, 10, 10, 100, 100, 10];
    test_filter!(4, input, output);
}

#[test]
fn triple_outlier_5() {
    let input = [10, 10, 100, 100, 100, 10, 10];
    let output = vec![10, 10, 10, 10, 100, 100, 100];
    test_filter!(5, input, output);
}

#[test]
fn quintuple_outlier_4() {
    let input = [10, 100, 100, 100, 100, 100, 10];
    let output = vec![10, 10, 100, 100, 100, 100, 100];
    test_filter!(4, input, output);
}

#[test]
fn quintuple_outlier_5() {
    let input = [10, 100, 100, 100, 100, 100, 10];
    let output = vec![10, 10, 100, 100, 100, 100, 100];
    test_filter!(5, input, output);
}

#[test]
fn alternating_4() {
    let input = [10, 20, 10, 20, 10, 20, 10];
    let output = vec![10, 10, 10, 10, 10, 10, 10];
    test_filter!(4, input, output);
}

#[test]
fn alternating_5() {
    let input = [10, 20, 10, 20, 10, 20, 10];
    let output = vec![10, 10, 10, 10, 10, 20, 10];
    test_filter!(5, input, output);
}

#[test]
fn ascending_4() {
    let input = [10, 20, 30, 40, 50, 60, 70];
    let output = vec![10, 10, 20, 20, 30, 40, 50];
    test_filter!(4, input, output);
}

#[test]
fn ascending_5() {
    let input = [10, 20, 30, 40, 50, 60, 70];
    let output = vec![10, 10, 20, 20, 30, 40, 50];
    test_filter!(5, input, output);
}

#[test]
fn descending_4() {
    let input = [70, 60, 50, 40, 30, 20, 10];
    let output = vec![70, 60, 60, 50, 40, 30, 20];
    test_filter!(4, input, output);
}

#[test]
fn descending_5() {
    let input = [70, 60, 50, 40, 30, 20, 10];
    let output = vec![70, 60, 60, 50, 50, 40, 30];
    test_filter!(5, input, output);
}

#[test]
fn min_max_median() {
    let input = vec![70, 50, 30, 10, 20, 40, 60];
    let mut filter: MedianArray<_, 5> = MedianArray::default();
    for input in input {
        filter.filter(input);
    }
    assert_eq!(filter.min(), Some(10));
    assert_eq!(filter.max(), Some(60));
    assert_eq!(filter.median(), Some(30));
}

#[test]
#[should_panic(expected = "window size N must be > 0")]
fn zero_window_panics() {
    let _: MedianArray<f32, 0> = MedianArray::default();
}

#[test]
fn max_is_not_most_recently_inserted() {
    // Feed values such that the max is not the last inserted element.
    let mut filter: MedianArray<_, 3> = MedianArray::default();
    filter.filter(1);
    filter.filter(100);
    filter.filter(2);
    // Window = [1, 100, 2]; max must be 100, not 2.
    assert_eq!(filter.max(), Some(100));
}

fn get_input() -> Vec<f32> {
    vec![
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0, 106.0, 5.0,
        26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0,
        11.0, 24.0, 24.0,
    ]
}

fn get_output() -> Vec<f32> {
    vec![
        0.0, 0.0, 1.0, 1.0, 2.0, 5.0, 7.0, 5.0, 8.0, 8.0, 14.0, 9.0, 9.0, 9.0, 14.0, 9.0, 12.0,
        17.0, 17.0, 12.0, 12.0, 15.0, 15.0, 10.0, 15.0, 15.0, 15.0, 18.0, 18.0, 18.0, 18.0, 18.0,
        18.0, 18.0, 13.0, 13.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 29.0, 16.0, 16.0, 16.0, 16.0,
        16.0, 16.0, 24.0,
    ]
}

#[test]
#[should_panic(expected = "Median: window size N must be > 0")]
fn from_parts_empty_buffer_panics() {
    let empty: &mut [ListNode<f32>] = &mut [];
    let _ = MedianRefMut::from_parts(empty);
}

#[cfg(feature = "alloc")]
#[test]
#[should_panic(expected = "Median: window size N must be > 0")]
fn median_vec_new_zero_panics() {
    let _ = MedianVec::<f32>::new(0);
}

#[cfg(feature = "alloc")]
#[test]
fn median_vec_new_filters_correctly() {
    let mut filter = MedianVec::<i32>::new(3);
    assert_eq!(filter.filter(10), 10);
    assert_eq!(filter.filter(20), 10);
    assert_eq!(filter.filter(30), 20);
    assert_eq!(filter.filter(100), 30);
}

#[test]
fn median_ref_mut_filters_correctly() {
    let mut buffer: [ListNode<i32>; 3] = core::array::from_fn(|index| ListNode {
        value: None,
        previous: (index + 2) % 3,
        next: (index + 1) % 3,
    });
    let mut filter = MedianRefMut::from_parts(&mut buffer);
    assert_eq!(filter.filter(10), 10);
    assert_eq!(filter.filter(20), 10);
    assert_eq!(filter.filter(30), 20);
    assert_eq!(filter.filter(100), 30);
}

#[test]
fn test() {
    let filter: MedianArray<_, 5> = MedianArray::default();
    // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
    let input = get_input();
    let output: Vec<_> = input
        .iter()
        .scan(filter, |filter, &input| Some(filter.filter(input)))
        .collect();
    assert_abs_diff_eq!(output.as_slice(), get_output().as_slice(), epsilon = 0.001);
}
