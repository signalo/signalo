// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec;
use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use crate::traits::Filter;

use super::super::Decomposition;

use super::*;

fn get_input() -> Vec<f32> {
    // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
    vec![
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0, 106.0, 5.0,
        26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0,
        11.0, 24.0, 24.0,
    ]
}

fn get_analysis_2() -> Vec<Decomposition<f32>> {
    get_low_2()
        .into_iter()
        .zip(get_high_2())
        .map(|(low, high)| Decomposition { low, high })
        .collect()
}

fn get_low_2() -> Vec<f32> {
    vec![
        0.0, 0.5, 4.0, 4.5, 3.5, 6.5, 12.0, 14.5, 16.0, 12.5, 10.0, 11.5, 9.0, 13.0, 17.0, 10.5,
        8.0, 16.0, 20.0, 13.5, 7.0, 11.0, 15.0, 12.5, 16.5, 16.5, 60.5, 145.5, 144.0, 63.0, 62.0,
        55.5, 15.5, 19.5, 13.0, 17.0, 21.0, 21.0, 27.5, 21.0, 58.5, 58.5, 18.5, 22.5, 16.0, 16.0,
        60.0, 57.5, 17.5, 24.0,
    ]
}

fn get_high_2() -> Vec<f32> {
    vec![
        0.0, 0.5, 3.0, -2.5, 1.5, 1.5, 4.0, -1.5, 3.0, -6.5, 4.0, -2.5, 0.0, 4.0, 0.0, -6.5, 4.0,
        4.0, 0.0, -6.5, 0.0, 4.0, 0.0, -2.5, 6.5, -6.5, 50.5, 34.5, -36.0, -45.0, 44.0, -50.5,
        10.5, -6.5, 0.0, 4.0, 0.0, 0.0, 6.5, -13.0, 50.5, -50.5, 10.5, -6.5, 0.0, 0.0, 44.0, -46.5,
        6.5, 0.0,
    ]
}

fn get_synthesis_2() -> Vec<f32> {
    // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
    vec![
        0.0, 0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
        12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0, 106.0,
        5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0,
        104.0, 11.0, 24.0,
    ]
}

fn get_analysis_4() -> Vec<Decomposition<f32>> {
    get_low_4()
        .into_iter()
        .zip(get_high_4())
        .map(|(low, high)| Decomposition { low, high })
        .collect()
}

fn get_low_4() -> Vec<f32> {
    vec![
        0.000, 0.342, 2.982, 4.982, 3.908, 5.366, 10.806, 14.714, 15.982, 13.884, 10.152, 10.567,
        10.067, 11.275, 16.464, 13.292, 7.603, 13.007, 20.196, 16.292, 7.871, 8.542, 14.464,
        14.025, 14.775, 17.232, 46.553, 126.609, 160.032, 88.401, 47.493, 57.377, 26.990, 10.912,
        15.792, 14.542, 20.464, 21.732, 25.440, 24.250, 45.423, 65.363, 31.179, 13.912, 18.792,
        14.810, 46.053, 66.345, 29.722, 14.619,
    ]
}

fn get_high_4() -> Vec<f32> {
    vec![
        0.000, -0.092, -0.799, -0.701, 3.025, -2.732, -0.458, -0.701, 2.933, -1.335, 4.567, -5.982,
        3.982, -2.440, -2.000, 3.922, 2.518, -7.172, 0.732, 3.922, 3.250, -5.172, -2.000, 3.190,
        0.060, -3.768, -1.553, -36.004, 23.831, 49.800, -10.141, -43.493, 53.381, -38.553, 10.422,
        -5.172, -2.000, 2.732, -1.190, -0.871, 1.697, -24.887, 57.821, -38.553, 10.422, -4.440,
        -8.053, -13.490, 52.113, -35.010,
    ]
}

fn get_synthesis_4() -> Vec<f32> {
    // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
    vec![
        0.000, 0.000, -0.000, 0.000, 1.000, 7.000, 2.000, 5.000, 8.000, 16.000, 13.000, 19.000,
        6.000, 14.000, 9.000, 9.000, 17.000, 17.000, 4.000, 12.000, 20.000, 20.000, 7.000, 7.000,
        15.000, 15.000, 10.001, 23.000, 10.000, 111.001, 180.001, 108.000, 18.000, 106.000, 5.000,
        26.001, 13.000, 13.000, 21.000, 21.000, 21.000, 34.000, 8.000, 109.000, 7.999, 29.000,
        16.000, 15.999, 16.000, 104.000,
    ]
}

fn split_analysis(output: &[Decomposition<f32>]) -> (Vec<f32>, Vec<f32>) {
    let low: Vec<_> = output
        .iter()
        .map(|Decomposition { low, .. }| *low)
        .collect();
    let high: Vec<_> = output
        .iter()
        .map(|Decomposition { high, .. }| *high)
        .collect();
    (low, high)
}

fn with_padding<T: Clone>(vec: Vec<T>, prefix: usize, suffix: usize) -> Vec<T> {
    debug_assert!(!vec.is_empty());

    let len = vec.len();
    let prefix_item = vec[0].clone();
    let suffix_item = vec[len - 1].clone();

    let prefix_iter = ::core::iter::repeat(prefix_item).take(prefix);
    let suffix_iter = ::core::iter::repeat(suffix_item).take(suffix);

    prefix_iter.chain(vec).chain(suffix_iter).collect()
}

fn without_padding<T: Clone>(vec: Vec<T>, prefix: usize, suffix: usize) -> Vec<T> {
    debug_assert!(vec.len() >= prefix + suffix);

    let take_len = vec.len() - prefix - suffix;

    vec.into_iter().skip(prefix).take(take_len).collect()
}

#[test]
fn daubechies_analysis_2() {
    const PADDING: usize = 2;

    let input = with_padding(get_input(), PADDING, PADDING);

    let analyze: AnalyzeArray<f32, 2> = AnalyzeArray::daubechies();
    let padded_analysis: Vec<_> = input
        .into_iter()
        .scan(analyze, |filter, input| Some(filter.filter(input)))
        .collect();
    let analysis = without_padding(padded_analysis, PADDING, PADDING);
    let (low, high) = split_analysis(&analysis);

    assert_abs_diff_eq!(low.as_slice(), get_low_2().as_slice(), epsilon = 0.001);
    assert_abs_diff_eq!(high.as_slice(), get_high_2().as_slice(), epsilon = 0.001);
}

#[test]
fn daubechies_synthesis_2() {
    const PADDING: usize = 2;

    let input = with_padding(get_analysis_2(), PADDING, PADDING);

    let synthesize: SynthesizeArray<f32, 2> = SynthesizeArray::daubechies();
    let padded_synthesis: Vec<_> = input
        .into_iter()
        .scan(synthesize, |filter, input| Some(filter.filter(input)))
        .collect();
    let synthesis = without_padding(padded_synthesis, PADDING, PADDING);

    assert_abs_diff_eq!(
        synthesis.as_slice(),
        get_synthesis_2().as_slice(),
        epsilon = 0.001
    );
}

#[test]
fn daubechies_analysis_4() {
    const PADDING: usize = 4;

    let input = with_padding(get_input(), PADDING, PADDING);

    let analyze: AnalyzeArray<f32, 4> = AnalyzeArray::daubechies();

    let padded_analysis: Vec<_> = input
        .into_iter()
        .scan(analyze, |filter, input| Some(filter.filter(input)))
        .collect();
    let analysis = without_padding(padded_analysis, PADDING, PADDING);
    let (low, high) = split_analysis(&analysis);

    assert_abs_diff_eq!(low.as_slice(), get_low_4().as_slice(), epsilon = 0.001);
    assert_abs_diff_eq!(high.as_slice(), get_high_4().as_slice(), epsilon = 0.001);
}

#[test]
fn daubechies_synthesis_4() {
    const PADDING: usize = 4;

    let input = with_padding(get_analysis_4(), PADDING, PADDING);

    let synthesize: SynthesizeArray<f32, 4> = SynthesizeArray::daubechies();

    let padded_synthesis: Vec<_> = input
        .into_iter()
        .scan(synthesize, |filter, input| Some(filter.filter(input)))
        .collect();
    let synthesis = without_padding(padded_synthesis, PADDING, PADDING);

    assert_abs_diff_eq!(
        synthesis.as_slice(),
        get_synthesis_4().as_slice(),
        epsilon = 0.001
    );
}
