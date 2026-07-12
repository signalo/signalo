// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;

use super::{taps, taps_with_norm};
use crate::filters::fir::design::{len_from_span, Normalization};

#[test]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
fn taps_match_reference() {
    const LEN: usize = len_from_span(4, 4);
    let expected = [
        7.438_679_225_487_701e-7,
        2.012_670_875_194_407e-5,
        3.1727481185161663e-4,
        2.9462688262236955e-3,
        1.6399631027448684e-2,
        5.628_444_112_335_107e-2,
        1.2468272901724888e-1,
        1.9074915548782734e-1,
        2.1719925825874842e-1,
        1.9074915548782734e-1,
        1.2468272901724888e-1,
        5.628_444_112_335_107e-2,
        1.6399631027448684e-2,
        2.9462688262236955e-3,
        3.1727481185161663e-4,
        2.012_670_875_194_407e-5,
        7.438_679_225_487_701e-7,
    ];

    let mut actual = [0.0; LEN];
    taps(&mut actual, 4, 4, 0.4);

    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }
}

#[cfg(feature = "alloc")]
#[test]
fn in_place_matches_vec() {
    const LEN: usize = len_from_span(6, 8);
    let expected = super::taps_vec(6, 8, 0.4_f64);
    let mut actual = [0.0; LEN];

    taps(&mut actual, 6, 8, 0.4);

    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn passband_gain_normalizes_sum_to_one() {
    const LEN: usize = len_from_span(6, 8);
    let mut taps = [0.0; LEN];
    super::taps(&mut taps, 6, 8, 0.4);
    let sum = taps.into_iter().sum::<f64>();

    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
}

#[test]
fn explicit_unit_energy_normalizes_energy_to_one() {
    const LEN: usize = len_from_span(6, 8);
    let mut taps = [0.0; LEN];
    taps_with_norm(&mut taps, 6, 8, 0.4, Normalization::UnitEnergy);
    let energy = taps.into_iter().map(|tap| tap * tap).sum::<f64>().sqrt();

    assert_abs_diff_eq!(energy, 1.0, epsilon = 1e-12);
}

#[test]
#[should_panic(expected = "tap count must equal span * sps + 1")]
fn incompatible_length_panics() {
    let mut taps = [0.0; 16];
    super::taps(&mut taps, 4, 4, 0.4);
}
