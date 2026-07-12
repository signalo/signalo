// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;

use super::taps;
use crate::filters::fir::design::{
    len_from_span, raised_cosine, root_raised_cosine, Normalization,
};

#[test]
#[allow(clippy::unreadable_literal)]
fn taps_match_reference() {
    const LEN: usize = len_from_span(4, 4);
    let expected = [
        2.8607931050595666e-2,
        -1.10549735681297e-2,
        -6.769_630_089_467_077e-2,
        -9.44759673964384e-2,
        -4.241_671_436_495_778e-2,
        1.0361045396849711e-1,
        3.044_005_078_498_384e-1,
        4.793_719_664_108_898e-1,
        5.487_429_654_786_262e-1,
        4.793_719_664_108_898e-1,
        3.044_005_078_498_384e-1,
        1.0361045396849711e-1,
        -4.241_671_436_495_778e-2,
        -9.44759673964384e-2,
        -6.769_630_089_467_077e-2,
        -1.10549735681297e-2,
        2.8607931050595666e-2,
    ];

    let mut actual = [0.0; LEN];
    taps(&mut actual, 4, 4, 0.35);

    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }
}

#[cfg(feature = "alloc")]
#[test]
fn in_place_matches_vec() {
    const LEN: usize = len_from_span(6, 8);
    let expected = super::taps_with_norm_vec(6, 8, 0.25_f64, Normalization::PassbandGain);
    let mut actual = [0.0; LEN];

    super::taps_with_norm(&mut actual, 6, 8, 0.25, Normalization::PassbandGain);

    assert_eq!(actual.as_slice(), expected.as_slice());
}

#[test]
fn unit_energy_normalizes_energy_to_one() {
    const LEN: usize = len_from_span(6, 8);
    let mut taps = [0.0; LEN];
    super::taps(&mut taps, 6, 8, 0.35);
    let energy = taps.into_iter().map(|tap| tap * tap).sum::<f64>().sqrt();

    assert_abs_diff_eq!(energy, 1.0, epsilon = 1e-12);
}

#[test]
fn passband_gain_normalizes_sum_to_one() {
    const LEN: usize = len_from_span(6, 8);
    let mut taps = [0.0; LEN];
    super::taps_with_norm(&mut taps, 6, 8, 0.35, Normalization::PassbandGain);
    let sum = taps.into_iter().sum::<f64>();

    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
}

#[test]
fn matched_filter_gives_raised_cosine_at_symbol_instants() {
    const SPAN: usize = 64;
    const SPS: usize = 8;
    const LEN: usize = len_from_span(SPAN, SPS);
    const CONV_LEN: usize = 2 * LEN - 1;
    let rolloff = 0.35;
    let mut rrc = [0.0; LEN];
    let mut rc = [0.0; LEN];
    let mut conv = [0.0; CONV_LEN];

    root_raised_cosine::taps_with_norm(&mut rrc, SPAN, SPS, rolloff, Normalization::None);
    raised_cosine::taps_with_norm(&mut rc, SPAN, SPS, rolloff, Normalization::None);

    // Convolve the root-raised-cosine with itself to get a raised-cosine.
    for (i, &a) in rrc.iter().enumerate() {
        for (j, &b) in rrc.iter().enumerate() {
            conv[i + j] += a * b;
        }
    }

    // The central `n` samples of the convolution should match rc (up to a
    // common scalar, since both are un-normalized).  Check only at symbol
    // instants where ISI nulling should hold:
    let conv_center = rrc.len() - 1;
    let rc_center = (rc.len() - 1) / 2;
    let scale = conv[conv_center] / rc[rc_center];

    for symbol in 0..=4 {
        assert_abs_diff_eq!(
            conv[conv_center + symbol * SPS] / scale,
            rc[rc_center + symbol * SPS],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            conv[conv_center - symbol * SPS] / scale,
            rc[rc_center - symbol * SPS],
            epsilon = 1e-6
        );
    }
}

#[test]
#[should_panic(expected = "tap count must equal span * sps + 1")]
fn incompatible_length_panics() {
    let mut taps = [0.0; 16];
    super::taps(&mut taps, 4, 4, 0.35);
}
