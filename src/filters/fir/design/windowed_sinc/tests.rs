// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;

#[test]
fn lowpass_normalizes_sum_to_one() {
    let mut taps = [0.0_f64; 33];

    super::hann::lowpass(&mut taps, 0.2);

    assert_abs_diff_eq!(taps.into_iter().sum::<f64>(), 1.0, epsilon = 1e-12);
}

#[cfg(feature = "alloc")]
#[test]
fn in_place_matches_vec() {
    let expected = super::hann::lowpass_vec(32, 0.2_f64);
    let mut actual = [0.0; 32];

    super::hann::lowpass(&mut actual, 0.2);

    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn lowpass_is_symmetric_for_odd_and_even_lengths() {
    let mut even_taps = [0.0_f64; 32];
    super::hamming::lowpass(&mut even_taps, 0.2);
    for index in 0..even_taps.len() {
        assert_abs_diff_eq!(
            even_taps[index],
            even_taps[even_taps.len() - 1 - index],
            epsilon = 1e-12
        );
    }

    let mut odd_taps = [0.0_f64; 33];
    super::hamming::lowpass(&mut odd_taps, 0.2);
    for index in 0..odd_taps.len() {
        assert_abs_diff_eq!(
            odd_taps[index],
            odd_taps[odd_taps.len() - 1 - index],
            epsilon = 1e-12
        );
    }
}

#[test]
#[should_panic(expected = "N=1 is degenerate")]
fn lowpass_rejects_single_tap() {
    let mut taps = [0.0_f64; 1];

    super::hann::lowpass(&mut taps, 0.2);
}

#[test]
fn lowpass_leaves_zero_sum_taps_zeroed() {
    let mut taps = [0.0_f64; 3];

    super::lowpass_unnormalized(&mut taps, 0.2, |_, _| 0.0);
    super::normalize(&mut taps, super::Normalization::PassbandGain);

    assert_eq!(taps, [0.0; 3]);
}

#[cfg(feature = "alloc")]
#[test]
fn window_lowpass_vec_helpers_match_in_place() {
    macro_rules! check_window {
        ($design:ident) => {
            let actual = super::$design::lowpass_vec(33, 0.2_f64);
            let mut expected = [0.0; 33];
            super::$design::lowpass(&mut expected, 0.2);

            for (actual, expected) in actual.iter().zip(expected) {
                assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
            }
        };
    }

    check_window!(rectangular);
    check_window!(triangular);
    check_window!(hann);
    check_window!(hamming);
    check_window!(blackman);
    check_window!(blackman_harris);
    check_window!(flat_top);
}

#[test]
fn window_family_helpers_have_expected_reference_gain() {
    macro_rules! check_window {
        ($design:ident) => {
            let mut highpass = [0.0_f64; 33];
            super::$design::highpass(&mut highpass, 0.2);
            assert_abs_diff_eq!(super::gain_at_freq(&highpass, 0.5), 1.0, epsilon = 1e-10);

            let mut bandpass = [0.0_f64; 33];
            super::$design::bandpass(&mut bandpass, 0.1, 0.3);
            assert_abs_diff_eq!(super::gain_at_freq(&bandpass, 0.2), 1.0, epsilon = 1e-10);

            let mut bandstop = [0.0_f64; 33];
            super::$design::bandstop(&mut bandstop, 0.1, 0.3);
            assert_abs_diff_eq!(bandstop.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
        };
    }

    check_window!(rectangular);
    check_window!(triangular);
    check_window!(hann);
    check_window!(hamming);
    check_window!(blackman);
    check_window!(blackman_harris);
    check_window!(flat_top);
    check_window!(kaiser);
}

#[cfg(feature = "alloc")]
#[test]
fn window_family_vec_helpers_match_in_place() {
    macro_rules! check_window {
        ($design:ident) => {
            let highpass_vec = super::$design::highpass_vec(33, 0.2_f64);
            let mut highpass = [0.0; 33];
            super::$design::highpass(&mut highpass, 0.2);
            for (actual, expected) in highpass_vec.iter().zip(highpass) {
                assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
            }

            let bandpass_vec = super::$design::bandpass_vec(33, 0.1_f64, 0.3_f64);
            let mut bandpass = [0.0; 33];
            super::$design::bandpass(&mut bandpass, 0.1, 0.3);
            for (actual, expected) in bandpass_vec.iter().zip(bandpass) {
                assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
            }

            let bandstop_vec = super::$design::bandstop_vec(33, 0.1_f64, 0.3_f64);
            let mut bandstop = [0.0; 33];
            super::$design::bandstop(&mut bandstop, 0.1, 0.3);
            for (actual, expected) in bandstop_vec.iter().zip(bandstop) {
                assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
            }
        };
    }

    check_window!(rectangular);
    check_window!(triangular);
    check_window!(hann);
    check_window!(hamming);
    check_window!(blackman);
    check_window!(blackman_harris);
    check_window!(flat_top);
    check_window!(kaiser);
}

#[cfg(feature = "alloc")]
#[test]
fn kaiser_lowpass_vec_helper_matches_default_beta() {
    let fc = 0.2;
    let actual = super::kaiser::lowpass_vec(33, fc);
    let expected = super::kaiser::lowpass_with_beta_vec(33, 6.0, fc);

    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[cfg(feature = "alloc")]
#[test]
fn kaiser_lowpass_with_beta_vec_helper_matches_in_place() {
    let beta = 8.0_f64;
    let fc = 0.2;
    let actual = super::kaiser::lowpass_with_beta_vec(33, beta, fc);
    let mut expected = [0.0; 33];
    super::kaiser::lowpass_with_beta(&mut expected, beta, fc);

    for (actual, expected) in actual.iter().zip(expected) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn kaiser_with_beta_helpers_have_expected_reference_gain() {
    let beta = 8.0_f64;

    let mut highpass = [0.0_f64; 33];
    super::kaiser::highpass_with_beta(&mut highpass, beta, 0.2);
    assert_abs_diff_eq!(super::gain_at_freq(&highpass, 0.5), 1.0, epsilon = 1e-10);

    let mut bandpass = [0.0_f64; 33];
    super::kaiser::bandpass_with_beta(&mut bandpass, beta, 0.1, 0.3);
    assert_abs_diff_eq!(super::gain_at_freq(&bandpass, 0.2), 1.0, epsilon = 1e-10);

    let mut bandstop = [0.0_f64; 33];
    super::kaiser::bandstop_with_beta(&mut bandstop, beta, 0.1, 0.3);
    assert_abs_diff_eq!(bandstop.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
}

#[cfg(feature = "alloc")]
#[test]
fn kaiser_with_beta_vec_helpers_match_in_place() {
    let beta = 8.0_f64;

    let highpass_vec = super::kaiser::highpass_with_beta_vec(33, beta, 0.2);
    let mut highpass = [0.0; 33];
    super::kaiser::highpass_with_beta(&mut highpass, beta, 0.2);
    for (actual, expected) in highpass_vec.iter().zip(highpass) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
    }

    let bandpass_vec = super::kaiser::bandpass_with_beta_vec(33, beta, 0.1, 0.3);
    let mut bandpass = [0.0; 33];
    super::kaiser::bandpass_with_beta(&mut bandpass, beta, 0.1, 0.3);
    for (actual, expected) in bandpass_vec.iter().zip(bandpass) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
    }

    let bandstop_vec = super::kaiser::bandstop_with_beta_vec(33, beta, 0.1, 0.3);
    let mut bandstop = [0.0; 33];
    super::kaiser::bandstop_with_beta(&mut bandstop, beta, 0.1, 0.3);
    for (actual, expected) in bandstop_vec.iter().zip(bandstop) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn bandpass_signed_centre_gain_preserves_polarity() {
    const N: usize = 9;
    let f_lo: f64 = 0.1;
    let f_hi: f64 = 0.3;
    let f_c = (f_lo + f_hi) / 2.0;
    let neg_window = |_: usize, _: usize| -0.5_f64;

    let mut raw = [0.0_f64; N];
    let mut norm = [0.0_f64; N];
    super::bandpass_unnormalized(&mut raw, f_lo, f_hi, neg_window);
    super::bandpass_unnormalized(&mut norm, f_lo, f_hi, neg_window);
    super::normalize_bandpass(&mut norm, f_lo, f_hi);

    let two_pi = 2.0 * core::f64::consts::PI;
    let m = (N as f64 - 1.0) / 2.0;

    let a_raw: f64 = raw.iter().enumerate().fold(0.0, |acc, (k, &hk)| {
        acc + hk * (two_pi * f_c * (k as f64 - m)).cos()
    });
    assert!(a_raw < 0.0, "expected negative raw A(f_c), got {a_raw}");

    let a_norm: f64 = norm.iter().enumerate().fold(0.0, |acc, (k, &hk)| {
        acc + hk * (two_pi * f_c * (k as f64 - m)).cos()
    });
    assert_abs_diff_eq!(a_norm, 1.0, epsilon = 1e-10);
}
