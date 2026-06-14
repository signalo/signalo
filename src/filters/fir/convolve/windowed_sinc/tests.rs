// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;

fn feed_dc<T, const N: usize>(filter: &mut Convolve<T, N>) -> T
where
    T: num_traits::Num + Clone,
{
    (0..(4 * N)).fold(T::zero(), |_, _| filter.filter(T::one()))
}

#[cfg(any(feature = "libm", feature = "std"))]
fn feed_sine<T: Float, const N: usize>(
    filter: &mut Convolve<T, N>,
    freq: T,
    n: usize,
) -> Vec<T> {
    let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
    let mut outputs = Vec::with_capacity(n);
    for i in 0..n {
        let input = (two_pi * freq * T::from(i).unwrap()).sin();
        outputs.push(filter.filter(input));
    }
    outputs
}

// MARK: - Lowpass tests

#[cfg(any(feature = "libm", feature = "std"))]
mod lowpass {
    use approx::assert_abs_diff_eq;

    use super::super::*;
    use super::{feed_dc, feed_sine};

    const FC: f64 = 0.25;
    const FC_STOP: f64 = 0.2;

    #[test]
    fn coefficient_symmetry() {
        macro_rules! check_symmetry {
            ($ty:ident) => {
                let c = $ty::<Convolve<f64, 9>>::lowpass(FC)
                    .config_ref()
                    .coefficients;
                for k in 0..9 {
                    assert_abs_diff_eq!(c[k], c[8 - k], epsilon = 1e-12);
                }
            };
        }

        check_symmetry!(RectangularSinc);
        check_symmetry!(TriangularSinc);
        check_symmetry!(HannSinc);
        check_symmetry!(HammingSinc);
        check_symmetry!(BlackmanSinc);
        check_symmetry!(BlackmanHarrisSinc);
        check_symmetry!(FlatTopSinc);
        check_symmetry!(KaiserSinc);
    }

    #[test]
    fn dc_gain() {
        let mut filter = HannSinc::<Convolve<f64, 33>>::lowpass(FC);
        let out = feed_dc(&mut filter);
        assert_abs_diff_eq!(out, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn stopband_rejection() {
        let mut filter = HannSinc::<Convolve<f64, 33>>::lowpass(FC_STOP);
        let f_stop = 0.4;
        let outputs = feed_sine(&mut filter, f_stop, 200);
        let steady = &outputs[33..];
        let mean_sq: f64 = steady.iter().map(|x| x * x).sum::<f64>() / steady.len() as f64;
        let rms = mean_sq.sqrt();
        assert!(rms < 1e-2, "stopband RMS gain {} >= 1e-2", rms);
    }

    #[test]
    fn golden_table_n9() {
        let win = |k: usize, n: usize| -> f64 {
            let two_pi = core::f64::consts::PI * 2.0;
            0.5 * (1.0 - (two_pi * k as f64 / ((n - 1) as f64)).cos())
        };
        let reference = sinc_lowpass::<f64, 9>(FC, win, true);
        let filter = HannSinc::<Convolve<f64, 9>>::lowpass(FC);
        let actual = filter.config_ref().coefficients;
        for (a, b) in actual.iter().zip(reference.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn determinism() {
        let a = HannSinc::<Convolve<f64, 33>>::lowpass(FC);
        let b = HannSinc::<Convolve<f64, 33>>::lowpass(FC);
        for (ca, cb) in a
            .config_ref()
            .coefficients
            .iter()
            .zip(b.config_ref().coefficients.iter())
        {
            assert_abs_diff_eq!(ca, cb, epsilon = 1e-15);
        }
    }

    #[test]
    #[should_panic(expected = "N=1 is degenerate")]
    fn n1_panics() {
        let _ = HannSinc::<Convolve<f64, 1>>::lowpass(0.25);
    }

    #[test]
    fn hz_convenience() {
        let a = HannSinc::<Convolve<f64, 9>>::lowpass(FC);
        let b = HannSinc::<Convolve<f64, 9>>::lowpass_hz(1.0, FC);
        for (ca, cb) in a
            .config_ref()
            .coefficients
            .iter()
            .zip(b.config_ref().coefficients.iter())
        {
            assert_abs_diff_eq!(ca, cb, epsilon = 1e-15);
        }
    }

    #[test]
    fn unnormalized_differs_from_normalized() {
        let norm = HannSinc::<Convolve<f64, 9>>::lowpass(FC);
        let raw = HannSinc::<Convolve<f64, 9>>::lowpass_unnormalized(FC);
        let c_norm = norm.config_ref().coefficients;
        let c_raw = raw.config_ref().coefficients;
        let sum_norm: f64 = c_norm.iter().sum();
        let sum_raw: f64 = c_raw.iter().sum();
        assert_abs_diff_eq!(sum_norm, 1.0, epsilon = 1e-12);
        assert!(
            (sum_raw - 1.0).abs() > 1e-6,
            "unnormalized sum should differ from 1"
        );
    }
}

// MARK: - Highpass tests

#[cfg(any(feature = "libm", feature = "std"))]
mod highpass {
    use approx::assert_abs_diff_eq;

    use super::super::*;

    const FC: f64 = 0.25;

    #[test]
    fn signed_nyquist_alt_sum_is_one() {
        let filter = HannSinc::<Convolve<f64, 33>>::highpass(FC);
        let coeffs = filter.config_ref().coefficients;

        let h0 = gain_at_freq(&coeffs, 0.0);
        assert!(h0.abs() < 1e-3, "H_hp(0) = {}, expected ~0", h0);

        let alt_sum: f64 =
            coeffs.iter().enumerate().fold(
                0.0,
                |acc, (k, &hk)| {
                    if k % 2 == 0 {
                        acc + hk
                    } else {
                        acc - hk
                    }
                },
            );
        assert_abs_diff_eq!(alt_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn signed_nyquist_gain_for_various_n() {
        macro_rules! check_alt_sum_for_n {
            ($n:literal) => {{
                let filter = HannSinc::<Convolve<f64, $n>>::highpass(0.25);
                let coeffs = filter.config_ref().coefficients;
                let alt_sum: f64 = coeffs.iter().enumerate().fold(0.0, |acc, (k, &hk)| {
                    if k % 2 == 0 {
                        acc + hk
                    } else {
                        acc - hk
                    }
                });
                assert_abs_diff_eq!(alt_sum, 1.0, epsilon = 1e-10);
            }};
        }
        for &n in &[3_usize, 5, 7, 9, 11] {
            match n {
                3 => check_alt_sum_for_n!(3),
                5 => check_alt_sum_for_n!(5),
                7 => check_alt_sum_for_n!(7),
                9 => check_alt_sum_for_n!(9),
                11 => check_alt_sum_for_n!(11),
                _ => {}
            }
        }
    }

    #[test]
    #[should_panic(expected = "must be odd")]
    fn even_n_rejected() {
        let _ = HannSinc::<Convolve<f64, 8>>::highpass(FC);
    }

    #[test]
    #[should_panic(expected = "must be >= 3")]
    fn n1_panics() {
        let _ = HannSinc::<Convolve<f64, 1>>::highpass(FC);
    }

    #[test]
    #[should_panic(expected = "must be >= 3")]
    fn n_below_3_rejected() {
        let _ = HannSinc::<Convolve<f64, 2>>::highpass(FC);
    }

    #[test]
    fn nyquist_magnitude() {
        let f = HannSinc::<Convolve<f64, 33>>::highpass(0.25);
        let mag = gain_at_freq(&f.config_ref().coefficients, 0.5);
        assert!((mag - 1.0).abs() < 1e-10);
    }

    #[test]
    fn nyquist_gain_small_n() {
        macro_rules! check_nyquist_for_n {
            ($n:literal) => {{
                let filter = HannSinc::<Convolve<f64, $n>>::highpass(0.25);
                let coeffs = filter.config_ref().coefficients;
                let h_nyq = gain_at_freq(&coeffs, 0.5);
                assert_abs_diff_eq!(h_nyq, 1.0, epsilon = 1e-10);
            }};
        }
        for &n in &[3_usize, 7_usize] {
            match n {
                3 => check_nyquist_for_n!(3),
                7 => check_nyquist_for_n!(7),
                _ => {}
            }
        }
    }
}

// MARK: - Bandpass tests

#[cfg(any(feature = "libm", feature = "std"))]
mod bandpass {
    use approx::assert_abs_diff_eq;

    use super::super::*;
    use super::feed_sine;

    const F_LO: f64 = 0.1;
    const F_HI: f64 = 0.3;

    #[test]
    fn midband_gain() {
        let mut filter = HannSinc::<Convolve<f64, 33>>::bandpass(F_LO, F_HI);
        let f_c = (F_LO + F_HI) / 2.0;
        let outputs = feed_sine(&mut filter, f_c, 200);
        let steady = &outputs[33..];
        let mean_sq: f64 = steady.iter().map(|x| x * x).sum::<f64>() / steady.len() as f64;
        let rms = mean_sq.sqrt();
        let amplitude = rms * (2.0_f64).sqrt();
        assert!(
            (amplitude - 1.0).abs() < 0.01,
            "midband amplitude gain {} not ~1",
            amplitude
        );
    }

    #[test]
    fn passband_peak_gain() {
        let max_gain = {
            let mut max_g = 0.0_f64;
            for i in 0..=50 {
                let mut filter = HannSinc::<Convolve<f64, 33>>::bandpass(F_LO, F_HI);
                let f = F_LO + (F_HI - F_LO) * (i as f64) / 50.0;
                let outputs = feed_sine(&mut filter, f, 200);
                let steady = &outputs[33..];
                let mean_sq: f64 =
                    steady.iter().map(|x| x * x).sum::<f64>() / steady.len() as f64;
                let amplitude = mean_sq.sqrt() * (2.0_f64).sqrt();
                if amplitude > max_g {
                    max_g = amplitude;
                }
            }
            max_g
        };
        assert!(
            max_gain <= 1.0 + 0.015,
            "passband peak gain {} exceeds 1.0",
            max_gain
        );
    }

    #[test]
    fn signed_centre_gain_preserves_polarity() {
        const N: usize = 9;
        let f_lo: f64 = 0.1;
        let f_hi: f64 = 0.3;
        let f_c = (f_lo + f_hi) / 2.0;
        let neg_window = |_: usize, _: usize| -0.5_f64;

        let h_raw = sinc_bandpass::<f64, N>(f_lo, f_hi, neg_window, false);
        let h_norm = sinc_bandpass::<f64, N>(f_lo, f_hi, neg_window, true);

        let two_pi = 2.0 * core::f64::consts::PI;
        let m = (N as f64 - 1.0) / 2.0;

        let a_raw: f64 = h_raw.iter().enumerate().fold(0.0, |acc, (k, &hk)| {
            acc + hk * (two_pi * f_c * (k as f64 - m)).cos()
        });
        assert!(a_raw < 0.0, "expected negative raw A(f_c), got {}", a_raw);

        let a_norm: f64 = h_norm.iter().enumerate().fold(0.0, |acc, (k, &hk)| {
            acc + hk * (two_pi * f_c * (k as f64 - m)).cos()
        });
        assert_abs_diff_eq!(a_norm, 1.0, epsilon = 1e-10);
    }
}

// MARK: - Bandstop tests

#[cfg(any(feature = "libm", feature = "std"))]
mod bandstop {
    use approx::assert_abs_diff_eq;

    use super::super::*;
    use super::{feed_dc, feed_sine};

    const F_LO: f64 = 0.1;
    const F_HI: f64 = 0.3;

    #[test]
    fn dc_gain() {
        let mut filter = HannSinc::<Convolve<f64, 33>>::bandstop(F_LO, F_HI);
        let out = feed_dc(&mut filter);
        assert_abs_diff_eq!(out, 1.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "must be >= 3")]
    fn n_below_3_rejected() {
        let _ = HannSinc::<Convolve<f64, 1>>::bandstop(F_LO, F_HI);
    }

    #[test]
    #[should_panic(expected = "must be odd")]
    fn even_n_rejected() {
        let _ = HannSinc::<Convolve<f64, 8>>::bandstop(F_LO, F_HI);
    }

    #[test]
    fn notch_attenuation() {
        let mut filter = HannSinc::<Convolve<f64, 33>>::bandstop(F_LO, F_HI);
        let f_notch = (F_LO + F_HI) / 2.0;
        let outputs = feed_sine(&mut filter, f_notch, 200);
        let steady = &outputs[33..];
        let mean_sq: f64 = steady.iter().map(|x| x * x).sum::<f64>() / steady.len() as f64;
        let rms = mean_sq.sqrt();
        assert!(rms < 0.01, "notch RMS gain {} >= 0.01", rms);
    }
}

// MARK: - Existing tests (preserved from original)

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kernel_hand_derived_n5() {
    let h = sinc_lowpass::<f64, 5>(0.25, |_, _| 1.0, false);
    let expected = [0.0, 0.3183098861837907, 0.5, 0.3183098861837907, 0.0];
    for (a, b) in h.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kernel_normalised_dc_gain() {
    let h = sinc_lowpass::<f64, 9>(0.25, |_, _| 1.0, true);
    let sum: f64 = h.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
#[should_panic(expected = "floor")]
fn sinc_lowpass_rejects_zero_sum_window() {
    let _ = sinc_lowpass::<f64, 3>(0.25, |_, _| 0.0, true);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kernel_symmetry() {
    let h = sinc_lowpass::<f64, 9>(0.2, |_, _| 1.0, false);
    for k in 0..9 {
        assert_abs_diff_eq!(h[k], h[8 - k], epsilon = 1e-12);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kernel_symmetry_even_n() {
    let h = sinc_lowpass::<f64, 8>(0.2_f64, |_, _| 1.0_f64, false);
    for k in 0..8 {
        assert_abs_diff_eq!(h[k], h[7 - k], epsilon = 1e-12);
    }
    let h = sinc_lowpass::<f64, 8>(0.25_f64, |_, _| 1.0_f64, false);
    for k in 0..8 {
        assert_abs_diff_eq!(h[k], h[7 - k], epsilon = 1e-12);
    }
    let h = sinc_lowpass::<f64, 10>(0.2_f64, |_, _| 1.0_f64, false);
    for k in 0..10 {
        assert_abs_diff_eq!(h[k], h[9 - k], epsilon = 1e-12);
    }
    let h = sinc_lowpass::<f64, 10>(0.25_f64, |_, _| 1.0_f64, false);
    for k in 0..10 {
        assert_abs_diff_eq!(h[k], h[9 - k], epsilon = 1e-12);
    }
    let h = sinc_lowpass::<f64, 10>(0.2_f64, hann_window::<f64>, false);
    for k in 0..10 {
        assert_abs_diff_eq!(h[k], h[9 - k], epsilon = 1e-12);
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kernel_window_n1() {
    assert_abs_diff_eq!(hann_window::<f64>(0, 1), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(triangular_window::<f64>(0, 1), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(hamming_window::<f64>(0, 1), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(blackman_window::<f64>(0, 1), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(blackman_harris_window::<f64>(0, 1), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(flat_top_window::<f64>(0, 1), 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!((kaiser_window::<f64>(6.0))(0, 1), 1.0, epsilon = 1e-15);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn kaiser_window_n2_computes_naturally() {
    for beta in &[0.0_f64, 1.0, 5.0, 8.0, 12.0, 20.0] {
        let expected = 1.0 / bessel_i0(*beta);
        assert_abs_diff_eq!(
            (kaiser_window::<f64>(*beta))(0, 2),
            expected,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            (kaiser_window::<f64>(*beta))(1, 2),
            expected,
            epsilon = 1e-12
        );
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn hann_window_n2_is_zero() {
    assert!((hann_window::<f64>(0, 2) - 0.0).abs() < 1e-12);
    assert!((hann_window::<f64>(1, 2) - 0.0).abs() < 1e-12);
}

#[cfg(any(feature = "libm", feature = "std"))]
#[test]
fn numpy_hanning_n2_parity() {
    assert!((hann_window::<f64>(0, 2)).abs() < 1e-12);
    assert!((hann_window::<f64>(1, 2)).abs() < 1e-12);
}

// MARK: - Allpass complement (lowpass + highpass ≈ δ[M])

#[cfg(any(feature = "libm", feature = "std"))]
mod allpass_complement {
    use approx::assert_abs_diff_eq;

    use super::super::*;

    #[test]
    fn lowpass_plus_highpass_is_delta() {
        for &fc in &[0.1_f64, 0.2, 0.25, 0.3, 0.4] {
            for &n in &[3_usize, 5, 7, 9, 11] {
                macro_rules! check_arm {
                    ($ty:ident, $n:literal, $m:literal) => {{
                        let h_lp = $ty::<Convolve<f64, $n>>::lowpass_unnormalized(fc)
                            .config_ref()
                            .coefficients;
                        let h_hp = $ty::<Convolve<f64, $n>>::highpass_unnormalized(fc)
                            .config_ref()
                            .coefficients;
                        for k in 0..$n {
                            let expected = if k == $m { 1.0 } else { 0.0 };
                            assert_abs_diff_eq!(h_lp[k] + h_hp[k], expected, epsilon = 1e-12);
                        }
                    }};
                }

                macro_rules! check_complement {
                    ($ty:ident) => {
                        match n {
                            3 => check_arm!($ty, 3, 1),
                            5 => check_arm!($ty, 5, 2),
                            7 => check_arm!($ty, 7, 3),
                            9 => check_arm!($ty, 9, 4),
                            11 => check_arm!($ty, 11, 5),
                            _ => unreachable!(),
                        }
                    };
                }

                check_complement!(HannSinc);
                check_complement!(HammingSinc);
                check_complement!(BlackmanSinc);
            }
        }
    }
}

// MARK: - Unnormalized HP/BP/BS tests

#[cfg(any(feature = "libm", feature = "std"))]
mod unnormalized {
    use approx::assert_abs_diff_eq;

    use super::super::*;

    const FC: f64 = 0.25;
    const F_LO: f64 = 0.15;
    const F_HI: f64 = 0.35;

    #[test]
    fn unnormalized_highpass_differs_from_normalized() {
        let norm = HannSinc::<Convolve<f64, 9>>::highpass(FC);
        let raw = HannSinc::<Convolve<f64, 9>>::highpass_unnormalized(FC);
        let c_norm = norm.config_ref().coefficients;
        let c_raw = raw.config_ref().coefficients;
        let mut differ = false;
        for (n, r) in c_norm.iter().zip(c_raw.iter()) {
            if (n - r).abs() > 1e-12 {
                differ = true;
                break;
            }
        }
        assert!(
            differ,
            "unnormalized highpass should differ from normalized"
        );
    }

    #[test]
    fn unnormalized_bandpass_differs_from_normalized() {
        let norm = HannSinc::<Convolve<f64, 9>>::bandpass(F_LO, F_HI);
        let raw = HannSinc::<Convolve<f64, 9>>::bandpass_unnormalized(F_LO, F_HI);
        let c_norm = norm.config_ref().coefficients;
        let c_raw = raw.config_ref().coefficients;
        let mut differ = false;
        for (n, r) in c_norm.iter().zip(c_raw.iter()) {
            if (n - r).abs() > 1e-12 {
                differ = true;
                break;
            }
        }
        assert!(
            differ,
            "unnormalized bandpass should differ from normalized"
        );
    }

    #[test]
    fn unnormalized_bandstop_differs_from_normalized() {
        let norm = HannSinc::<Convolve<f64, 9>>::bandstop(F_LO, F_HI);
        let raw = HannSinc::<Convolve<f64, 9>>::bandstop_unnormalized(F_LO, F_HI);
        let c_norm = norm.config_ref().coefficients;
        let c_raw = raw.config_ref().coefficients;
        let mut differ = false;
        for (n, r) in c_norm.iter().zip(c_raw.iter()) {
            if (n - r).abs() > 1e-12 {
                differ = true;
                break;
            }
        }
        assert!(
            differ,
            "unnormalized bandstop should differ from normalized"
        );
    }

    #[test]
    fn unnormalized_highpass_raw_delta_minus_lowpass() {
        let h_lp = sinc_lowpass::<f64, 9>(FC, hann_window::<f64>, false);
        let h_hp = sinc_highpass::<f64, 9>(FC, hann_window::<f64>, false);
        let m = 4;
        for k in 0..9 {
            let expected = if k == m { 1.0 - h_lp[k] } else { -h_lp[k] };
            assert_abs_diff_eq!(h_hp[k], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn unnormalized_bandpass_raw_difference_of_lowpass() {
        let h_hi = sinc_lowpass::<f64, 9>(F_HI, hann_window::<f64>, false);
        let h_lo = sinc_lowpass::<f64, 9>(F_LO, hann_window::<f64>, false);
        let h_bp = sinc_bandpass::<f64, 9>(F_LO, F_HI, hann_window::<f64>, false);
        for k in 0..9 {
            assert_abs_diff_eq!(h_bp[k], h_hi[k] - h_lo[k], epsilon = 1e-12);
        }
    }

    #[test]
    fn unnormalized_bandstop_raw_delta_minus_bandpass() {
        let h_bp = sinc_bandpass::<f64, 9>(F_LO, F_HI, hann_window::<f64>, false);
        let h_bs = sinc_bandstop::<f64, 9>(F_LO, F_HI, hann_window::<f64>, false);
        let m = 4;
        for k in 0..9 {
            let expected = if k == m { 1.0 - h_bp[k] } else { -h_bp[k] };
            assert_abs_diff_eq!(h_bs[k], expected, epsilon = 1e-12);
        }
    }
}

// MARK: - Integration test: window + Convolve pipeline

#[cfg(any(feature = "libm", feature = "std"))]
mod integration_window_convolve {
    use approx::assert_abs_diff_eq;

    use crate::filters::fir::window::hann;

    use super::super::*;

    #[test]
    fn hann_window_plus_sinc_equals_hann_sinc() {
        const N: usize = 9;
        let fc: f64 = 0.25;

        let sinc_coeffs = sinc_lowpass::<f64, N>(fc, |_, _| 1.0, false);

        let hann_config = hann::Config::<f64, N>::new();
        let win = hann_config.weights;

        let mut manual = [0.0_f64; N];
        for k in 0..N {
            manual[k] = sinc_coeffs[k] * win[k];
        }

        let sum: f64 = manual.iter().sum();
        for coeff in &mut manual {
            *coeff /= sum;
        }

        let filter = HannSinc::<Convolve<f64, N>>::lowpass(fc);
        let expected = filter.config_ref().coefficients;

        for (a, b) in manual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }
}

// MARK: - KaiserSinc custom-β tests

#[cfg(any(feature = "libm", feature = "std"))]
mod kaiser_sinc {
    use approx::assert_abs_diff_eq;

    use super::super::*;

    const FC: f64 = 0.25;
    const F_LO: f64 = 0.1;
    const F_HI: f64 = 0.3;

    #[test]
    fn highpass_with_beta_matches_default() {
        let custom = KaiserSinc::<Convolve<f64, 9>>::highpass_with_beta(6.0, FC);
        let default = KaiserSinc::<Convolve<f64, 9>>::highpass(FC);
        let cc = custom.config_ref().coefficients;
        let cd = default.config_ref().coefficients;
        for (a, b) in cc.iter().zip(cd.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "Kaiser beta must be non-negative")]
    fn lowpass_with_beta_rejects_negative_beta() {
        let _ = KaiserSinc::<Convolve<f64, 33>>::lowpass_with_beta(-1.0, 0.25);
    }

    #[test]
    fn lowpass_with_beta_dc_gain() {
        let filter = KaiserSinc::<Convolve<f64, 33>>::lowpass_with_beta(6.0, 0.25);
        let sum: f64 = filter.config_ref().coefficients.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "frequency must be > 0")]
    fn lowpass_with_beta_hz_freq_zero_panics() {
        let _ = KaiserSinc::<Convolve<f64, 33>>::lowpass_with_beta_hz(6.0, 44_100.0, 0.0);
    }

    #[test]
    fn bandpass_with_beta_smoke() {
        let filter = KaiserSinc::<Convolve<f64, 33>>::bandpass_with_beta(6.0, F_LO, F_HI);
        let sum: f64 = filter.config_ref().coefficients.iter().sum();
        assert!(sum < 1e-6, "bandpass should have near-zero DC gain");
    }

    #[test]
    fn bandstop_with_beta_smoke() {
        let filter = KaiserSinc::<Convolve<f64, 33>>::bandstop_with_beta(6.0, F_LO, F_HI);
        let sum: f64 = filter.config_ref().coefficients.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }
}
