// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn test_identity_coefficients() {
    use alloc::vec::Vec;

    // Identity coefficients: b0=1, b1=b2=0, a1=a2=0
    // Expected: output equals input
    let filter = Biquad::with_config(Config {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    });

    let input = [
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0,
    ];

    let output: Vec<_> = input
        .iter()
        .scan(filter, |filter, &input| Some(filter.filter(input)))
        .collect();

    assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
}

#[test]
fn test_step_response_dc_gain() {
    // DC gain = (b0 + b1 + b2) / (1 + a1 + a2)
    // With b0=0.5, b1=0.25, b2=0.25, a1=0.0, a2=0.0 => DC gain = 1.0
    let mut filter = Biquad::with_config(Config {
        b0: 0.5_f64,
        b1: 0.25,
        b2: 0.25,
        a1: 0.0,
        a2: 0.0,
    });

    // Drive a step input long enough to reach steady state
    let mut output = 0.0;

    for _ in 0..1000 {
        output = filter.filter(1.0);
    }

    let expected_dc_gain = (0.5 + 0.25 + 0.25) / (1.0 + 0.0 + 0.0);

    assert_abs_diff_eq!(output, expected_dc_gain, epsilon = 1e-6);
}

#[test]
fn test_impulse_response_matches_hand_computation() {
    // Simple 1-pole IIR: b0=1, b1=0, b2=0, a1=-0.5, a2=0
    // Impulse response: y[0]=1, y[1]=0.5, y[2]=0.25, ...
    let mut filter = Biquad::with_config(Config {
        b0: 1.0_f64,
        b1: 0.0,
        b2: 0.0,
        a1: -0.5,
        a2: 0.0,
    });

    let y0 = filter.filter(1.0);
    let y1 = filter.filter(0.0);
    let y2 = filter.filter(0.0);
    let y3 = filter.filter(0.0);

    assert_abs_diff_eq!(y0, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(y1, 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(y2, 0.25, epsilon = 1e-10);
    assert_abs_diff_eq!(y3, 0.125, epsilon = 1e-10);
}

#[test]
fn test_reset_clears_state() {
    let config = Config {
        b0: 1.0_f64,
        b1: 0.5,
        b2: 0.25,
        a1: -0.3,
        a2: 0.1,
    };

    let mut filter = Biquad::with_config(config.clone());

    // Drive filter to accumulate non-zero state
    for _ in 0..50 {
        filter.filter(1.0);
    }

    // State should be non-zero now
    {
        let st = filter.state_mut();
        #[allow(clippy::float_cmp)]
        let state_is_nonzero = st.s1 != 0.0 || st.s2 != 0.0;
        assert!(state_is_nonzero);
    }

    let mut filter = filter.reset();

    {
        let st = filter.state_mut();
        assert_eq!(st.s1.to_bits(), 0.0_f64.to_bits());
        assert_eq!(st.s2.to_bits(), 0.0_f64.to_bits());
    }

    // First output after reset should match a fresh filter
    let mut fresh = Biquad::with_config(config);
    assert_abs_diff_eq!(filter.filter(1.0), fresh.filter(1.0), epsilon = 1e-10);
}

#[cfg(all(feature = "complex", any(feature = "libm", feature = "std")))]
#[test]
fn real_coefficients_filter_complex_step_like_independent_real_filters() {
    use crate::complex::Complex32;

    let config = Config::from(Butterworth::lowpass(48_000.0_f32, 4_000.0));
    let mut real_filter = Biquad::with_config(config.clone());
    let mut imag_filter = Biquad::with_config(config.clone());
    let mut complex_filter = Biquad::<Complex32, f32>::with_config(config);

    for _ in 0..64 {
        let real_output = real_filter.filter(1.0);
        let imag_output = imag_filter.filter(-0.5);
        let complex_output = complex_filter.filter(Complex32::new(1.0, -0.5));

        assert_eq!(complex_output.re.to_bits(), real_output.to_bits());
        assert_eq!(complex_output.im.to_bits(), imag_output.to_bits());
    }
}

#[cfg(all(feature = "complex", any(feature = "libm", feature = "std")))]
#[test]
fn real_coefficients_filter_complex_noise_like_independent_real_filters() {
    use crate::complex::Complex32;

    let config = Config::from(Butterworth::lowpass(48_000.0_f32, 4_000.0));
    let mut real_filter = Biquad::with_config(config.clone());
    let mut imag_filter = Biquad::with_config(config.clone());
    let mut complex_filter = Biquad::<Complex32, f32>::with_config(config);

    let real_input = [
        0.125_f32, -0.75, 0.5, 1.25, -1.5, 0.875, -0.25, 0.0625, 1.75, -0.9375,
    ];
    let imag_input = [
        -1.0_f32, 0.375, 1.125, -0.625, 0.25, -1.75, 0.8125, 0.5, -0.125, 1.5,
    ];

    for (&real, &imag) in real_input.iter().zip(&imag_input) {
        let real_output = real_filter.filter(real);
        let imag_output = imag_filter.filter(imag);
        let complex_output = complex_filter.filter(Complex32::new(real, imag));

        assert_eq!(complex_output.re.to_bits(), real_output.to_bits());
        assert_eq!(complex_output.im.to_bits(), imag_output.to_bits());
    }
}
