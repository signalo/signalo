// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec;
use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn test_nan_propagation() {
    let config = Config::new([BiquadConfig {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    }]);
    let mut filter = BiquadCascadeArray::with_config(config);
    let result = filter.filter(f32::NAN);
    assert!(result.is_nan());
}

#[test]
fn test_identity_single_stage() {
    let config = Config::new([BiquadConfig {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    }]);

    let mut filter = BiquadCascadeArray::with_config(config);

    let input = [
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0,
    ];

    let output: Vec<_> = input.iter().map(|&x| filter.filter(x)).collect();

    assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
}

#[test]
fn test_reset() {
    let config = Config::new([BiquadConfig {
        b0: 1.0_f64,
        b1: 0.5,
        b2: 0.25,
        a1: -0.3,
        a2: 0.1,
    }]);

    let mut filter = BiquadCascadeArray::with_config(config.clone());

    // Accumulate non-zero state
    for _ in 0..50 {
        filter.filter(1.0);
    }

    // State must be non-zero before reset
    {
        let st = filter.state_mut();
        #[allow(clippy::float_cmp)]
        let state_is_nonzero = st.sections[0].s1 != 0.0 || st.sections[0].s2 != 0.0;
        assert!(state_is_nonzero);
    }

    let mut filter = filter.reset();

    // State must be zero after reset
    {
        let st = filter.state_mut();
        assert_eq!(st.sections[0].s1.to_bits(), 0.0_f64.to_bits());
        assert_eq!(st.sections[0].s2.to_bits(), 0.0_f64.to_bits());
    }

    // First sample after reset matches a fresh filter
    let mut fresh = BiquadCascadeArray::with_config(config);
    assert_abs_diff_eq!(filter.filter(1.0), fresh.filter(1.0), epsilon = 1e-10);
}

#[test]
fn test_n8_identity() {
    let sections: [BiquadConfig<f64>; 8] = core::array::from_fn(|_| BiquadConfig {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    });
    let config = Config::new(sections);
    let mut filter = BiquadCascadeArray::with_config(config);
    let result = filter.filter(42.0);
    assert_eq!(result, 42.0);
}

#[test]
fn test_integer_type() {
    let config = Config::new([BiquadConfig {
        b0: 1_i32,
        b1: 0,
        b2: 0,
        a1: 0,
        a2: 0,
    }]);
    let mut filter = BiquadCascadeArray::with_config(config);
    assert_eq!(filter.filter(7), 7);
}

#[test]
fn test_state_mut() {
    let config = Config::new([BiquadConfig {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.5,
        a2: 0.0,
    }]);

    let mut filter = BiquadCascadeArray::with_config(config);
    let _ = filter.filter(1.0);

    let state = filter.state_mut();
    let s1 = state.sections[0].s1;
    // output = b0*1.0 + 0 = 1.0; s1_new = b1*1.0 - a1*1.0 + 0 = 0 - 0.5*1.0 = -0.5
    assert_eq!(s1, -0.5);
}

#[test]
fn test_two_stages() {
    let config = Config::new([
        BiquadConfig {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        },
        BiquadConfig {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        },
    ]);

    let mut filter = BiquadCascadeArray::with_config(config);

    let input = [
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0,
    ];

    let output: Vec<_> = input.iter().map(|&x| filter.filter(x)).collect();

    assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
}

#[test]
fn test_two_stages_nonidentity() {
    use super::super::Biquad;
    use crate::traits::Filter;

    // Two non-trivial stages: compare cascade output against two sequential Biquad filters
    let cfg_a = BiquadConfig {
        b0: 0.5_f64,
        b1: 0.25,
        b2: 0.0,
        a1: -0.3,
        a2: 0.0,
    };
    let cfg_b = BiquadConfig {
        b0: 0.8_f64,
        b1: 0.0,
        b2: 0.1,
        a1: 0.2,
        a2: -0.05,
    };

    let mut cascade = BiquadCascadeArray::with_config(Config::new([cfg_a.clone(), cfg_b.clone()]));

    let mut biquad_a = Biquad::with_config(cfg_a);
    let mut biquad_b = Biquad::with_config(cfg_b);

    let input = [1.0, 0.0, -1.0, 0.5, 0.3, 0.0, 1.2, -0.7, 0.0, 0.1_f64];

    for &x in &input {
        let cascade_out = cascade.filter(x);
        let sequential_out = biquad_b.filter(biquad_a.filter(x));
        assert_abs_diff_eq!(cascade_out, sequential_out, epsilon = 1e-12);
    }
}

#[cfg(feature = "alloc")]
#[test]
fn biquad_cascade_vec_filters_without_panic() {
    use crate::traits::guts::FromGuts;

    let configs: Vec<BiquadConfig<f32>> = vec![BiquadConfig::default(); 2];
    let states: Vec<BiquadState<f32>> = vec![BiquadState::default(); 2];
    let config: Config<f32, Vec<BiquadConfig<f32>>> = Config::new(configs);
    let state: State<f32, Vec<BiquadState<f32>>> = State::new(states);
    let mut cascade: BiquadCascadeVec<f32> = BiquadCascade::from_guts((config, state));
    let out = cascade.filter(1.5);
    assert_eq!(out, 1.5);
}

#[test]
fn biquad_cascade_ref_mut_filters_without_panic() {
    use crate::traits::guts::FromGuts;

    let mut configs: [BiquadConfig<f32>; 2] = core::array::from_fn(|_| BiquadConfig::default());
    let mut states: [BiquadState<f32>; 2] = core::array::from_fn(|_| BiquadState::default());
    let config: Config<f32, &mut [BiquadConfig<f32>]> = Config::new(&mut configs[..]);
    let state: State<f32, &mut [BiquadState<f32>]> = State::new(&mut states[..]);
    let mut cascade = BiquadCascadeRefMut::from_guts((config, state));
    let out = cascade.filter(1.5);
    assert_eq!(out, 1.5);
}

#[cfg(feature = "complex")]
#[test]
fn real_coefficients_filter_complex_samples_like_independent_real_cascades() {
    use crate::complex::Complex32;

    let config = Config::new([
        BiquadConfig {
            b0: 0.5_f32,
            b1: 0.25,
            b2: 0.125,
            a1: -0.375,
            a2: 0.0625,
        },
        BiquadConfig {
            b0: 0.75_f32,
            b1: -0.125,
            b2: 0.03125,
            a1: 0.25,
            a2: -0.015625,
        },
    ]);
    let mut real_filter = BiquadCascadeArray::with_config(config.clone());
    let mut imag_filter = BiquadCascadeArray::with_config(config.clone());
    let mut complex_filter = BiquadCascadeArray::<Complex32, 2, f32>::with_config(config);

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
