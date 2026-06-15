// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use alloc::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn test_nan_propagation() {
    let config = Config {
        sections: [BiquadConfig {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }],
    };
    let mut filter = BiquadCascade::with_config(config);
    let result = filter.filter(f32::NAN);
    assert!(result.is_nan());
}

#[test]
fn test_identity_single_stage() {
    let config = Config {
        sections: [BiquadConfig {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }],
    };

    let mut filter = BiquadCascade::with_config(config);

    let input = [
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0,
    ];

    let output: Vec<_> = input.iter().map(|&x| filter.filter(x)).collect();

    assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
}

#[test]
fn test_reset() {
    let config = Config {
        sections: [BiquadConfig {
            b0: 1.0_f64,
            b1: 0.5,
            b2: 0.25,
            a1: -0.3,
            a2: 0.1,
        }],
    };

    let mut filter = BiquadCascade::with_config(config.clone());

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
    let mut fresh = BiquadCascade::with_config(config);
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
    let config = Config { sections };
    let mut filter = BiquadCascade::with_config(config);
    let result = filter.filter(42.0);
    assert_eq!(result, 42.0);
}

#[test]
fn test_integer_type() {
    let config = Config {
        sections: [BiquadConfig {
            b0: 1_i32,
            b1: 0,
            b2: 0,
            a1: 0,
            a2: 0,
        }],
    };
    let mut filter = BiquadCascade::with_config(config);
    assert_eq!(filter.filter(7), 7);
}

#[test]
fn test_state_mut() {
    let config = Config {
        sections: [BiquadConfig {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.5,
            a2: 0.0,
        }],
    };

    let mut filter = BiquadCascade::with_config(config);
    let _ = filter.filter(1.0);

    let state = filter.state_mut();
    let s1 = state.sections[0].s1;
    // output = b0*1.0 + 0 = 1.0; s1_new = b1*1.0 - a1*1.0 + 0 = 0 - 0.5*1.0 = -0.5
    assert_eq!(s1, -0.5);
}

#[test]
fn test_two_stages() {
    let config = Config {
        sections: [
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
        ],
    };

    let mut filter = BiquadCascade::with_config(config);

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

    let mut cascade = BiquadCascade::with_config(Config {
        sections: [cfg_a.clone(), cfg_b.clone()],
    });

    let mut biquad_a = Biquad::with_config(cfg_a);
    let mut biquad_b = Biquad::with_config(cfg_b);

    let input = [1.0, 0.0, -1.0, 0.5, 0.3, 0.0, 1.2, -0.7, 0.0, 0.1_f64];

    for &x in &input {
        let cascade_out = cascade.filter(x);
        let sequential_out = biquad_b.filter(biquad_a.filter(x));
        assert_abs_diff_eq!(cascade_out, sequential_out, epsilon = 1e-12);
    }
}
