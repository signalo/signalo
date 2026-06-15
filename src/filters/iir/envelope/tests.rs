// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::vec::Vec;

use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn test_envelope_rises_on_attack() {
    let filter = Envelope::with_config(Config {
        attack: 0.9f32,
        release: 0.1f32,
    });

    let inputs = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let outputs: Vec<_> = inputs
        .iter()
        .scan(filter, |filter, &input| Some(filter.filter(input)))
        .collect();

    // First two samples: envelope stays at 0
    assert_abs_diff_eq!(outputs[0], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(outputs[1], 0.0, epsilon = 1e-6);

    // At index 2: input jumps to 1.0, attack=0.9 so envelope = 0.9*1.0 + 0.1*0.0 = 0.9
    assert_abs_diff_eq!(outputs[2], 0.9, epsilon = 1e-6);

    // Index 3: input still 1.0, envelope = 0.9*1.0 + 0.1*0.9 = 0.99
    assert_abs_diff_eq!(outputs[3], 0.99, epsilon = 1e-6);

    // Index 4: input still 1.0, envelope = 0.9*1.0 + 0.1*0.99 = 0.999
    assert_abs_diff_eq!(outputs[4], 0.999, epsilon = 1e-6);

    // Index 5: input drops to 0.0, release=0.1
    // envelope = 0.1*0.0 + 0.9*0.999 = 0.8991
    assert_abs_diff_eq!(outputs[5], 0.8991, epsilon = 1e-4);

    // Index 6: envelope continues to fall: 0.1*0.0 + 0.9*0.8991 = 0.80919
    assert_abs_diff_eq!(outputs[6], 0.80919, epsilon = 1e-4);

    // Index 7: envelope = 0.1*0.0 + 0.9*0.80919 ≈ 0.728271
    assert_abs_diff_eq!(outputs[7], 0.728271, epsilon = 1e-4);

    // Verify that envelope at index 4 is > 0.9 (quick attack as expected)
    assert!(outputs[4] > 0.9);

    assert!(outputs[7] > 0.3); // Slow decay
}

#[test]
fn test_envelope_zero_input() {
    let filter = Envelope::with_config(Config {
        attack: 0.5f32,
        release: 0.5f32,
    });

    let inputs = [0.0, 0.0, 0.0, 0.0, 0.0];
    let outputs: Vec<_> = inputs
        .iter()
        .scan(filter, |filter, &input| Some(filter.filter(input)))
        .collect();

    // All outputs should be zero
    for output in outputs {
        assert_abs_diff_eq!(output, 0.0, epsilon = 1e-6);
    }
}

#[test]
fn test_envelope_config_ref() {
    let config = Config {
        attack: 0.8f32,
        release: 0.2f32,
    };
    let filter = Envelope::with_config(config);
    let config_ref = filter.config_ref();
    assert_abs_diff_eq!(config_ref.attack, 0.8, epsilon = 1e-6);
    assert_abs_diff_eq!(config_ref.release, 0.2, epsilon = 1e-6);
}

#[test]
fn test_envelope_config_clone() {
    let config = Config {
        attack: 0.7f32,
        release: 0.3f32,
    };
    let filter = Envelope::with_config(config.clone());
    let cloned_config = filter.config();
    assert_abs_diff_eq!(cloned_config.attack, 0.7, epsilon = 1e-6);
    assert_abs_diff_eq!(cloned_config.release, 0.3, epsilon = 1e-6);
}

#[test]
fn test_envelope_guts() {
    let config = Config {
        attack: 0.6f32,
        release: 0.4f32,
    };
    let mut filter = Envelope::with_config(config);

    // Process some samples
    filter.filter(0.5);
    filter.filter(1.0);
    let (config_out, state_out) = filter.into_guts();

    // Reconstruct
    let filter2 = Envelope::from_guts((config_out, state_out));
    let config_ref = filter2.config_ref();
    assert_abs_diff_eq!(config_ref.attack, 0.6, epsilon = 1e-6);
}

#[test]
fn test_envelope_reset() {
    let config = Config {
        attack: 0.9f32,
        release: 0.1f32,
    };
    let mut filter = Envelope::with_config(config);

    // Process samples
    filter.filter(1.0);
    filter.filter(1.0);
    let output_before = filter.filter(1.0);

    // Envelope should be high
    assert!(output_before > 0.8);

    // Reset
    let mut reset_filter = filter.reset();
    let output_after = reset_filter.filter(0.0);

    // After reset, output should be 0
    assert_abs_diff_eq!(output_after, 0.0, epsilon = 1e-6);
}

#[test]
fn test_envelope_negative_input() {
    let filter = Envelope::with_config(Config {
        attack: 0.9f32,
        release: 0.1f32,
    });

    let inputs = [0.0, -1.0, -1.0, 0.0];
    let outputs: Vec<_> = inputs
        .iter()
        .scan(filter, |filter, &input| Some(filter.filter(input)))
        .collect();

    // Envelope should track the absolute value
    assert_abs_diff_eq!(outputs[0], 0.0, epsilon = 1e-6);
    // -1.0 has abs = 1.0, so envelope = 0.9*1.0 + 0.1*0.0 = 0.9
    assert_abs_diff_eq!(outputs[1], 0.9, epsilon = 1e-6);
    // Still tracking 1.0
    assert_abs_diff_eq!(outputs[2], 0.99, epsilon = 1e-6);
}

#[test]
fn smoke() {
    // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
    let filter = Envelope::with_config(Config {
        attack: 1.0,
        release: 1.0,
    });
    let input = [
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0,
    ];
    let output: Vec<_> = input
        .iter()
        .scan(filter, |f, &x| Some(f.filter(x)))
        .collect();
    assert_abs_diff_eq!(output.as_slice(), input.as_slice(), epsilon = 1e-6);
}
