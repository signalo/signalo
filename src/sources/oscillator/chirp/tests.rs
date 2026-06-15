// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::*;
use approx::assert_abs_diff_eq;

#[test]
fn test_chirp_terminates_after_num_samples() {
    let config = Config {
        phase_increment_start: 0.1f32,
        phase_increment_end: 0.2f32,
        num_samples: 10,
    };
    let mut chirp = Chirp::with_config(config);

    let mut count = 0;
    while chirp.source().is_some() {
        count += 1;
    }

    assert_eq!(count, 10, "Chirp should output exactly num_samples values");
}

#[test]
fn test_chirp_returns_none_after_termination() {
    let config = Config {
        phase_increment_start: 0.1f32,
        phase_increment_end: 0.2f32,
        num_samples: 5,
    };
    let mut chirp = Chirp::with_config(config);

    for _ in 0..5 {
        let _ = chirp.source();
    }

    assert_eq!(chirp.source(), None, "Should return None after num_samples");
}

#[test]
fn test_chirp_phase_increases_monotonically() {
    let config = Config {
        phase_increment_start: 0.1f32,
        phase_increment_end: 0.2f32,
        num_samples: 100,
    };
    let mut chirp = Chirp::with_config(config);

    for _ in 0..100 {
        let phase_before = chirp.state.phase;
        let _ = chirp.source();
        let phase_after = chirp.state.phase;

        assert!(
            phase_after > phase_before,
            "Phase should increase monotonically"
        );
    }
}

#[test]
fn test_chirp_phase_increment_interpolation() {
    let config = Config {
        phase_increment_start: 0.1f32,
        phase_increment_end: 0.3f32,
        num_samples: 10,
    };
    let chirp = Chirp::with_config(config);

    // Delta = (0.3 - 0.1) / (10 - 1) = 0.022222...
    let delta = (0.3f32 - 0.1f32) / 9.0f32;
    assert_abs_diff_eq!(chirp.state.current_phase_increment, 0.1, epsilon = 0.0001);
    assert_abs_diff_eq!(chirp.state.phase_increment_delta, delta, epsilon = 0.0001);
}

#[test]
fn test_chirp_reset() {
    let config = Config {
        phase_increment_start: 0.1f32,
        phase_increment_end: 0.2f32,
        num_samples: 10,
    };
    let mut chirp = Chirp::with_config(config);

    for _ in 0..3 {
        let _ = chirp.source();
    }

    assert_ne!(
        chirp.state.sample_index, 0,
        "Sample index should have advanced"
    );
    assert_ne!(chirp.state.phase, 0.0, "Phase should have advanced");

    let chirp = chirp.reset();

    assert_eq!(
        chirp.state.sample_index, 0,
        "Reset should reset sample index to 0"
    );
    assert_abs_diff_eq!(chirp.state.phase, 0.0, epsilon = 1e-5);
}

#[test]
fn test_chirp_sine_output() {
    use alloc::vec::Vec;
    use core::f32::consts::PI;

    let config = Config {
        phase_increment_start: PI / 2.0,
        phase_increment_end: PI / 2.0,
        num_samples: 4,
    };
    let mut chirp = Chirp::with_config(config);

    let samples: Vec<f32> = (0..4).filter_map(|_| chirp.source()).collect();

    assert_eq!(samples.len(), 4);
    assert_abs_diff_eq!(samples[0], 0.0, epsilon = 0.01);
    assert_abs_diff_eq!(samples[1], 1.0, epsilon = 0.01);
    assert_abs_diff_eq!(samples[2], 0.0, epsilon = 0.01);
    assert_abs_diff_eq!(samples[3], -1.0, epsilon = 0.01);
}

#[test]
fn test_state_mut() {
    let config = Config {
        phase_increment_start: 0.5f32,
        phase_increment_end: 0.5f32,
        num_samples: 10,
    };
    let mut chirp = Chirp::with_config(config);

    let state = chirp.state_mut();
    state.phase = core::f32::consts::PI / 2.0;

    let result = chirp.source();
    assert!(result.is_some());
    // With phase=PI/2, sin should give ~1.0
    assert_abs_diff_eq!(result.unwrap(), 1.0, epsilon = 0.01);
}

#[test]
fn test_chirp_default_config() {
    let chirp = Chirp::<f32>::default();
    assert_abs_diff_eq!(chirp.config.phase_increment_start, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(chirp.config.phase_increment_end, 1.0, epsilon = 1e-5);
    assert_eq!(chirp.config.num_samples, 1000);
}

#[test]
fn test_chirp_state_reset_at_construction() {
    let config = Config {
        phase_increment_start: 0.1f32,
        phase_increment_end: 0.2f32,
        num_samples: 10,
    };
    let chirp = Chirp::with_config(config);

    assert_abs_diff_eq!(chirp.state.phase, 0.0, epsilon = 1e-5);
    assert_eq!(
        chirp.state.sample_index, 0,
        "Initial sample index should be 0"
    );
}

#[test]
fn test_chirp_f64() {
    use alloc::vec::Vec;
    use core::f64::consts::PI;
    // Verify Chirp works with f64 (not just f32)
    let config = Config {
        phase_increment_start: PI / 2.0,
        phase_increment_end: PI / 2.0,
        num_samples: 4,
    };
    let mut chirp = Chirp::<f64>::with_config(config);
    let samples: Vec<f64> = (0..4).filter_map(|_| chirp.source()).collect();
    assert_eq!(samples.len(), 4);
    assert_abs_diff_eq!(samples[0], 0.0, epsilon = 0.01);
    assert_abs_diff_eq!(samples[1], 1.0, epsilon = 0.01);
    assert_abs_diff_eq!(samples[2], 0.0, epsilon = 0.01);
    assert_abs_diff_eq!(samples[3], -1.0, epsilon = 0.01);
}
