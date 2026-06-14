// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Chirp (frequency sweep) oscillators.
//!
//! This module provides a finite-duration source that sweeps frequency linearly from
//! a starting frequency to an ending frequency over a specified number of samples.
//! The phase is advanced each sample using phase increment values that are linearly
//! interpolated between start and end increments.

use num_traits::{Float, One, Zero};

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Reset, Source, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The chirp oscillator's configuration.
///
/// # Configuration
///
/// The `phase_increment_start` and `phase_increment_end` values should be pre-computed
/// using the desired start and end frequencies and sample rate:
/// - `phase_increment_start = 2π * frequency_start / sample_rate`
/// - `phase_increment_end = 2π * frequency_end / sample_rate`
///
/// For `#![no_std]` environments, use a std-feature-gated preprocessing step
/// or provide pre-computed increments from a lookup table.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Phase increment at the start of the chirp
    pub(crate) phase_increment_start: T,
    /// Phase increment at the end of the chirp
    pub(crate) phase_increment_end: T,
    pub(crate) num_samples: usize,
}

impl<T> Config<T>
where
    T: Float,
{
    /// Creates a new `Config` for a linear frequency sweep.
    ///
    /// `frequency_start` and `frequency_end` define the sweep range, `sample_rate`
    /// is the sample rate in Hz, and `num_samples` is the total number of samples
    /// in the chirp.
    ///
    /// # Panics
    ///
    /// Panics if `T` cannot represent `2π`. This is infallible for standard `f32` and `f64` types.
    pub fn new(frequency_start: T, frequency_end: T, sample_rate: T, num_samples: usize) -> Self {
        let two_pi = T::from(core::f64::consts::TAU).expect("2π is representable");
        Self {
            phase_increment_start: two_pi * frequency_start / sample_rate,
            phase_increment_end: two_pi * frequency_end / sample_rate,
            num_samples,
        }
    }
}

impl<T> Default for Config<T>
where
    T: Zero + One,
{
    fn default() -> Self {
        Self {
            phase_increment_start: T::zero(),
            phase_increment_end: T::one(),
            num_samples: 1000,
        }
    }
}

/// The chirp oscillator's state.
///
/// Maintains the current phase, the sample index, and incremental phase tracking state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Current phase value
    pub(crate) phase: T,
    /// Current sample index (increments from 0 to num_samples-1)
    pub(crate) sample_index: usize,
    /// Current phase increment (starts at `phase_increment_start`)
    pub(crate) current_phase_increment: T,
    /// Phase increment delta per sample: (end - start) / (`num_samples` - 1)
    pub(crate) phase_increment_delta: T,
}

impl<T> Default for State<T>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            phase: T::zero(),
            sample_index: 0,
            current_phase_increment: T::zero(),
            phase_increment_delta: T::zero(),
        }
    }
}

/// A chirp (linear frequency sweep) oscillator.
///
/// This source generates a sine wave with linearly increasing or decreasing frequency
/// over a finite duration. The frequency changes monotonically from `f_start` to `f_end`
/// as represented by phase increments.
///
/// After `num_samples` samples, the source returns `None`.
#[derive(Clone, Debug)]
pub struct Chirp<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> ConfigTrait for Chirp<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for Chirp<T> {
    type State = State<T>;
}

impl<T> WithConfig for Chirp<T>
where
    T: Float,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let delta = if config.num_samples <= 1 {
            T::zero()
        } else {
            let num_samples_t = T::from(config.num_samples).unwrap_or(T::one());
            (config.phase_increment_end - config.phase_increment_start) / (num_samples_t - T::one())
        };
        Self {
            state: State {
                phase: T::zero(),
                sample_index: 0,
                current_phase_increment: config.phase_increment_start,
                phase_increment_delta: delta,
            },
            config,
        }
    }
}

impl<T> Default for Chirp<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T> ConfigRef for Chirp<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for Chirp<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for Chirp<T> {
    /// # Safety
    ///
    /// The caller is responsible for upholding the oscillator's internal invariants:
    /// - `state.sample_index` should not exceed `config.num_samples`
    /// - `state.phase_increment_delta` and `state.current_phase_increment` must be
    ///   consistent with the configuration
    /// - Modifying state fields arbitrarily may produce nonsensical output values
    #[doc(hidden)]
    fn state_mut(&mut self) -> &mut Self::State {
        // SAFETY: `&mut self` guarantees exclusive access; no other references
        // to state exist within the program at this point.
        &mut self.state
    }
}

impl<T> HasGuts for Chirp<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for Chirp<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for Chirp<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for Chirp<T>
where
    T: Float,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for Chirp<T> where Self: Reset {}

impl<T> Chirp<T>
where
    T: Float,
{
    /// Updates the oscillator state by one sample period.
    #[inline]
    fn update(&mut self) {
        self.state.phase = self.state.phase + self.state.current_phase_increment;
        self.state.current_phase_increment =
            self.state.current_phase_increment + self.state.phase_increment_delta;
    }
}

impl<T> Source for Chirp<T>
where
    T: Float,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        if self.state.sample_index >= self.config.num_samples {
            return None;
        }

        let output = self.state.phase.sin();
        self.update();
        self.state.sample_index += 1;

        Some(output)
    }
}

#[cfg(test)]
mod tests {
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
        while let Some(_) = chirp.source() {
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
            let phase_before = chirp.state.phase.clone();
            let _ = chirp.source();
            let phase_after = chirp.state.phase.clone();

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
}
