// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Chirp (frequency sweep) source for frequency analysis.
//!
//! Generates a finite-duration source that sweeps frequency linearly from a start
//! frequency to an end frequency over a specified number of samples.
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
///
/// # Complexity
///
/// - **Time per sample:** O(1); one sine evaluation, two additions, and one bounds check.
/// - **Space:** O(1); stores current phase, sample index, and phase increment.
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
mod tests;
