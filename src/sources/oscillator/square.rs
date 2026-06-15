// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Square wave oscillator with configurable duty cycle.
//!
//! Generates square wave output with a fixed 50% duty cycle.
//! Amplitude and frequency are configurable.
//! This module provides a basic square wave oscillator using phase accumulation.
//! The oscillator outputs the configured amplitude for the first half of each period
//! and the negated amplitude for the second half.
//!
//! # Limitations
//!
//! This is a basic (non-band-limited) oscillator. For applications sensitive to
//! aliasing, consider using a band-limited waveform generator or oversampling.

use num_traits::float::FloatCore;

use crate::traits::Source;

/// The square oscillator's configuration.
///
/// # Phase and Frequency
///
/// The `phase_increment` should be computed as `frequency / sample_rate`.
/// This value will be added to the internal phase on each sample, and the phase
/// wraps at 1.0.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Phase increment per sample: `frequency / sample_rate`
    pub(crate) phase_increment: T,
    /// Amplitude (positive value for output level)
    pub(crate) amplitude: T,
    /// Threshold at phase 0.5 for the square wave transition
    pub(crate) half_threshold: T,
}

impl<T> Config<T>
where
    T: FloatCore,
{
    /// Creates a new `Config` for the given frequency, sample rate, and amplitude.
    pub fn new(frequency: T, sample_rate: T, amplitude: T) -> Self {
        Self {
            phase_increment: frequency / sample_rate,
            amplitude,
            #[allow(clippy::missing_panics_doc)]
            half_threshold: T::from(0.5).expect("0.5 is representable"),
        }
    }
}

impl<T> Default for Config<T>
where
    T: FloatCore,
{
    fn default() -> Self {
        Self {
            phase_increment: T::zero(),
            amplitude: T::one(),
            #[allow(clippy::missing_panics_doc)]
            half_threshold: T::from(0.5).expect("0.5 is representable in any Float"),
        }
    }
}

/// The square oscillator's state.
///
/// Maintains the current phase (0.0 to 1.0 per period).
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Current phase: 0.0 to 1.0 per period
    pub(crate) phase: T,
}

impl<T> Default for State<T>
where
    T: FloatCore,
{
    fn default() -> Self {
        Self { phase: T::zero() }
    }
}

/// A square wave oscillator using phase accumulation.
///
/// Outputs amplitude for the first half of each period (phase < 0.5) and
/// -amplitude for the second half.
#[derive(Clone, Debug)]
pub struct SquareOscillator<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> SquareOscillator<T>
where
    T: FloatCore,
{
    /// Updates the phase for the next sample and returns the current output.
    #[inline]
    fn next_sample(&mut self) -> T {
        let output = if self.state.phase < self.config.half_threshold {
            self.config.amplitude
        } else {
            -self.config.amplitude
        };

        // Advance phase and wrap at 1.0 (while loop handles increment > 1.0)
        self.state.phase = self.state.phase + self.config.phase_increment;
        while self.state.phase >= T::one() {
            self.state.phase = self.state.phase - T::one();
        }

        output
    }
}

impl_oscillator_traits!(SquareOscillator, T: FloatCore);

impl<T> Source for SquareOscillator<T>
where
    T: FloatCore,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        Some(self.next_sample())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use approx::assert_abs_diff_eq;

    #[allow(unused_imports)]
    use crate::traits::WithConfig;

    #[allow(unused_imports)]
    use super::*;

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_square_wave_alternates() {
        use alloc::vec::Vec;
        // phase_increment = 0.5 means 2 samples per period
        let config = Config {
            phase_increment: 0.5f32,
            amplitude: 1.0f32,
            half_threshold: 0.5f32,
        };

        let mut oscillator = SquareOscillator::with_config(config);

        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();

        // First half period: amplitude, second half: -amplitude
        // With increment 0.5: phase goes [0, 0.5, 0, 0.5, ...]
        // Output: [+1, -1, +1, -1]
        assert_abs_diff_eq!(samples[0], 1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[1], -1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[2], 1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[3], -1.0f32, epsilon = 1e-5);
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_square_wave_with_larger_period() {
        use alloc::vec::Vec;

        // phase_increment = 0.25 means 4 samples per period
        let config = Config {
            phase_increment: 0.25f32,
            amplitude: 2.0f32,
            half_threshold: 0.5f32,
        };

        let mut oscillator = SquareOscillator::with_config(config);

        let samples: Vec<_> = (0..8).map(|_| oscillator.source().unwrap()).collect();

        // phase: [0, 0.25, 0.5, 0.75, 0, 0.25, 0.5, 0.75]
        // output: [+2, +2, -2, -2, +2, +2, -2, -2]
        for i in 0..8 {
            let expected = if i % 4 < 2 { 2.0f32 } else { -2.0f32 };
            assert_abs_diff_eq!(samples[i], expected, epsilon = 1e-5);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_phase_wraps_correctly() {
        use alloc::vec::Vec;

        let config = Config {
            phase_increment: 0.3f32,
            amplitude: 1.0f32,
            half_threshold: 0.5f32,
        };

        let mut oscillator = SquareOscillator::with_config(config);

        // Collect 10 samples to verify wrap behavior
        let samples: Vec<_> = (0..10).map(|_| oscillator.source().unwrap()).collect();

        // All should be either +1 or -1 (no NaN or out-of-bounds values)
        for sample in samples {
            assert!(sample == 1.0f32 || sample == -1.0f32);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_square_f64() {
        use alloc::vec::Vec;
        // Verify SquareOscillator works with f64 (not just f32)
        let config = Config {
            phase_increment: 0.5f64,
            amplitude: 1.0f64,
            half_threshold: 0.5f64,
        };
        let mut oscillator = SquareOscillator::<f64>::with_config(config);
        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();
        assert_abs_diff_eq!(samples[0], 1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[1], -1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[2], 1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[3], -1.0f64, epsilon = 1e-10);
    }
}
