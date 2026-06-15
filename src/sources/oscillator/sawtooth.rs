// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sawtooth wave oscillator with linear ramps.
//!
//! Generates sawtooth wave output that rises linearly from negative to positive peaks
//! over each period before resetting.
//! This module provides a basic sawtooth wave oscillator using phase accumulation.
//! The oscillator outputs a linear ramp from -amplitude to +amplitude per period.
//!
//! # Limitations
//!
//! This is a basic (non-band-limited) oscillator. For applications sensitive to
//! aliasing, consider using a band-limited waveform generator or oversampling.

use num_traits::float::FloatCore;

use crate::traits::Source;

/// The sawtooth oscillator's configuration.
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
    /// Amplitude (positive value for peak level)
    pub(crate) amplitude: T,
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
        }
    }
}

/// The sawtooth oscillator's state.
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

/// A sawtooth wave oscillator using phase accumulation.
///
/// Outputs a linearly rising ramp from -amplitude to +amplitude per period.
/// Uses the formula: `output = amplitude * (2.0 * phase - 1.0)`
#[derive(Clone, Debug)]
pub struct SawtoothOscillator<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> SawtoothOscillator<T>
where
    T: FloatCore,
{
    /// Updates the phase for the next sample and returns the current output.
    #[inline]
    fn next_sample(&mut self) -> T {
        // Formula: output = amplitude * (2.0 * phase - 1.0) = amplitude * (phase + phase - 1.0)
        // This produces a ramp from -amplitude to +amplitude per period
        let output = self.config.amplitude * (self.state.phase + self.state.phase - T::one());

        // Advance phase and wrap at 1.0 (while loop handles increment > 1.0)
        self.state.phase = self.state.phase + self.config.phase_increment;
        while self.state.phase >= T::one() {
            self.state.phase = self.state.phase - T::one();
        }

        output
    }
}

impl_oscillator_traits!(SawtoothOscillator, T: FloatCore);

impl<T> Source for SawtoothOscillator<T>
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
    fn test_sawtooth_ramps() {
        use alloc::vec::Vec;
        // phase_increment = 0.25 means 4 samples per period
        let config = Config {
            phase_increment: 0.25f32,
            amplitude: 1.0f32,
        };

        let mut oscillator = SawtoothOscillator::with_config(config);

        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();

        // phase: [0, 0.25, 0.5, 0.75]
        // formula: 1.0 * (2.0 * phase - 1.0)
        // phase 0.0: 1.0 * (0 - 1.0) = -1.0
        // phase 0.25: 1.0 * (0.5 - 1.0) = -0.5
        // phase 0.5: 1.0 * (1.0 - 1.0) = 0.0
        // phase 0.75: 1.0 * (1.5 - 1.0) = 0.5
        assert_abs_diff_eq!(samples[0], -1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[1], -0.5f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[2], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[3], 0.5f32, epsilon = 1e-5);
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_sawtooth_period_8() {
        use alloc::vec::Vec;

        // phase_increment = 0.125 means 8 samples per period
        let config = Config {
            phase_increment: 0.125f32,
            amplitude: 2.0f32,
        };

        let mut oscillator = SawtoothOscillator::with_config(config);

        let samples: Vec<_> = (0..8).map(|_| oscillator.source().unwrap()).collect();

        // Verify first and last samples (just before wrap)
        // First: phase 0.0 → 2.0 * (0 - 1.0) = -2.0
        // Last (index 7): phase 0.875 → 2.0 * (1.75 - 1.0) = 1.5
        assert_abs_diff_eq!(samples[0], -2.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[7], 1.5f32, epsilon = 1e-5);

        // Verify monotonic increase
        for i in 1..8 {
            assert!(samples[i] > samples[i - 1]);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_sawtooth_linear_ramp() {
        use alloc::vec::Vec;

        let config = Config {
            phase_increment: 0.1f32,
            amplitude: 1.0f32,
        };

        let mut oscillator = SawtoothOscillator::with_config(config);

        let samples: Vec<_> = (0..10).map(|_| oscillator.source().unwrap()).collect();

        // Verify output spans from -amplitude to near +amplitude
        assert!(samples[0] < -0.9f32);
        assert!(samples[9] < 1.0f32);
        assert!(samples[9] > 0.7f32);

        // Verify linear progression with equal step size (approximately)
        let mut steps: Vec<_> = Vec::new();
        for i in 1..samples.len() {
            steps.push(samples[i] - samples[i - 1]);
        }
        let first_step = steps[0];
        for step in &steps {
            assert_abs_diff_eq!(*step, first_step, epsilon = 1e-5);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_sawtooth_f64() {
        use alloc::vec::Vec;
        // Verify SawtoothOscillator works with f64 (not just f32)
        let config = Config {
            phase_increment: 0.25f64,
            amplitude: 1.0f64,
        };
        let mut oscillator = SawtoothOscillator::<f64>::with_config(config);
        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();
        assert_abs_diff_eq!(samples[0], -1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[1], -0.5f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[2], 0.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[3], 0.5f64, epsilon = 1e-10);
    }
}
