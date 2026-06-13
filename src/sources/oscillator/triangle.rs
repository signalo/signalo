// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Triangle wave oscillator.
//!
//! This module provides a basic triangle wave oscillator using phase accumulation.
//! The oscillator outputs a linear ramp from -amplitude to +amplitude in the first
//! half period, then from +amplitude to -amplitude in the second half, repeating.
//!
//! # Limitations
//!
//! This is a basic (non-band-limited) oscillator. For applications sensitive to
//! aliasing, consider using a band-limited waveform generator or oversampling.

use num_traits::float::FloatCore;

use crate::traits::Source;

/// The triangle oscillator's configuration.
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
    /// Precomputed 0.5 for phase offset
    pub(crate) half: T,
    /// Precomputed 4.0 for triangle scaling
    pub(crate) four_times: T,
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
            half: T::from(0.5).expect("0.5 is representable"),
            #[allow(clippy::missing_panics_doc)]
            four_times: T::from(4.0).expect("4.0 is representable"),
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
            half: T::from(0.5).expect("0.5 is representable in any Float"),
            #[allow(clippy::missing_panics_doc)]
            four_times: T::from(4.0).expect("4.0 is representable in any Float"),
        }
    }
}

/// The triangle oscillator's state.
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

/// A triangle wave oscillator using phase accumulation.
///
/// Outputs a triangle waveform with peaks at period/4 and period*3/4.
/// Uses the formula: `output = amplitude * (1.0 - 4.0 * |phase - 0.5|)`
#[derive(Clone, Debug)]
pub struct TriangleOscillator<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> TriangleOscillator<T>
where
    T: FloatCore,
{
    /// Updates the phase for the next sample and returns the current output.
    #[inline]
    fn next_sample(&mut self) -> T {
        // Formula: output = amplitude * (1.0 - 4.0 * |phase - 0.5|)
        let phase_offset = self.state.phase - self.config.half;
        let abs_offset = phase_offset.abs();
        let output = self.config.amplitude * (T::one() - self.config.four_times * abs_offset);

        // Advance phase and wrap at 1.0 (while loop handles increment > 1.0)
        self.state.phase = self.state.phase + self.config.phase_increment;
        while self.state.phase >= T::one() {
            self.state.phase = self.state.phase - T::one();
        }

        output
    }
}

impl_oscillator_traits!(TriangleOscillator, T: FloatCore);

impl<T> Source for TriangleOscillator<T>
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
    fn test_triangle_peaks() {
        use alloc::vec::Vec;
        // phase_increment = 0.25 means 4 samples per period
        // Peaks should be at phase 0.25 and phase 0.75
        let config = Config {
            phase_increment: 0.25f32,
            amplitude: 1.0f32,
            half: 0.5f32,
            four_times: 4.0f32,
        };

        let mut oscillator = TriangleOscillator::with_config(config);

        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();

        // phase: [0, 0.25, 0.5, 0.75]
        // formula: 1.0 * (1.0 - 4.0 * |phase - 0.5|)
        // phase 0.0: |0.0 - 0.5| = 0.5 → 1.0 - 4*0.5 = 1 - 2 = -1
        // phase 0.25: |0.25 - 0.5| = 0.25 → 1.0 - 4*0.25 = 1 - 1 = 0 (zero crossing while rising)
        // phase 0.5: |0.5 - 0.5| = 0.0 → 1.0 - 4*0 = 1 (peak)
        // phase 0.75: |0.75 - 0.5| = 0.25 → 1.0 - 4*0.25 = 1 - 1 = 0 (zero crossing while falling)
        assert_abs_diff_eq!(samples[0], -1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[1], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[2], 1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[3], 0.0f32, epsilon = 1e-5);
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_triangle_period_8() {
        use alloc::vec::Vec;

        // phase_increment = 0.125 means 8 samples per period
        let config = Config {
            phase_increment: 0.125f32,
            amplitude: 2.0f32,
            half: 0.5f32,
            four_times: 4.0f32,
        };

        let mut oscillator = TriangleOscillator::with_config(config);

        let samples: Vec<_> = (0..8).map(|_| oscillator.source().unwrap()).collect();

        // Verify peak magnitudes at expected phase points
        // Peak at phase 0.5, zero at phase 0.0 and 1.0 (which wraps)
        // At phase 0.125: |0.125 - 0.5| = 0.375 → 2 * (1 - 4*0.375) = 2 * (-0.5) = -1.0
        // At phase 0.25: |0.25 - 0.5| = 0.25 → 2 * (1 - 4*0.25) = 2 * 0 = 0
        // At phase 0.375: |0.375 - 0.5| = 0.125 → 2 * (1 - 4*0.125) = 2 * 0.5 = 1.0
        // At phase 0.5: |0.5 - 0.5| = 0 → 2 * (1 - 0) = 2.0
        assert_abs_diff_eq!(samples[4], 2.0f32, epsilon = 1e-5); // phase 0.5
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_triangle_symmetry() {
        use alloc::vec::Vec;

        let config = Config {
            phase_increment: 0.1f32,
            amplitude: 1.0f32,
            half: 0.5f32,
            four_times: 4.0f32,
        };

        let mut oscillator = TriangleOscillator::with_config(config);

        let samples: Vec<_> = (0..10).map(|_| oscillator.source().unwrap()).collect();

        // Triangle wave should be symmetric around phase=0.5
        // sample[i] at phase 0.1*i
        // sample[5] is at phase 0.5 (peak)
        assert_abs_diff_eq!(samples[5], 1.0f32, epsilon = 1e-5); // peak

        // Verify all outputs are within [-amplitude, amplitude]
        for sample in &samples {
            assert!((*sample).abs() <= 1.0f32 + 1e-5);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_triangle_f64() {
        use alloc::vec::Vec;
        // Verify TriangleOscillator works with f64 (not just f32)
        let config = Config {
            phase_increment: 0.25f64,
            amplitude: 1.0f64,
            half: 0.5f64,
            four_times: 4.0f64,
        };
        let mut oscillator = TriangleOscillator::<f64>::with_config(config);
        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();
        assert_abs_diff_eq!(samples[0], -1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[1], 0.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[2], 1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[3], 0.0f64, epsilon = 1e-10);
    }
}
