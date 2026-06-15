// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Pulse wave oscillator with configurable duty cycle.
//!
//! Generates pulse wave output with a configurable duty cycle, allowing asymmetric
//! waveforms where the positive portion of the period can be varied from 0% to 100%.
//! This module provides a pulse wave oscillator using phase accumulation.
//! The oscillator outputs the configured amplitude when the phase is within the
//! duty cycle and the negated amplitude otherwise.
//!
//! A duty cycle of 0.5 produces a symmetric square wave equivalent
//! to [`crate::sources::oscillator::square::SquareOscillator`].
//!
//! # Limitations
//!
//! This is a basic (non-band-limited) oscillator. For applications sensitive to
//! aliasing, consider using a band-limited waveform generator or oversampling.

use num_traits::float::FloatCore;

use crate::traits::Source;

/// The pulse oscillator's configuration.
///
/// # Phase and Frequency
///
/// The `phase_increment` should be computed as `frequency / sample_rate`.
/// This value will be added to the internal phase on each sample, and the phase
/// wraps at 1.0.
///
/// # Duty Cycle
///
/// The `duty_cycle` controls the fraction of each period where the output is at
/// positive amplitude. Values should be in the range `[0.0, 1.0]`:
/// - `0.0`: output is always at negative amplitude
/// - `0.5`: symmetric square wave (equivalent to 50% duty cycle)
/// - `1.0`: output is always at positive amplitude
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Phase increment per sample: `frequency / sample_rate`
    pub(crate) phase_increment: T,
    /// Amplitude (positive value for output level)
    pub(crate) amplitude: T,
    /// Duty cycle: fraction of period at positive amplitude (0.0 to 1.0)
    pub(crate) duty_cycle: T,
}

impl<T> Config<T>
where
    T: FloatCore,
{
    /// Creates a new `Config` for the given frequency, sample rate, amplitude, and duty cycle.
    ///
    /// `duty_cycle` should be in the range `[0.0, 1.0]`.
    pub fn new(frequency: T, sample_rate: T, amplitude: T, duty_cycle: T) -> Self {
        Self {
            phase_increment: frequency / sample_rate,
            amplitude,
            duty_cycle,
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
            duty_cycle: T::from(0.5).expect("0.5 is representable in any Float"),
        }
    }
}

/// The pulse oscillator's state.
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

/// A pulse wave oscillator using phase accumulation.
///
/// Outputs amplitude when `phase < duty_cycle` and `-amplitude` otherwise.
///
/// A duty cycle of 0.5 produces a symmetric square wave.
#[derive(Clone, Debug)]
pub struct PulseOscillator<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> PulseOscillator<T>
where
    T: FloatCore,
{
    #[inline]
    fn next_sample(&mut self) -> T {
        let output = if self.state.phase < self.config.duty_cycle {
            self.config.amplitude
        } else {
            -self.config.amplitude
        };

        self.state.phase = self.state.phase + self.config.phase_increment;
        while self.state.phase >= T::one() {
            self.state.phase = self.state.phase - T::one();
        }

        output
    }
}

impl_oscillator_traits!(PulseOscillator, T: FloatCore);

impl<T> Source for PulseOscillator<T>
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
    fn test_pulse_50_percent_duty_matches_square() {
        use alloc::vec::Vec;
        // 50% duty cycle at 2 samples per period
        let config = Config {
            phase_increment: 0.5f32,
            amplitude: 1.0f32,
            duty_cycle: 0.5f32,
        };

        let mut oscillator = PulseOscillator::with_config(config);

        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();

        // phase: [0, 0.5, 0, 0.5, ...]
        // With duty_cycle=0.5: phase<0.5 → +1, phase>=0.5 → -1
        assert_abs_diff_eq!(samples[0], 1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[1], -1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[2], 1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[3], -1.0f32, epsilon = 1e-5);
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_pulse_25_percent_duty_cycle() {
        use alloc::vec::Vec;
        // 25% duty cycle at 4 samples per period
        let config = Config {
            phase_increment: 0.25f32,
            amplitude: 1.0f32,
            duty_cycle: 0.25f32,
        };

        let mut oscillator = PulseOscillator::with_config(config);

        let samples: Vec<_> = (0..8).map(|_| oscillator.source().unwrap()).collect();

        // phase: [0, 0.25, 0.5, 0.75, 0, 0.25, 0.5, 0.75]
        // duty=0.25, high when phase < 0.25
        // output: [+1, -1, -1, -1, +1, -1, -1, -1]
        for i in 0..8 {
            let expected = if i % 4 == 0 { 1.0f32 } else { -1.0f32 };
            assert_abs_diff_eq!(samples[i], expected, epsilon = 1e-5);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_pulse_75_percent_duty_cycle() {
        use alloc::vec::Vec;
        // 75% duty cycle at 4 samples per period
        let config = Config {
            phase_increment: 0.25f32,
            amplitude: 2.0f32,
            duty_cycle: 0.75f32,
        };

        let mut oscillator = PulseOscillator::with_config(config);

        let samples: Vec<_> = (0..8).map(|_| oscillator.source().unwrap()).collect();

        // phase: [0, 0.25, 0.5, 0.75, 0, 0.25, 0.5, 0.75]
        // duty=0.75, high when phase < 0.75
        // output: [+2, +2, +2, -2, +2, +2, +2, -2]
        for i in 0..8 {
            let expected = if i % 4 < 3 { 2.0f32 } else { -2.0f32 };
            assert_abs_diff_eq!(samples[i], expected, epsilon = 1e-5);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_pulse_zero_duty_cycle() {
        use alloc::vec::Vec;
        // 0% duty cycle means always negative
        let config = Config {
            phase_increment: 0.25f32,
            amplitude: 1.0f32,
            duty_cycle: 0.0f32,
        };

        let mut oscillator = PulseOscillator::with_config(config);

        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();

        for sample in &samples {
            assert_abs_diff_eq!(*sample, -1.0f32, epsilon = 1e-5);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_pulse_full_duty_cycle() {
        use alloc::vec::Vec;
        // 100% duty cycle means always positive
        let config = Config {
            phase_increment: 0.25f32,
            amplitude: 1.0f32,
            duty_cycle: 1.0f32,
        };

        let mut oscillator = PulseOscillator::with_config(config);

        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();

        for sample in &samples {
            assert_abs_diff_eq!(*sample, 1.0f32, epsilon = 1e-5);
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_pulse_f64() {
        use alloc::vec::Vec;
        // Verify PulseOscillator works with f64 (not just f32)
        let config = Config {
            phase_increment: 0.5f64,
            amplitude: 1.0f64,
            duty_cycle: 0.5f64,
        };
        let mut oscillator = PulseOscillator::<f64>::with_config(config);
        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();
        assert_abs_diff_eq!(samples[0], 1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[1], -1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[2], 1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[3], -1.0f64, epsilon = 1e-10);
    }
}
