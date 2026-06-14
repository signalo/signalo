// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sine and cosine oscillators.
//!
//! This module provides stable recursive oscillators that use the identity:
//! ```text
//! sin(θ + Δ) = sin(θ)·cos(Δ) + cos(θ)·sin(Δ)
//! cos(θ + Δ) = cos(θ)·cos(Δ) - sin(θ)·sin(Δ)
//! ```
//! for efficient phase advancement without accumulated drift.

use num_traits::float::FloatCore;

use crate::traits::Source;

/// The sine oscillator's configuration.
///
/// # Stability and Accuracy
///
/// The `sin_delta` and `cos_delta` values should be pre-computed using trigonometric
/// functions with the desired frequency and sample rate:
/// - `cos_delta = cos(2π * frequency / sample_rate)`
/// - `sin_delta = sin(2π * frequency / sample_rate)`
///
/// For `#![no_std]` environments, use a std-feature-gated preprocessing step
/// or provide pre-computed deltas from a lookup table.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Cosine of phase delta: `cos(2π * frequency / sample_rate)`
    pub(crate) cos_delta: T,
    /// Sine of phase delta: `sin(2π * frequency / sample_rate)`
    pub(crate) sin_delta: T,
}

#[cfg(feature = "std")]
impl<T> Config<T>
where
    T: num_traits::float::Float,
{
    /// Creates a new `Config` for the given frequency and sample rate.
    ///
    /// Pre-computes `cos(2π · freq / sample_rate)` and `sin(2π · freq / sample_rate)`.
    /// Requires the `std` feature (trigonometric functions).
    ///
    /// # Panics
    ///
    /// Panics if `T` cannot represent `2π`. This is infallible for standard `f32` and `f64` types.
    pub fn new(frequency: T, sample_rate: T) -> Self {
        let two_pi = T::from(core::f64::consts::TAU).expect("2π is representable");
        let delta = two_pi * frequency / sample_rate;
        Self {
            cos_delta: delta.cos(),
            sin_delta: delta.sin(),
        }
    }
}

impl<T> Default for Config<T>
where
    T: FloatCore,
{
    fn default() -> Self {
        Self {
            cos_delta: T::one(),
            sin_delta: T::zero(),
        }
    }
}

/// The sine oscillator's state.
///
/// Maintains the current sine and cosine values for phase accumulation.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Current sine value
    pub(crate) sin: T,
    /// Current cosine value
    pub(crate) cos: T,
    /// Renormalization counter (triggers renormalization every 256 samples)
    pub(crate) renorm_counter: u32,
}

impl<T> Default for State<T>
where
    T: FloatCore,
{
    fn default() -> Self {
        Self {
            sin: T::zero(),
            cos: T::one(),
            renorm_counter: 0,
        }
    }
}

/// A sine and cosine oscillator using recursive quadrature generation.
///
/// This oscillator uses stable quadrature phase tracking to generate sine waves
/// without accumulation of phase drift over long sequences.
#[derive(Clone, Debug)]
pub struct SineOscillator<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> SineOscillator<T>
where
    T: FloatCore,
{
    /// Updates the oscillator state by one sample period.
    #[inline]
    fn update(&mut self) {
        let sin_new =
            self.state.sin * self.config.cos_delta + self.state.cos * self.config.sin_delta;
        let cos_new =
            self.state.cos * self.config.cos_delta - self.state.sin * self.config.sin_delta;

        self.state.sin = sin_new;
        self.state.cos = cos_new;
        self.state.renorm_counter += 1;

        #[cfg(feature = "std")]
        self.renormalize();
    }

    /// Periodically renormalizes the quadrature pair to prevent amplitude drift.
    ///
    /// The recursive rotation matrix is not perfectly orthogonal in floating-point.
    /// Over long sequences, `sin² + cos²` drifts away from 1.0. This corrects the
    /// drift every 256 samples using a first-order Taylor approximation for
    /// `1/sqrt(x)` around x=1, avoiding the need for hardware `sqrt()`.
    #[cfg(feature = "std")]
    fn renormalize(&mut self) {
        if self.state.renorm_counter >= 256 {
            self.state.renorm_counter = 0;
            let mag_sq = self.state.sin * self.state.sin + self.state.cos * self.state.cos;
            if mag_sq != T::one() {
                let two = T::one() + T::one();
                // 1/sqrt(mag_sq) ≈ (3 - mag_sq) / 2 for mag_sq ≈ 1.0 (Taylor expansion)
                let inv_mag = (two + T::one() - mag_sq) / two;
                self.state.sin = self.state.sin * inv_mag;
                self.state.cos = self.state.cos * inv_mag;
            }
        }
    }
}

impl_oscillator_traits!(SineOscillator, T: FloatCore);

impl<T> Source for SineOscillator<T>
where
    T: FloatCore,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        let output = self.state.sin;
        self.update();
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::traits::WithConfig;

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_quarter_nyquist_period() {
        use alloc::vec::Vec;
        // At 1000 Hz sample rate, 250 Hz frequency:
        // Period = sample_rate / frequency = 1000 / 250 = 4 samples
        // Phase delta = 2π * 250 / 1000 = π/2
        // cos(π/2) = 0, sin(π/2) = 1

        let config = Config {
            cos_delta: 0.0f32,
            sin_delta: 1.0f32,
        };

        let mut oscillator = SineOscillator::with_config(config);

        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();

        // Expected: [0, 1, 0, -1] for sine at π/2 steps
        assert_abs_diff_eq!(samples[0], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[1], 1.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[2], 0.0f32, epsilon = 1e-5);
        assert_abs_diff_eq!(samples[3], -1.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_stability_long_sequence() {
        // Test that amplitude doesn't grow after many samples
        // Phase delta for 100 Hz at 1000 Hz sample rate = 2π/10 ≈ 0.628
        let phase_delta = 2.0 * core::f32::consts::PI / 10.0;
        let config = Config {
            cos_delta: phase_delta.cos(),
            sin_delta: phase_delta.sin(),
        };

        let mut oscillator = SineOscillator::with_config(config);

        // Verify bounded output over many samples
        for _ in 0..10000 {
            let sample = oscillator.source().unwrap();
            assert!(
                sample >= -1.1 && sample <= 1.1,
                "sine amplitude out of bounds: {sample}"
            );
        }
    }

    #[test]
    fn test_sine_f64() {
        use alloc::vec::Vec;
        // Verify SineOscillator works with f64 (not just f32)
        let config = Config {
            cos_delta: 0.0f64,
            sin_delta: 1.0f64,
        };
        let mut oscillator = SineOscillator::<f64>::with_config(config);
        let samples: Vec<_> = (0..4).map(|_| oscillator.source().unwrap()).collect();
        assert_abs_diff_eq!(samples[0], 0.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[1], 1.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[2], 0.0f64, epsilon = 1e-10);
        assert_abs_diff_eq!(samples[3], -1.0f64, epsilon = 1e-10);
    }
}
