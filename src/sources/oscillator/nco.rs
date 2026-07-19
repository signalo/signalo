// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Fixed-point numerically controlled oscillator.
//!
//! The NCO stores phase as a wrapping `u32` full-turn phase word and frequency
//! as a signed `i32` phase step. This mirrors common SDR NCOs: the phase
//! accumulator wraps naturally, while the signed step represents positive and
//! negative rotation without requiring callers to manually encode two's
//! complement values.

use crate::math;
use crate::traits::Source;

use core::marker::PhantomData;
use num_traits::float::FloatCore;

#[allow(clippy::cast_precision_loss)]
const HALF_TURN_PHASE_WORD: f32 = 0x8000_0000_u32 as f32;

/// NCO configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config<T = f32> {
    /// Signed phase increment per output sample.
    pub(crate) phase_step: i32,
    pub(crate) scalar: PhantomData<T>,
}

impl<T> Default for Config<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T> Config<T> {
    /// Creates a config from a signed phase step.
    #[must_use]
    pub const fn new(phase_step: i32) -> Self {
        Self {
            phase_step,
            scalar: PhantomData,
        }
    }

    /// Creates a config from a signed phase step.
    #[must_use]
    pub const fn from_phase_step(phase_step: i32) -> Self {
        Self::new(phase_step)
    }

    /// Creates a config from frequency and sample rate in Hz.
    ///
    /// Frequencies outside the Nyquist interval of `sample_rate_hz` are folded
    /// modulo the sample rate and alias to the represented frequency.
    ///
    /// # Panics
    ///
    /// Panics if either input is not finite or `sample_rate_hz <= 0`.
    #[must_use]
    pub fn from_frequency(frequency_hz: f32, sample_rate_hz: f32) -> Self {
        Self {
            phase_step: Nco::<T>::phase_step_from_frequency(frequency_hz, sample_rate_hz),
            scalar: PhantomData,
        }
    }

    /// Returns the signed phase step.
    #[must_use]
    pub const fn phase_step(&self) -> i32 {
        self.phase_step
    }

    /// Returns the represented frequency in Hz for `sample_rate_hz`.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate_hz` is not finite or `sample_rate_hz <= 0`.
    #[must_use]
    pub fn frequency(&self, sample_rate_hz: f32) -> f32 {
        Nco::<T>::frequency_from_phase_step(self.phase_step, sample_rate_hz)
    }
}

/// NCO state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct State<T = f32> {
    /// Current wrapping full-turn phase word.
    pub(crate) phase: u32,
    pub(crate) scalar: PhantomData<T>,
}

impl<T> Default for State<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T> State<T> {
    /// Creates state from a full-turn phase word.
    #[must_use]
    pub const fn new(phase: u32) -> Self {
        Self {
            phase,
            scalar: PhantomData,
        }
    }

    /// Returns the current full-turn phase word.
    #[must_use]
    pub const fn phase(&self) -> u32 {
        self.phase
    }
}

/// A fixed-point numerically controlled oscillator.
///
/// The current phase is a `u32` full-turn phase word. The phase step is signed,
/// so positive values rotate forward and negative values rotate backward.
///
/// The scalar type `T` controls the output sample type only. Internal phase
/// accumulation and phase-to-sine/cosine conversion use the `u32`/`f32`
/// [`math::phase`] backend.
///
/// # Complexity
///
/// - **Time per sample:** O(1); one wrapping addition and one lookup/approximation for sin/cos.
/// - **Space:** O(1); stores one 32-bit phase word.
#[derive(Clone, Debug)]
pub struct Nco<T = f32> {
    config: Config<T>,
    state: State<T>,
}

impl<T> Nco<T> {
    /// Creates an NCO from an initial phase and signed phase step.
    #[must_use]
    pub const fn new(phase: u32, phase_step: i32) -> Self {
        Self {
            config: Config::new(phase_step),
            state: State::new(phase),
        }
    }

    /// Creates an NCO from a signed phase step and zero initial phase.
    #[must_use]
    pub const fn from_phase_step(phase_step: i32) -> Self {
        Self::new(0, phase_step)
    }

    /// Creates an NCO from frequency and sample rate in Hz.
    ///
    /// Frequencies outside the Nyquist interval of `sample_rate_hz` are folded
    /// modulo the sample rate and alias to the represented frequency.
    ///
    /// # Panics
    ///
    /// Panics if either input is not finite or `sample_rate_hz <= 0`.
    #[must_use]
    pub fn from_frequency(frequency_hz: f32, sample_rate_hz: f32) -> Self {
        Self::from_phase_step(Self::phase_step_from_frequency(
            frequency_hz,
            sample_rate_hz,
        ))
    }

    /// Returns the current full-turn phase word.
    #[must_use]
    pub const fn phase(&self) -> u32 {
        self.state.phase
    }

    /// Sets the current full-turn phase word.
    pub const fn set_phase(&mut self, phase: u32) {
        self.state.phase = phase;
    }

    /// Adjusts phase by a signed phase-word delta.
    pub fn adjust_phase(&mut self, delta_phase: i32) {
        self.state.phase = self.state.phase.wrapping_add(delta_phase.cast_unsigned());
    }

    /// Returns the signed phase step.
    #[must_use]
    pub const fn phase_step(&self) -> i32 {
        self.config.phase_step
    }

    /// Sets the signed phase step.
    pub const fn set_phase_step(&mut self, phase_step: i32) {
        self.config.phase_step = phase_step;
    }

    /// Adjusts the signed phase step.
    pub fn adjust_phase_step(&mut self, delta_phase_step: i32) {
        self.config.phase_step = self.config.phase_step.wrapping_add(delta_phase_step);
    }

    /// Sets the frequency from frequency and sample rate in Hz.
    ///
    /// Frequencies outside the Nyquist interval of `sample_rate_hz` are folded
    /// modulo the sample rate and alias to the represented frequency.
    ///
    /// # Panics
    ///
    /// Panics if either input is not finite or `sample_rate_hz <= 0`.
    pub fn set_frequency(&mut self, frequency_hz: f32, sample_rate_hz: f32) {
        self.config.phase_step = Self::phase_step_from_frequency(frequency_hz, sample_rate_hz);
    }

    /// Adjusts the frequency by `delta_frequency_hz`.
    ///
    /// Deltas outside the Nyquist interval of `sample_rate_hz` are folded
    /// modulo the sample rate and alias to the represented delta.
    ///
    /// # Panics
    ///
    /// Panics if either input is not finite or `sample_rate_hz <= 0`.
    pub fn adjust_frequency(&mut self, delta_frequency_hz: f32, sample_rate_hz: f32) {
        self.adjust_phase_step(Self::phase_step_from_frequency(
            delta_frequency_hz,
            sample_rate_hz,
        ));
    }

    /// Returns the represented frequency in Hz for `sample_rate_hz`.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate_hz` is not finite or `sample_rate_hz <= 0`.
    #[must_use]
    pub fn frequency(&self, sample_rate_hz: f32) -> f32 {
        Self::frequency_from_phase_step(self.config.phase_step, sample_rate_hz)
    }

    /// Advances the phase by one sample.
    pub fn step(&mut self) {
        self.state.phase = self
            .state
            .phase
            .wrapping_add(self.config.phase_step.cast_unsigned());
    }

    /// Advances the phase by `samples` signed sample periods.
    pub fn step_by(&mut self, samples: i32) {
        let delta = self.config.phase_step.wrapping_mul(samples);
        self.state.phase = self.state.phase.wrapping_add(delta.cast_unsigned());
    }

    /// Computes sine at the current phase.
    #[must_use]
    pub fn sin(&self) -> T
    where
        T: FloatCore + From<f32>,
    {
        From::from(math::phase::sin(self.state.phase))
    }

    /// Computes cosine at the current phase.
    #[must_use]
    pub fn cos(&self) -> T
    where
        T: FloatCore + From<f32>,
    {
        From::from(math::phase::cos(self.state.phase))
    }

    /// Computes sine and cosine at the current phase.
    ///
    /// The return order is `(sin, cos)`, matching Rust's `sin_cos` convention.
    #[must_use]
    pub fn sin_cos(&self) -> (T, T)
    where
        T: FloatCore + From<f32>,
    {
        let (sin, cos) = math::phase::sin_cos(self.state.phase);
        (From::from(sin), From::from(cos))
    }

    /// Computes the complex phasor `cos(phase) + j sin(phase)`.
    #[cfg(feature = "complex")]
    #[must_use]
    pub fn phasor(&self) -> crate::complex::Complex<T>
    where
        T: FloatCore + From<f32>,
    {
        let (sin, cos) = self.sin_cos();
        crate::complex::Complex::new(cos, sin)
    }
    /// Converts frequency and sample rate in Hz to a signed phase step.
    ///
    /// The returned step represents `frequency_hz / sample_rate_hz` turns per
    /// sample, folded into `[-1 / 2, 1 / 2)` turns per sample.
    /// Frequencies outside the Nyquist interval of `sample_rate_hz` therefore
    /// alias to the folded frequency.
    ///
    /// # Panics
    ///
    /// Panics if either input is not finite or `sample_rate_hz <= 0`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn phase_step_from_frequency(frequency_hz: f32, sample_rate_hz: f32) -> i32 {
        assert!(frequency_hz.is_finite(), "nco: frequency must be finite");
        assert!(
            sample_rate_hz.is_finite(),
            "nco: sample rate must be finite"
        );
        assert!(sample_rate_hz > 0.0, "nco: sample rate must be > 0");

        let turns_per_sample = frequency_hz / sample_rate_hz;
        let folded_turns_per_sample = turns_per_sample - (turns_per_sample + 0.5).floor();
        let step = (folded_turns_per_sample * 2.0 * HALF_TURN_PHASE_WORD).trunc();
        step as i32
    }

    /// Converts a signed phase step to frequency in Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate_hz` is not finite or `sample_rate_hz <= 0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frequency_from_phase_step(phase_step: i32, sample_rate_hz: f32) -> f32 {
        assert!(
            sample_rate_hz.is_finite(),
            "nco: sample rate must be finite"
        );
        assert!(sample_rate_hz > 0.0, "nco: sample rate must be > 0");

        phase_step as f32 * sample_rate_hz / (2.0 * HALF_TURN_PHASE_WORD)
    }
}

impl_oscillator_traits!(Nco, T: FloatCore + From<f32>);

impl<T> Source for Nco<T>
where
    T: FloatCore + From<f32>,
{
    type Output = (T, T);

    fn source(&mut self) -> Option<Self::Output> {
        let output = self.sin_cos();
        self.step();
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::traits::{Reset, Source, WithConfig};

    const EPS: f32 = 1.0e-6;

    #[test]
    fn source_outputs_sin_cos_then_steps() {
        // 0x4000_0000 advances by a quarter turn per sample.
        let mut nco = Nco::<f32>::from_phase_step(0x4000_0000);

        let (sin, cos) = nco.source().expect("NCO source is infinite");
        assert_abs_diff_eq!(sin, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(cos, 1.0, epsilon = EPS);

        let (sin, cos) = nco.source().expect("NCO source is infinite");
        assert_abs_diff_eq!(sin, 1.0, epsilon = EPS);
        assert_abs_diff_eq!(cos, 0.0, epsilon = EPS);

        let (sin, cos) = nco.source().expect("NCO source is infinite");
        assert_abs_diff_eq!(sin, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(cos, -1.0, epsilon = EPS);

        let (sin, cos) = nco.source().expect("NCO source is infinite");
        assert_abs_diff_eq!(sin, -1.0, epsilon = EPS);
        assert_abs_diff_eq!(cos, 0.0, epsilon = EPS);
    }

    #[test]
    fn negative_phase_step_rotates_backward() {
        let mut nco = Nco::<f32>::from_phase_step(-0x4000_0000);

        let (sin, cos) = nco.source().expect("NCO source is infinite");
        assert_abs_diff_eq!(sin, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(cos, 1.0, epsilon = EPS);

        let (sin, cos) = nco.source().expect("NCO source is infinite");
        assert_abs_diff_eq!(sin, -1.0, epsilon = EPS);
        assert_abs_diff_eq!(cos, 0.0, epsilon = EPS);
    }

    #[test]
    fn source_stays_balanced_over_many_exact_cycles() {
        const SAMPLES: usize = 4096;

        let mut nco = Nco::<f32>::from_phase_step(0x0100_0000);
        let mut sin_sum = 0.0_f64;
        let mut cos_sum = 0.0_f64;
        let mut power_sum = 0.0_f64;

        for _ in 0..SAMPLES {
            let (sin, cos) = nco.source().expect("NCO source is infinite");
            let sin = f64::from(sin);
            let cos = f64::from(cos);

            sin_sum += sin;
            cos_sum += cos;
            power_sum += sin.mul_add(sin, cos * cos);
        }

        assert_eq!(nco.phase(), 0);
        assert_abs_diff_eq!(sin_sum, 0.0, epsilon = 1.0e-4);
        assert_abs_diff_eq!(cos_sum, 0.0, epsilon = 1.0e-4);
        assert_abs_diff_eq!(power_sum / SAMPLES as f64, 1.0, epsilon = 1.0e-6);
    }

    #[test]
    fn frequency_helpers_convert_hz_and_phase_step() {
        let step = Nco::<f32>::phase_step_from_frequency(250.0, 1000.0);
        assert_eq!(step, 0x4000_0000);

        let negative_step = Nco::<f32>::phase_step_from_frequency(-250.0, 1000.0);
        assert_eq!(negative_step, -0x4000_0000);

        assert_abs_diff_eq!(
            Nco::<f32>::frequency_from_phase_step(step, 1000.0),
            250.0,
            epsilon = EPS
        );
    }

    #[test]
    fn config_from_frequency_sets_phase_step() {
        let config = Config::<f32>::from_frequency(125.0, 1000.0);

        assert_eq!(config.phase_step(), 0x2000_0000);
        assert_abs_diff_eq!(config.frequency(1000.0), 125.0, epsilon = EPS);
    }

    #[test]
    fn set_frequency_updates_phase_step() {
        let mut nco = Nco::<f32>::default();

        nco.set_frequency(-125.0, 1000.0);

        assert_eq!(nco.phase_step(), -0x2000_0000);
        assert_abs_diff_eq!(nco.frequency(1000.0), -125.0, epsilon = EPS);
    }

    #[test]
    fn step_by_matches_repeated_steps() {
        let mut repeated = Nco::<f32>::new(0x1234_5678, 0x0100_0000);
        let mut skipped = repeated.clone();

        for _ in 0..10 {
            repeated.step();
        }
        skipped.step_by(10);

        assert_eq!(skipped.phase(), repeated.phase());
    }

    #[test]
    fn adjust_phase_and_frequency_use_signed_deltas() {
        let mut nco = Nco::<f32>::new(0, 0);

        nco.adjust_phase(-1);
        nco.adjust_phase_step(-2);

        assert_eq!(nco.phase(), u32::MAX);
        assert_eq!(nco.phase_step(), -2);
    }

    #[test]
    fn reset_keeps_config_and_clears_phase() {
        let mut nco = Nco::with_config(Config::<f32>::new(42));
        nco.set_phase(123);

        let nco = nco.reset();

        assert_eq!(nco.phase_step(), 42);
        assert_eq!(nco.phase(), 0);
    }

    #[test]
    fn nyquist_frequency_folds_to_negative_nyquist() {
        assert_eq!(
            Nco::<f32>::phase_step_from_frequency(500.0, 1000.0),
            i32::MIN
        );
        assert_eq!(
            Nco::<f32>::phase_step_from_frequency(-500.0, 1000.0),
            i32::MIN
        );
    }

    #[test]
    fn frequency_aliases_fold_modulo_sample_rate() {
        assert_eq!(
            Nco::<f32>::phase_step_from_frequency(750.0, 1000.0),
            -0x4000_0000
        );
        assert_eq!(
            Nco::<f32>::phase_step_from_frequency(-750.0, 1000.0),
            0x4000_0000
        );
    }

    #[test]
    fn f64_source_outputs_f64_sin_cos() {
        let mut nco = Nco::<f64>::from_phase_step(0x4000_0000);

        let (sin, cos) = nco.source().expect("NCO source is infinite");

        assert_abs_diff_eq!(sin, 0.0_f64, epsilon = f64::from(EPS));
        assert_abs_diff_eq!(cos, 1.0_f64, epsilon = f64::from(EPS));
    }

    #[cfg(feature = "complex")]
    #[test]
    fn phasor_returns_cos_plus_j_sin() {
        let nco = Nco::<f32>::new(0x4000_0000, 0);
        let phasor = nco.phasor();

        assert_abs_diff_eq!(phasor.re, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(phasor.im, 1.0, epsilon = EPS);
    }
}
