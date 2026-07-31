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
//!
//! Both are [`crate::math::phase`]'s representation, and every conversion into
//! and out of them delegates there.

use crate::math;
use crate::traits::Source;

use core::marker::PhantomData;
use num_traits::float::FloatCore;

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

    /// Creates a config from a phase step in turns per sample.
    ///
    /// # Panics
    ///
    /// Panics if `turns_per_sample` is not finite.
    #[must_use]
    pub fn from_turns_per_sample(turns_per_sample: f32) -> Self {
        Self {
            phase_step: Nco::<T>::phase_step_from_turns_per_sample(turns_per_sample),
            scalar: PhantomData,
        }
    }

    /// Creates a config from a phase step in radians per sample.
    ///
    /// # Panics
    ///
    /// Panics if `radians_per_sample` is not finite.
    #[must_use]
    pub fn from_radians_per_sample(radians_per_sample: f32) -> Self {
        Self {
            phase_step: Nco::<T>::phase_step_from_radians_per_sample(radians_per_sample),
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

    /// Sets the phase step from turns per sample.
    ///
    /// # Panics
    ///
    /// Panics if `turns_per_sample` is not finite.
    pub fn set_turns_per_sample(&mut self, turns_per_sample: f32) {
        self.config.phase_step = Self::phase_step_from_turns_per_sample(turns_per_sample);
    }

    /// Sets the phase step from radians per sample.
    ///
    /// # Panics
    ///
    /// Panics if `radians_per_sample` is not finite.
    pub fn set_radians_per_sample(&mut self, radians_per_sample: f32) {
        self.config.phase_step = Self::phase_step_from_radians_per_sample(radians_per_sample);
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

    /// Creates an NCO from a phase step in turns per sample and zero initial phase.
    ///
    /// Turns per sample is the same quantity as a normalized frequency in cycles per sample, so a
    /// carrier offset measured that way needs no conversion.
    ///
    /// # Panics
    ///
    /// Panics if `turns_per_sample` is not finite.
    #[must_use]
    pub fn from_turns_per_sample(turns_per_sample: f32) -> Self {
        Self::from_phase_step(Self::phase_step_from_turns_per_sample(turns_per_sample))
    }

    /// Creates an NCO from an initial phase and a phase step, both in turns.
    ///
    /// Turns are this oscillator's native unit: one turn is the full range of the phase word, and a
    /// normalized frequency in cycles per sample is already turns per sample. Steps outside the
    /// Nyquist interval of half a turn per sample fold and alias, as with [`Self::from_frequency`].
    ///
    /// # Panics
    ///
    /// Panics if either input is not finite.
    #[must_use]
    pub fn from_turns(phase_turns: f32, turns_per_sample: f32) -> Self {
        Self::new(
            Self::phase_word_from_turns(phase_turns),
            Self::phase_step_from_turns_per_sample(turns_per_sample),
        )
    }

    /// Returns the phase step in turns per sample, folded into `[-1 / 2, 1 / 2)`.
    #[must_use]
    pub fn turns_per_sample(&self) -> f32 {
        Self::turns_per_sample_from_phase_step(self.config.phase_step)
    }

    /// Converts a phase in turns to a wrapping full-turn phase word.
    ///
    /// # Panics
    ///
    /// Panics if `turns` is not finite.
    #[must_use]
    pub fn phase_word_from_turns(turns: f32) -> u32 {
        math::phase::phase_word_from_turns(turns)
    }

    /// Converts turns per sample to a signed phase step.
    ///
    /// This is where the conversion actually happens. Turns per sample is the phase word's own unit,
    /// one turn being its full range, so every other spelling of a frequency reduces to this one:
    /// [`Self::phase_step_from_frequency`] divides by the sample rate first and
    /// [`Self::phase_step_from_radians_per_sample`] divides by `2π`. Rates outside the Nyquist interval of
    /// half a turn per sample fold and alias.
    ///
    /// # Panics
    ///
    /// Panics if `turns_per_sample` is not finite.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn phase_step_from_turns_per_sample(turns_per_sample: f32) -> i32 {
        math::phase::phase_step_from_turns_per_sample(turns_per_sample)
    }

    /// Converts a signed phase step to turns per sample.
    ///
    /// The inverse counterpart, and likewise the one place the scale is applied.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn turns_per_sample_from_phase_step(phase_step: i32) -> f32 {
        math::phase::turns_per_sample_from_phase_step(phase_step)
    }

    /// Creates an NCO from an initial phase and a phase step, both in radians.
    ///
    /// Steps outside the Nyquist interval of `π` radians per sample fold and alias, as with
    /// [`Self::from_frequency`].
    ///
    /// # Panics
    ///
    /// Panics if either input is not finite.
    #[must_use]
    pub fn from_radians(phase_radians: f32, radians_per_sample: f32) -> Self {
        Self::new(
            Self::phase_word_from_radians(phase_radians),
            Self::phase_step_from_radians_per_sample(radians_per_sample),
        )
    }

    /// Returns the phase step in radians per sample, folded into `[-π, π)`.
    #[must_use]
    pub fn radians_per_sample(&self) -> f32 {
        Self::radians_per_sample_from_phase_step(self.config.phase_step)
    }

    /// Returns the current phase in turns, in `[0, 1)`.
    #[must_use]
    pub fn phase_turns(&self) -> f32 {
        Self::turns_from_phase_word(self.state.phase)
    }

    /// Returns the current phase in radians, in `[0, 2π)`.
    #[must_use]
    pub fn phase_radians(&self) -> f32 {
        Self::radians_from_phase_word(self.state.phase)
    }

    /// Converts a phase in radians to a wrapping full-turn phase word.
    ///
    /// One turn is `2π` radians, so any finite input folds into the single turn the phase word
    /// represents.
    ///
    /// # Panics
    ///
    /// Panics if `radians` is not finite.
    #[must_use]
    pub fn phase_word_from_radians(radians: f32) -> u32 {
        math::phase::phase_word_from_radians(radians)
    }

    /// Converts radians per sample to a signed phase step.
    ///
    /// Rates outside the Nyquist interval of `π` radians per sample fold modulo `2π` and
    /// alias, as with [`Self::phase_step_from_frequency`].
    ///
    /// # Panics
    ///
    /// Panics if `radians_per_sample` is not finite.
    #[must_use]
    pub fn phase_step_from_radians_per_sample(radians_per_sample: f32) -> i32 {
        math::phase::phase_step_from_radians_per_sample(radians_per_sample)
    }

    /// Converts a signed phase step to radians per sample.
    #[must_use]
    pub fn radians_per_sample_from_phase_step(phase_step: i32) -> f32 {
        math::phase::radians_per_sample_from_phase_step(phase_step)
    }

    /// Converts a wrapping full-turn phase word to radians in `[0, 2π)`.
    #[must_use]
    pub fn radians_from_phase_word(phase: u32) -> f32 {
        math::phase::radians_from_phase_word(phase)
    }

    /// Converts a wrapping full-turn phase word to turns in `[0, 1)`.
    ///
    /// The unsigned counterpart of [`Self::turns_per_sample_from_phase_step`], and the only other
    /// place the phase-word scale is applied.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn turns_from_phase_word(phase: u32) -> f32 {
        math::phase::turns_from_phase_word(phase)
    }

    /// Returns the phasor at the current phase, then advances one sample.
    ///
    /// The phasor is the one [`Self::phasor`] would return, not the one after the step. This is
    /// [`Self::phasor`] followed by [`Self::step`], in the order [`Source::source`] uses. Prefer it
    /// when derotating a stream, because the two-call form silently produces a stalled oscillator if
    /// the step is ever missed. Use [`Self::phasor`] when the phase must be read without advancing.
    #[cfg(feature = "complex")]
    #[must_use]
    pub fn phasor_then_step(&mut self) -> crate::complex::Complex<T>
    where
        T: FloatCore + From<f32>,
    {
        let phasor = self.phasor();
        self.step();
        phasor
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
        math::phase::phase_step_from_frequency(frequency_hz, sample_rate_hz)
    }

    /// Converts a signed phase step to frequency in Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate_hz` is not finite or `sample_rate_hz <= 0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frequency_from_phase_step(phase_step: i32, sample_rate_hz: f32) -> f32 {
        math::phase::frequency_from_phase_step(phase_step, sample_rate_hz)
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

    #[cfg(feature = "complex")]
    #[test]
    fn phasor_then_step_returns_the_current_phase_then_advances() {
        let mut advancing = Nco::<f32>::from_turns_per_sample(0.25);
        let mut manual = advancing.clone();

        for _ in 0..5 {
            let expected = manual.phasor();
            manual.step();
            let got = advancing.phasor_then_step();

            assert_eq!(got.re.to_bits(), expected.re.to_bits());
            assert_eq!(got.im.to_bits(), expected.im.to_bits());
            assert_eq!(advancing.phase(), manual.phase());
        }
    }

    #[test]
    fn every_conversion_delegates_to_math_phase() {
        for turns in [0.0_f32, 0.1, 0.25, -0.3, 0.499, 1.75, -2.5] {
            assert_eq!(
                Nco::<f32>::phase_word_from_turns(turns),
                math::phase::phase_word_from_turns(turns)
            );
            assert_eq!(
                Nco::<f32>::phase_step_from_turns_per_sample(turns),
                math::phase::phase_step_from_turns_per_sample(turns)
            );
            let radians = turns * core::f32::consts::TAU;
            assert_eq!(
                Nco::<f32>::phase_word_from_radians(radians),
                math::phase::phase_word_from_radians(radians)
            );
            assert_eq!(
                Nco::<f32>::phase_step_from_radians_per_sample(radians),
                math::phase::phase_step_from_radians_per_sample(radians)
            );
        }
        for step in [0_i32, 1, -1, i32::MAX, i32::MIN, 0x4000_0000] {
            assert_eq!(
                Nco::<f32>::turns_per_sample_from_phase_step(step).to_bits(),
                math::phase::turns_per_sample_from_phase_step(step).to_bits()
            );
            assert_eq!(
                Nco::<f32>::radians_per_sample_from_phase_step(step).to_bits(),
                math::phase::radians_per_sample_from_phase_step(step).to_bits()
            );
            assert_eq!(
                Nco::<f32>::frequency_from_phase_step(step, 48_000.0).to_bits(),
                math::phase::frequency_from_phase_step(step, 48_000.0).to_bits()
            );
        }
        for word in [0_u32, 1, 0x4000_0000, 0x8000_0000, 0xFFFF_FFFF] {
            assert_eq!(
                Nco::<f32>::turns_from_phase_word(word).to_bits(),
                math::phase::turns_from_phase_word(word).to_bits()
            );
            assert_eq!(
                Nco::<f32>::radians_from_phase_word(word).to_bits(),
                math::phase::radians_from_phase_word(word).to_bits()
            );
        }
        assert_eq!(
            Nco::<f32>::phase_step_from_frequency(1000.0, 48_000.0),
            math::phase::phase_step_from_frequency(1000.0, 48_000.0)
        );
    }

    #[test]
    fn from_turns_reads_back_its_own_rate() {
        assert_abs_diff_eq!(
            Nco::<f32>::from_turns(0.0, -0.125).turns_per_sample(),
            -0.125,
            epsilon = EPS
        );
    }

    #[test]
    fn from_turns_per_sample_matches_from_turns_at_zero_phase() {
        let nco = Nco::<f32>::from_turns_per_sample(-0.125);

        assert_eq!(nco.phase(), 0);
        assert_eq!(
            nco.phase_step(),
            Nco::<f32>::from_turns(0.0, -0.125).phase_step()
        );
    }

    #[test]
    fn normalized_setters_match_their_constructors() {
        let mut nco = Nco::<f32>::default();
        nco.set_phase(0x1234_5678);

        nco.set_turns_per_sample(0.125);
        assert_eq!(
            nco.phase_step(),
            Nco::<f32>::phase_step_from_turns_per_sample(0.125)
        );

        nco.set_radians_per_sample(-0.75);
        assert_eq!(
            nco.phase_step(),
            Nco::<f32>::phase_step_from_radians_per_sample(-0.75)
        );

        // A setter touches the rate only, never the accumulated phase.
        assert_eq!(nco.phase(), 0x1234_5678);
    }

    #[test]
    fn from_radians_keeps_a_small_negative_step_negative() {
        // Folding through `[0, 1)` turns would round a step this small away entirely at `f32`
        // precision, leaving a stopped oscillator instead of a slow backward rotation.
        let nco = Nco::<f32>::from_radians(0.0, -4.027e-7 * core::f32::consts::TAU);

        assert!(nco.phase_step() < 0);
        // The exact step is -1729.58 phase words, truncated towards zero.
        assert_eq!(nco.phase_step(), -1729);
    }

    #[test]
    fn from_radians_walks_the_unit_circle_in_quarter_turns() {
        let mut nco = Nco::<f32>::from_radians(0.0, core::f32::consts::TAU / 4.0);

        for &(sin, cos) in &[(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)] {
            let (got_sin, got_cos) = nco.sin_cos();
            assert_abs_diff_eq!(got_sin, sin, epsilon = EPS);
            assert_abs_diff_eq!(got_cos, cos, epsilon = EPS);
            nco.step();
        }
    }

    #[test]
    fn config_rate_constructors_match_the_nco_conversions() {
        let rate = 0.25_f32;

        assert_eq!(
            Config::<f32>::from_turns_per_sample(rate).phase_step(),
            Nco::<f32>::phase_step_from_turns_per_sample(rate)
        );
        assert_eq!(
            Config::<f32>::from_radians_per_sample(rate).phase_step(),
            Nco::<f32>::phase_step_from_radians_per_sample(rate)
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
