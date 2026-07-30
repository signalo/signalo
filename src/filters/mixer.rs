// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Complex mixing filter for frequency translation.
//!
//! A `Mixer` multiplies each input sample by a rotating phasor, shifting the
//! signal's spectrum without altering its shape.
//!
//! Difference equation: `y[n] = x[n] · e^(j·θ[n])`
//!
//! where `θ[n]` is the driving oscillator's phase. The multiply is pointwise, so
//! the filter has no delay and no transient. The operation is also called
//! frequency translation, up- or downconversion, or rotation.
//!
//! # Use cases
//!
//! - Bringing a carrier offset to zero before demodulation.
//! - Placing a baseband signal at an offset before transmission.
//! - Centring a channel of interest before filtering it.
//!
//! # Related
//!
//! - [`crate::sources::oscillator::nco::Nco`] for the oscillator that drives the
//!   phasor and sets the shift.
//! - [`crate::filters::iir::loop_filter`] for the loop filter of a carrier
//!   recovery loop that retunes the oscillator.
//! - [`crate::filters::fir`] and [`crate::filters::iir`] for filters that shape
//!   the spectrum rather than move it.

use num_traits::float::FloatCore;

use crate::complex::Complex;
use crate::sources::oscillator::nco::{self, Nco};
#[cfg(feature = "derive")]
use crate::traits::ResetMut;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

/// Multiplies each input sample by the driving oscillator's phasor, advancing the
/// oscillator one step per sample.
///
/// # Sign convention
///
/// The phasor is `cos(phase) + j sin(phase)`, so mixing translates the input
/// **up** in frequency by the oscillator's rate. Negate the rate to translate
/// down, as removing a carrier offset requires:
///
/// ```
/// # use signalo::filters::mixer::Mixer;
/// # use signalo::sources::oscillator::nco::Nco;
/// # use signalo::traits::Filter;
/// # use signalo::complex::Complex32;
/// // A tone at +0.05 cycles per sample, brought down to DC.
/// let mut tone = Nco::<f32>::from_turns_per_sample(0.05);
/// let mut mixer = Mixer::new(Nco::<f32>::from_turns_per_sample(-0.05));
///
/// for _ in 0..64 {
///     let mixed = mixer.filter(tone.phasor_then_step());
///     assert!((mixed.im).abs() < 1.0e-3);
/// }
/// ```
///
/// # Precision
///
/// `T` is the output type, not the working type. The phasor comes from [`Nco`],
/// which interpolates an `f32` table, so `Mixer<f64>` widens `f32` values rather
/// than refining them and is no more accurate than `Mixer<f32>`. See [`Nco`] for
/// the bound.
///
/// # Complexity
///
/// - **Time per sample:** O(1); one phasor evaluation and one complex multiply.
/// - **Space:** O(1); the oscillator's phase word and phase step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mixer<T = f32> {
    nco: Nco<T>,
}

impl<T> Mixer<T> {
    /// Creates a mixer driven by `nco`.
    #[must_use]
    pub const fn new(nco: Nco<T>) -> Self {
        Self { nco }
    }

    /// Returns a reference to the driving oscillator.
    #[must_use]
    pub const fn nco(&self) -> &Nco<T> {
        &self.nco
    }

    /// Returns a mutable reference to the driving oscillator.
    ///
    /// Retuning through this is how a feedback loop steers the mixer from an error
    /// signal. A rate change leaves the accumulated phase untouched.
    #[must_use]
    pub const fn nco_mut(&mut self) -> &mut Nco<T> {
        &mut self.nco
    }
}

impl<T> ConfigTrait for Mixer<T> {
    type Config = nco::Config<T>;
}

impl<T> StateTrait for Mixer<T> {
    type State = nco::State<T>;
}

impl<T> WithConfig for Mixer<T>
where
    T: FloatCore + From<f32>,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self {
            nco: Nco::with_config(config),
        }
    }
}

impl<T> Default for Mixer<T>
where
    Nco<T>: Default,
{
    fn default() -> Self {
        Self {
            nco: Nco::default(),
        }
    }
}

impl<T> ConfigRef for Mixer<T> {
    fn config_ref(&self) -> &Self::Config {
        self.nco.config_ref()
    }
}

impl<T> ConfigClone for Mixer<T>
where
    nco::Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.nco.config()
    }
}

impl<T> StateMut for Mixer<T> {
    #[doc(hidden)]
    fn state_mut(&mut self) -> &mut Self::State {
        self.nco.state_mut()
    }
}

impl<T> HasGuts for Mixer<T> {
    type Guts = Nco<T>;
}

impl<T> FromGuts for Mixer<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        Self { nco: guts }
    }
}

impl<T> IntoGuts for Mixer<T> {
    fn into_guts(self) -> Self::Guts {
        self.nco
    }
}

impl<T> Reset for Mixer<T>
where
    T: FloatCore + From<f32>,
{
    fn reset(self) -> Self {
        Self {
            nco: self.nco.reset(),
        }
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for Mixer<T> where Self: Reset {}

impl<T> Filter<Complex<T>> for Mixer<T>
where
    T: FloatCore + From<f32>,
{
    type Output = Complex<T>;

    fn filter(&mut self, input: Complex<T>) -> Self::Output {
        input * self.nco.phasor_then_step()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::complex::Complex32;

    // A mixed sample is a product of two table lookups, and the round-trip tests below chain two
    // mixers, so four lookups contribute. Derived from the table's own bound rather than measured,
    // so it tracks a change to the table's resolution.
    const EPS: f32 = 4.0 * crate::math::phase::MAX_ABS_ERROR as f32;

    #[test]
    fn mixing_by_the_negated_rate_brings_a_tone_to_dc() {
        let rate = 0.05_f32;
        let mut tone = Nco::<f32>::from_turns_per_sample(rate);
        let mut mixer = Mixer::new(Nco::<f32>::from_turns_per_sample(-rate));

        for _ in 0..256 {
            let mixed = mixer.filter(tone.phasor_then_step());
            assert_abs_diff_eq!(mixed.re, 1.0, epsilon = EPS);
            assert_abs_diff_eq!(mixed.im, 0.0, epsilon = EPS);
        }
    }

    #[test]
    fn mixing_up_then_down_restores_the_input() {
        let rate = 0.017_f32;
        let mut up = Mixer::new(Nco::<f32>::from_turns_per_sample(rate));
        let mut down = Mixer::new(Nco::<f32>::from_turns_per_sample(-rate));

        for n in 0..256 {
            let input = Complex32::new(n as f32 * 0.01 - 1.0, 0.5 - n as f32 * 0.003);
            let round_tripped = down.filter(up.filter(input));
            assert_abs_diff_eq!(round_tripped.re, input.re, epsilon = EPS);
            assert_abs_diff_eq!(round_tripped.im, input.im, epsilon = EPS);
        }
    }

    #[test]
    fn mixing_shifts_upward_not_downward() {
        // A tone at +rate mixed with +rate must land at 2 * rate, not at DC. Catches a conjugated
        // phasor, which would look correct to a round-trip test.
        let rate = 0.05_f32;
        let mut tone = Nco::<f32>::from_turns_per_sample(rate);
        let mut mixer = Mixer::new(Nco::<f32>::from_turns_per_sample(rate));
        let mut expected = Nco::<f32>::from_turns_per_sample(2.0 * rate);

        for _ in 0..64 {
            let mixed = mixer.filter(tone.phasor_then_step());
            let want = expected.phasor_then_step();
            assert_abs_diff_eq!(mixed.re, want.re, epsilon = EPS);
            assert_abs_diff_eq!(mixed.im, want.im, epsilon = EPS);
        }
    }

    #[test]
    fn the_oscillator_advances_one_step_per_sample() {
        let mut mixer = Mixer::new(Nco::<f32>::from_turns_per_sample(0.013));
        let mut reference = Nco::<f32>::from_turns_per_sample(0.013);

        for _ in 0..32 {
            let _ = mixer.filter(Complex32::new(1.0, 0.0));
            reference.step();
            assert_eq!(mixer.nco().phase(), reference.phase());
        }
    }

    #[test]
    fn retuning_through_nco_mut_leaves_the_phase_alone() {
        let mut mixer = Mixer::new(Nco::<f32>::from_turns_per_sample(0.1));
        for _ in 0..4 {
            let _ = mixer.filter(Complex32::new(1.0, 0.0));
        }

        let phase = mixer.nco().phase();
        mixer.nco_mut().set_turns_per_sample(-0.2);

        assert_eq!(mixer.nco().phase(), phase);
        assert_eq!(
            mixer.nco().phase_step(),
            Nco::<f32>::phase_step_from_turns_per_sample(-0.2)
        );
    }

    #[test]
    fn reset_rewinds_the_phase_but_keeps_the_rate() {
        let mut mixer = Mixer::new(Nco::<f32>::from_turns_per_sample(0.1));
        for _ in 0..8 {
            let _ = mixer.filter(Complex32::new(1.0, 0.0));
        }
        assert_ne!(mixer.nco().phase(), 0);

        let step = mixer.nco().phase_step();
        let mixer = mixer.reset();

        assert_eq!(mixer.nco().phase(), 0);
        assert_eq!(mixer.nco().phase_step(), step);
    }

    #[test]
    fn with_config_round_trips_through_config_ref() {
        let config = *Nco::<f32>::from_turns_per_sample(0.125).config_ref();
        let mixer = Mixer::<f32>::with_config(config);

        assert_eq!(*mixer.config_ref(), config);
        assert_eq!(mixer.config(), config);
        assert_eq!(mixer.nco().phase(), 0);
    }

    #[test]
    fn guts_round_trip_preserves_the_oscillator() {
        let mut mixer = Mixer::new(Nco::<f32>::from_turns_per_sample(0.05));
        let _ = mixer.filter(Complex32::new(1.0, 0.0));

        let expected = mixer.clone();
        let rebuilt = Mixer::from_guts(mixer.into_guts());

        assert_eq!(rebuilt, expected);
    }

    #[test]
    fn equal_mixers_compare_equal_and_diverge_once_stepped() {
        let mut a = Mixer::new(Nco::<f32>::from_turns_per_sample(0.05));
        let b = a.clone();
        assert_eq!(a, b);

        let _ = a.filter(Complex32::new(1.0, 0.0));
        assert_ne!(a, b);
    }

    #[test]
    fn a_zero_rate_mixer_is_the_identity() {
        let mut mixer = Mixer::<f32>::default();

        // Away from zero, because a complex multiply forms `im = re * 0 + im * 1`, which turns a
        // negative zero positive. That is a property of the multiply, not of the mixer.
        for n in 1..16 {
            let input = Complex32::new(n as f32, -(n as f32));
            let output = mixer.filter(input);
            assert_eq!(output.re.to_bits(), input.re.to_bits());
            assert_eq!(output.im.to_bits(), input.im.to_bits());
        }
    }
}
