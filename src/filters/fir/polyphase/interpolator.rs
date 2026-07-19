// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Polyphase FIR interpolation.

use core::ops::{Add, Mul};

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::{Num, Zero};

use crate::storage::{zero_filled_fixed_ring, AsSlice, RingBuffer};
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, MultirateFilter, Reset, WithConfig,
};

use super::{
    filter_bank::Config,
    fir::{PolyphaseFir, PolyphaseFirArray},
};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The polyphase interpolator's state.
#[derive(Clone, Debug)]
pub struct State {
    /// The phase branch that will produce the next output sample.
    ///
    /// `phase == num_phases` means no phase output is pending.
    pub phase: usize,
}

/// A stateful 1-to-P polyphase interpolator.
///
/// This type wraps [`PolyphaseFir`] and implements [`MultirateFilter`]. Each
/// consumed input sample is pushed into the shared delay line, then phases
/// `0..P` are evaluated in order.
///
/// The [`MultirateFilter::process`] implementation supports streaming operation
/// with arbitrary input and output slice sizes. If the output buffer fills before
/// all phases are produced, the next call resumes from the pending phase without
/// consuming another input sample first.
///
/// An empty output slice consumes no input because every accepted input sample
/// has phase outputs to produce.
///
/// The `phase` state starts at `num_phases`, which means no output is pending.
/// After an input sample is consumed, `phase` advances upward from zero until
/// all phases for that sample have been produced.
///
/// Coefficients follow the phase-major ordering described by
/// [`PolyphaseFilterBank`](super::filter_bank::PolyphaseFilterBank).
///
/// # Gain
///
/// This type does not apply interpolation gain scaling. To preserve the input
/// amplitude, the prototype filter must have passband gain equal to the
/// interpolation factor. A unity-gain prototype therefore produces output
/// attenuated by `1 / P`; multiply its coefficients by `P` before construction,
/// or design the prototype with the required gain.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`PolyphaseInterpolatorArray<T, N, H, K>`] for stack-allocated
///   coefficients and delay-line storage.
#[cfg_attr(
    feature = "alloc",
    doc = "- [`PolyphaseInterpolatorVec<T, K>`] for heap-allocated coefficients and delay-line storage."
)]
#[cfg_attr(
    not(feature = "alloc"),
    doc = "- `PolyphaseInterpolatorVec<T, K>` for heap-allocated coefficients and delay-line storage."
)]
/// - [`PolyphaseInterpolatorRefMut<'_, T, C, K>`] for caller-owned delay-line
///   storage.
///
/// # Complexity
///
/// - **Time per sample:** O(H) per input sample, where H is the total tap count; each input
///   triggers P phase evaluations of H/P taps each.
/// - **Space:** O(H) for coefficients plus O(H/P) for the shared delay line.
#[derive(Clone, Debug)]
pub struct PolyphaseInterpolator<T, C, R, K = T> {
    fir: PolyphaseFir<T, C, R, K>,
    state: State,
}

/// A polyphase interpolator backed by fixed coefficient and delay-line storage.
///
/// `N` is the total coefficient count. `H` is the number of coefficients in
/// each phase branch and must match the configuration's `taps_per_phase`.
pub type PolyphaseInterpolatorArray<T, const N: usize, const H: usize, K = T> =
    PolyphaseInterpolator<T, [K; N], FixedCircularBuffer<T, H>, K>;

/// A polyphase interpolator backed by heap-allocated storage.
///
/// Requires the `alloc` feature.
#[cfg(feature = "alloc")]
pub type PolyphaseInterpolatorVec<T, K = T> =
    PolyphaseInterpolator<T, alloc::vec::Vec<K>, HeapCircularBuffer<T>, K>;

/// A polyphase interpolator that borrows caller-owned delay-line storage.
pub type PolyphaseInterpolatorRefMut<'a, T, C, K = T> =
    PolyphaseInterpolator<T, C, &'a mut CircularBuffer<T>, K>;

impl<T, C, R, K> PolyphaseInterpolator<T, C, R, K>
where
    C: AsSlice<K>,
    R: RingBuffer<T>,
{
    /// Creates a [`PolyphaseInterpolator`] from an already-constructed `config`
    /// and delay-line buffer.
    ///
    /// Use this constructor when the delay-line storage is caller-owned or must
    /// be constructed with a runtime capacity.
    ///
    /// The delay-line buffer is taken as-is with its current contents. It must
    /// contain `taps_per_phase` samples before the first phase is evaluated.
    ///
    /// # Expected storage state
    ///
    /// For zero-padded cold-start behavior, prefill the buffer with
    /// `taps_per_phase` zeros before passing it here.
    ///
    /// # Panics
    ///
    /// Panics if `config.num_phases` or `config.taps_per_phase` is zero, if the
    /// coefficient count does not equal `config.num_phases *
    /// config.taps_per_phase`, or if the delay-line capacity does not equal
    /// `config.taps_per_phase`.
    pub fn from_parts(config: Config<C>, taps: R) -> Self {
        Self::from_fir(PolyphaseFir::from_parts(config, taps))
    }
}

impl<T, C, R, K> PolyphaseInterpolator<T, C, R, K> {
    fn from_fir(fir: PolyphaseFir<T, C, R, K>) -> Self {
        Self {
            state: State {
                phase: fir.num_phases(),
            },
            fir,
        }
    }

    /// Returns the number of polyphase branches and interpolation factor.
    #[must_use]
    pub fn num_phases(&self) -> usize {
        self.fir.num_phases()
    }

    /// Returns the number of coefficients in each phase branch.
    #[must_use]
    pub fn taps_per_phase(&self) -> usize {
        self.fir.taps_per_phase()
    }

    /// Returns the total number of coefficients.
    #[must_use]
    pub fn total_taps(&self) -> usize {
        self.fir.total_taps()
    }
}

impl<T, C, R, K> ConfigTrait for PolyphaseInterpolator<T, C, R, K> {
    type Config = Config<C>;
}

impl<T, const N: usize, const H: usize, K> WithConfig for PolyphaseInterpolatorArray<T, N, H, K>
where
    T: Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self::from_parts(config, zero_filled_fixed_ring::<T, H>())
    }
}

#[cfg(feature = "alloc")]
impl<T, K> PolyphaseInterpolatorVec<T, K>
where
    T: Zero,
    K: Clone + Zero,
{
    /// Creates a heap-backed polyphase interpolator from dense prototype
    /// coefficients.
    ///
    /// This convenience constructor allocates coefficient and delay-line storage.
    /// `prototype` is passed to
    /// [`PolyphaseFirVec::from_prototype_taps`](super::fir::PolyphaseFirVec::from_prototype_taps)
    /// for ordering and padding behavior.
    ///
    /// # Panics
    ///
    /// Panics if `interpolation` is zero, `prototype` is empty, or the padded
    /// coefficient count overflows `usize`.
    #[must_use]
    pub fn from_prototype_taps(interpolation: usize, prototype: &[K]) -> Self {
        Self::from_fir(super::fir::PolyphaseFirVec::from_prototype_taps(
            interpolation,
            prototype,
        ))
    }
}

impl<T, C, R, K> ConfigRef for PolyphaseInterpolator<T, C, R, K> {
    fn config_ref(&self) -> &Self::Config {
        self.fir.config_ref()
    }
}

impl<T, C, R, K> ConfigClone for PolyphaseInterpolator<T, C, R, K>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.fir.config()
    }
}

impl<T, C, R, K> HasGuts for PolyphaseInterpolator<T, C, R, K>
where
    PolyphaseFir<T, C, R, K>: HasGuts,
{
    type Guts = (<PolyphaseFir<T, C, R, K> as HasGuts>::Guts, State);
}

impl<T, C, R, K> FromGuts for PolyphaseInterpolator<T, C, R, K>
where
    PolyphaseFir<T, C, R, K>: FromGuts + HasGuts,
{
    fn from_guts(guts: Self::Guts) -> Self {
        let (fir, state) = guts;
        Self {
            fir: PolyphaseFir::from_guts(fir),
            state,
        }
    }
}

impl<T, C, R, K> IntoGuts for PolyphaseInterpolator<T, C, R, K>
where
    PolyphaseFir<T, C, R, K>: IntoGuts + HasGuts,
{
    fn into_guts(self) -> Self::Guts {
        (self.fir.into_guts(), self.state)
    }
}

impl<T, const N: usize, const H: usize, K> Reset for PolyphaseInterpolatorArray<T, N, H, K>
where
    PolyphaseFirArray<T, N, H, K>: Reset,
{
    fn reset(self) -> Self {
        Self::from_fir(self.fir.reset())
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize, const H: usize, K> ResetMut for PolyphaseInterpolatorArray<T, N, H, K> where
    Self: Reset
{
}

impl<T, C, R, K> MultirateFilter<T> for PolyphaseInterpolator<T, C, R, K>
where
    T: Clone + Zero + Add<Output = T> + Mul<K, Output = T>,
    K: Clone,
    C: AsSlice<K>,
    R: RingBuffer<T>,
{
    type Output = T;

    fn process(&mut self, input: &[T], output: &mut [Self::Output]) -> (usize, usize) {
        let mut input_consumed = 0;
        let mut output_produced = 0;
        let num_phases = self.num_phases();

        while output_produced < output.len() {
            if self.state.phase == num_phases {
                if input_consumed == input.len() {
                    break;
                }

                self.fir.push(input[input_consumed].clone());
                input_consumed += 1;
                self.state.phase = 0;
            }

            output[output_produced] = self.fir.execute(self.state.phase);
            output_produced += 1;
            self.state.phase += 1;
        }

        (input_consumed, output_produced)
    }
}

#[cfg(test)]
mod tests {
    use super::{PolyphaseInterpolator, PolyphaseInterpolatorArray, PolyphaseInterpolatorRefMut};
    use crate::filters::fir::polyphase::filter_bank::Config;
    use crate::traits::{
        guts::{FromGuts, IntoGuts},
        MultirateFilter, Reset, WithConfig,
    };

    #[test]
    fn process_interpolates_all_phases() {
        let mut interpolator = PolyphaseInterpolatorArray::<i32, 3, 1>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 1,
            coefficients: [1, 2, 3],
        });
        let input = [10, 20];
        let mut output = [0; 6];

        assert_eq!(interpolator.process(&input, &mut output), (2, 6));
        assert_eq!(output, [10, 20, 30, 20, 40, 60]);
    }

    #[test]
    fn process_resumes_pending_phases_before_consuming_input() {
        let mut interpolator = PolyphaseInterpolatorArray::<i32, 3, 1>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 1,
            coefficients: [1, 2, 3],
        });
        let input = [10, 20];
        let mut first_output = [0; 2];
        let mut second_output = [0; 4];

        assert_eq!(interpolator.process(&input, &mut first_output), (1, 2));
        assert_eq!(first_output, [10, 20]);

        assert_eq!(
            interpolator.process(&input[1..], &mut second_output),
            (1, 4)
        );
        assert_eq!(second_output, [30, 20, 40, 60]);
    }

    #[test]
    fn process_can_drain_pending_output_without_new_input() {
        let mut interpolator = PolyphaseInterpolatorArray::<i32, 3, 1>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 1,
            coefficients: [1, 2, 3],
        });
        let input = [10];
        let mut first_output = [0; 1];
        let mut second_output = [0; 2];

        assert_eq!(interpolator.process(&input, &mut first_output), (1, 1));
        assert_eq!(first_output, [10]);

        assert_eq!(interpolator.process(&[], &mut second_output), (0, 2));
        assert_eq!(second_output, [20, 30]);
    }

    #[test]
    fn empty_output_consumes_no_input() {
        let mut interpolator = PolyphaseInterpolatorArray::<i32, 3, 1>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 1,
            coefficients: [1, 2, 3],
        });

        assert_eq!(interpolator.process(&[10], &mut []), (0, 0));
    }

    #[test]
    fn multi_tap_phases_use_fir_history() {
        let mut interpolator = PolyphaseInterpolatorArray::<i32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        let input = [10, 20, 30];
        let mut output = [0; 6];

        assert_eq!(interpolator.process(&input, &mut output), (3, 6));
        assert_eq!(output, [10, 20, 50, 80, 90, 140]);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn real_taps_complex_samples_match_independent_real_interpolators() {
        use approx::assert_abs_diff_eq;

        use crate::complex::Complex32;

        let coefficients = [0.5_f32, -0.25, 0.125, 0.0625];
        let real_input = [1.0_f32, -2.0, 3.0];
        let imag_input = [5.0_f32, -8.0, 13.0];
        let complex_input = [
            Complex32::new(real_input[0], imag_input[0]),
            Complex32::new(real_input[1], imag_input[1]),
            Complex32::new(real_input[2], imag_input[2]),
        ];
        let mut real_interpolator = PolyphaseInterpolatorArray::<f32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients,
        });
        let mut imag_interpolator = PolyphaseInterpolatorArray::<f32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients,
        });
        let mut complex_interpolator =
            PolyphaseInterpolatorArray::<Complex32, 4, 2, f32>::with_config(Config {
                num_phases: 2,
                taps_per_phase: 2,
                coefficients,
            });
        let mut real_output = [0.0; 6];
        let mut imag_output = [0.0; 6];
        let mut complex_output = [Complex32::new(0.0, 0.0); 6];

        assert_eq!(
            real_interpolator.process(&real_input, &mut real_output),
            (3, 6)
        );
        assert_eq!(
            imag_interpolator.process(&imag_input, &mut imag_output),
            (3, 6)
        );
        assert_eq!(
            complex_interpolator.process(&complex_input, &mut complex_output),
            (3, 6)
        );

        for ((complex, real), imag) in complex_output
            .iter()
            .zip(real_output.iter())
            .zip(imag_output.iter())
        {
            assert_abs_diff_eq!(complex.re, *real, epsilon = 1e-6);
            assert_abs_diff_eq!(complex.im, *imag, epsilon = 1e-6);
        }
    }

    #[test]
    fn reset_clears_pending_phase_and_taps() {
        let mut interpolator = PolyphaseInterpolatorArray::<i32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        let mut output = [0; 1];

        assert_eq!(interpolator.process(&[10], &mut output), (1, 1));
        assert_eq!(output, [10]);

        let mut interpolator = interpolator.reset();
        let mut output = [0; 2];

        assert_eq!(interpolator.process(&[20], &mut output), (1, 2));
        assert_eq!(output, [20, 40]);
    }

    #[test]
    fn ref_mut_uses_caller_owned_delay_line() {
        let config = Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        };
        let mut taps = circular_buffer::FixedCircularBuffer::<i32, 2>::new();
        let _ = taps.push_back(0);
        let _ = taps.push_back(0);
        let mut interpolator: PolyphaseInterpolatorRefMut<'_, i32, [i32; 4]> =
            PolyphaseInterpolator::from_parts(config, &mut taps);
        let mut output = [0; 2];

        assert_eq!(interpolator.process(&[10], &mut output), (1, 2));
        assert_eq!(output, [10, 20]);
    }

    #[test]
    fn guts_round_trip_preserves_pending_phase() {
        let mut interpolator = PolyphaseInterpolatorArray::<i32, 3, 1>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 1,
            coefficients: [1, 2, 3],
        });
        let mut first_output = [0; 1];
        assert_eq!(interpolator.process(&[10], &mut first_output), (1, 1));

        let guts = interpolator.into_guts();
        let mut interpolator = PolyphaseInterpolatorArray::<i32, 3, 1>::from_guts(guts);
        let mut second_output = [0; 2];

        assert_eq!(interpolator.process(&[], &mut second_output), (0, 2));
        assert_eq!(second_output, [20, 30]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn vec_from_prototype_taps_interpolates() {
        let mut interpolator =
            super::PolyphaseInterpolatorVec::<i32>::from_prototype_taps(2, &[1, 2, 3, 4]);
        let input = [10, 20];
        let mut output = [0; 4];

        assert_eq!(interpolator.process(&input, &mut output), (2, 4));
        assert_eq!(output, [10, 20, 50, 80]);
    }
}
