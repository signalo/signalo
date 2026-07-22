// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Polyphase FIR decimation.

use core::marker::PhantomData;
use core::ops::{Add, Mul};

use circular_buffer::FixedCircularBuffer;
use num_traits::{Num, Zero};

use crate::storage::{zero_filled_fixed_ring, AsSlice, RingBuffer};
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, MultirateFilter, Reset, WithConfig,
};

use super::filter_bank::{Config, PolyphaseFilterBank};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "alloc")]
use super::filter_bank::PolyphaseFilterBankVec;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The polyphase decimator's state.
#[derive(Clone, Debug)]
pub struct State<R> {
    /// Buffered input samples, one delay-line buffer per phase.
    pub taps: R,
    /// The phase branch that will receive the next input sample.
    pub phase: usize,
}

/// A stateful M-to-1 polyphase decimator.
///
/// This type owns one delay-line buffer per phase branch. Each consumed input
/// sample updates the selected phase buffer. The input commutator counts down
/// from the last phase to phase zero. Whenever phase zero is updated, the
/// decimator evaluates all phase branches and produces one output sample.
///
/// The first output is therefore produced only after `num_phases` input samples
/// have been consumed. This matches the usual M-to-1 decimator convention: a
/// complete input block advances all phase branches, then one decimated sample
/// is emitted.
///
/// Coefficients follow the phase-major ordering described by
/// [`PolyphaseFilterBank`].
///
/// # Prototype design
///
/// For this M-to-1 decimator, design dense prototype taps at the input sample
/// rate. The anti-aliasing filter sees the input-rate spectrum before
/// downsampling.
///
/// # Gain
///
/// This type does not apply gain correction after summing the phase outputs.
/// Decimators are usually constructed with a unity-passband-gain prototype. A
/// decimator is equivalent to filtering at the input rate and then keeping every
/// `M`th output sample, so downsampling changes sample spacing but does not
/// require any amplitude correction.
///
/// # Streaming
///
/// The [`MultirateFilter::process`] implementation supports streaming operation
/// with arbitrary input and output slice sizes. If the output slice is full
/// before the phase-zero input can be consumed, that input remains unconsumed and
/// should be passed again on a later [`process`](MultirateFilter::process) call.
///
/// When the output slice is empty, the decimator can still consume input samples
/// for non-zero phase branches. Only the sample that would update phase zero and
/// produce an output is deferred; pass it again as the first input sample on the
/// next call with output space available.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`PolyphaseDecimatorArray<T, N, H, P, K>`] for stack-allocated
///   coefficients and per-phase delay-line storage.
#[cfg_attr(
    feature = "alloc",
    doc = "- [`PolyphaseDecimatorVec<T, K>`] for heap-allocated coefficients and per-phase delay-line storage."
)]
#[cfg_attr(
    not(feature = "alloc"),
    doc = "- `PolyphaseDecimatorVec<T, K>` for heap-allocated coefficients and per-phase delay-line storage."
)]
///
/// # Complexity
///
/// - **Time per sample:** O(H/M) amortized, where H is the total tap count and M is the
///   decimation factor; M input samples trigger one full-filter convolution of H taps.
/// - **Space:** O(H); H/M delay samples per phase times M phases.
#[derive(Clone, Debug)]
pub struct PolyphaseDecimator<T, C, R, B, K = T> {
    bank: PolyphaseFilterBank<C>,
    state: State<R>,
    _pd: PhantomData<(T, B, K)>,
}

/// A polyphase decimator backed by fixed coefficient and per-phase delay-line
/// storage.
///
/// `N` is the total coefficient count. `H` is the number of coefficients in
/// each phase branch. `P` is the decimation factor and must match the
/// configuration's `num_phases`.
pub type PolyphaseDecimatorArray<T, const N: usize, const H: usize, const P: usize, K = T> =
    PolyphaseDecimator<T, [K; N], [FixedCircularBuffer<T, H>; P], FixedCircularBuffer<T, H>, K>;

/// A polyphase decimator backed by heap-allocated storage.
///
/// Requires the `alloc` feature.
#[cfg(feature = "alloc")]
pub type PolyphaseDecimatorVec<T, K = T> = PolyphaseDecimator<
    T,
    alloc::vec::Vec<K>,
    alloc::vec::Vec<HeapCircularBuffer<T>>,
    HeapCircularBuffer<T>,
    K,
>;

impl<T, C, R, B, K> PolyphaseDecimator<T, C, R, B, K>
where
    C: AsSlice<K>,
    R: AsSlice<B>,
    B: RingBuffer<T>,
{
    /// Creates a [`PolyphaseDecimator`] from an already-constructed `config` and
    /// buffers.
    ///
    /// Use this constructor when the per-phase delay-line storage is
    /// caller-owned or must be constructed with runtime capacities.
    ///
    /// The delay-line buffers are taken as-is with their current contents. Each
    /// buffer must contain `taps_per_phase` samples before its phase branch is
    /// evaluated.
    ///
    /// # Expected storage state
    ///
    /// For zero-padded cold-start behavior, prefill every buffer with
    /// `taps_per_phase` zeros before passing them here.
    ///
    /// The initial phase is set to the last branch so the first output is
    /// produced after one full decimation block.
    ///
    /// # Panics
    ///
    /// Panics if the number of delay-line buffers does not equal
    /// `config.num_phases`, or if any buffer capacity does not equal
    /// `config.taps_per_phase`.
    pub fn from_parts(config: Config<C>, taps: R) -> Self {
        let bank = PolyphaseFilterBank::from_parts(config);
        assert_eq!(
            taps.as_slice().len(),
            bank.num_phases(),
            "PolyphaseDecimator: taps count must equal num_phases"
        );
        for tap_buffer in taps.as_slice() {
            assert_eq!(
                tap_buffer.capacity(),
                bank.taps_per_phase(),
                "PolyphaseDecimator: taps capacity must equal taps_per_phase"
            );
        }
        Self {
            state: State {
                taps,
                phase: bank.num_phases() - 1,
            },
            bank,
            _pd: PhantomData,
        }
    }
}

impl<T, C, R, B, K> PolyphaseDecimator<T, C, R, B, K> {
    /// Returns the number of polyphase branches and decimation factor.
    #[must_use]
    pub fn num_phases(&self) -> usize {
        self.bank.num_phases()
    }

    /// Returns the number of coefficients in each phase branch.
    #[must_use]
    pub fn taps_per_phase(&self) -> usize {
        self.bank.taps_per_phase()
    }

    /// Returns the total number of coefficients.
    #[must_use]
    pub fn total_taps(&self) -> usize {
        self.bank.total_taps()
    }
}

impl<T, C, R, B, K> ConfigTrait for PolyphaseDecimator<T, C, R, B, K> {
    type Config = Config<C>;
}

impl<T, const N: usize, const H: usize, const P: usize, K> WithConfig
    for PolyphaseDecimatorArray<T, N, H, P, K>
where
    T: Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let taps = core::array::from_fn(|_| zero_filled_fixed_ring::<T, H>());
        Self::from_parts(config, taps)
    }
}

#[cfg(feature = "alloc")]
impl<T, K> PolyphaseDecimatorVec<T, K>
where
    T: Clone + Zero,
    K: Clone + Zero,
{
    /// Creates a heap-backed polyphase decimator from dense prototype
    /// coefficients.
    ///
    /// This convenience constructor allocates coefficient and per-phase
    /// delay-line storage. `prototype` is passed to
    /// [`PolyphaseFilterBankVec::from_prototype_taps`] for ordering and padding
    /// behavior. The decimator delay-line buffers are zero-filled.
    ///
    /// # Panics
    ///
    /// Panics if `decimation` is zero, `prototype` is empty, or the padded
    /// coefficient count overflows `usize`.
    #[must_use]
    pub fn from_prototype_taps(decimation: usize, prototype: &[K]) -> Self {
        let bank = PolyphaseFilterBankVec::from_prototype_taps(decimation, prototype);
        let mut taps = alloc::vec::Vec::with_capacity(bank.num_phases());
        for _ in 0..bank.num_phases() {
            let mut tap_buffer = HeapCircularBuffer::with_capacity(bank.taps_per_phase());
            for _ in 0..bank.taps_per_phase() {
                let _ = tap_buffer.push_back(T::zero());
            }
            taps.push(tap_buffer);
        }
        Self::from_parts(bank.into_guts(), taps)
    }
}

impl<T, C, R, B, K> ConfigRef for PolyphaseDecimator<T, C, R, B, K> {
    fn config_ref(&self) -> &Self::Config {
        self.bank.config_ref()
    }
}

impl<T, C, R, B, K> ConfigClone for PolyphaseDecimator<T, C, R, B, K>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.bank.config()
    }
}

impl<T, C, R, B, K> HasGuts for PolyphaseDecimator<T, C, R, B, K> {
    type Guts = (Config<C>, State<R>);
}

impl<T, C, R, B, K> FromGuts for PolyphaseDecimator<T, C, R, B, K> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self {
            bank: PolyphaseFilterBank::from_guts(config),
            state,
            _pd: PhantomData,
        }
    }
}

impl<T, C, R, B, K> IntoGuts for PolyphaseDecimator<T, C, R, B, K> {
    fn into_guts(self) -> Self::Guts {
        (self.bank.into_guts(), self.state)
    }
}

impl<T, const N: usize, const H: usize, const P: usize, K> Reset
    for PolyphaseDecimatorArray<T, N, H, P, K>
where
    T: Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.bank.into_guts())
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize, const H: usize, const P: usize, K> ResetMut
    for PolyphaseDecimatorArray<T, N, H, P, K>
where
    Self: Reset,
{
}

impl<T, C, R, B, K> MultirateFilter<T> for PolyphaseDecimator<T, C, R, B, K>
where
    T: Clone + Zero + Add<Output = T> + Mul<K, Output = T>,
    K: Clone,
    C: AsSlice<K>,
    R: AsSlice<B>,
    B: RingBuffer<T>,
{
    type Output = T;

    fn process(&mut self, input: &[T], output: &mut [Self::Output]) -> (usize, usize) {
        let mut input_consumed = 0;
        let mut output_produced = 0;

        while input_consumed < input.len() {
            let phase = self.state.phase;
            if phase == 0 && output_produced == output.len() {
                break;
            }

            self.state.taps.as_mut_slice()[phase].push_back(input[input_consumed].clone());
            input_consumed += 1;

            let output_ready = phase == 0;
            self.state.phase = if phase == 0 {
                self.num_phases() - 1
            } else {
                phase - 1
            };

            if output_ready {
                let decimated = (0..self.num_phases())
                    .map(|phase| {
                        self.bank
                            .execute(phase, self.state.taps.as_slice()[phase].iter())
                    })
                    .fold(T::zero(), |sum, partial| sum + partial);
                output[output_produced] = decimated;
                output_produced += 1;
            }
        }

        (input_consumed, output_produced)
    }
}

#[cfg(test)]
mod tests {
    use super::{PolyphaseDecimator, PolyphaseDecimatorArray, State};
    use crate::filters::fir::convolve::{Config as ConvolveConfig, ConvolveArray};
    use crate::filters::fir::polyphase::filter_bank::Config;
    use crate::traits::{
        guts::{FromGuts, IntoGuts},
        Filter, MultirateFilter, Reset, WithConfig,
    };

    #[test]
    fn process_decimates_all_outputs() {
        let mut decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });
        let input = [10, 20, 30, 40, 50, 60];
        let mut output = [0; 2];

        assert_eq!(decimator.process(&input, &mut output), (6, 2));
        assert_eq!(output, [100, 560]);
    }

    #[test]
    fn process_waits_for_full_decimation_block_before_output() {
        let mut decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });
        let mut output = [999];

        assert_eq!(decimator.process(&[10, 20], &mut output), (2, 0));
        assert_eq!(output, [999]);

        assert_eq!(decimator.process(&[30], &mut output), (1, 1));
        assert_eq!(output, [100]);
    }

    #[test]
    fn process_matches_convolve_then_downsample() {
        let mut decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });
        let mut convolve = ConvolveArray::<i32, 6>::with_config(ConvolveConfig {
            coefficients: [1, 2, 3, 4, 5, 6],
        });
        let input = [10, 20, 30, 40, 50, 60, 70, 80, 90];
        let mut expected = [0; 3];
        let mut expected_len = 0;

        for (index, sample) in input.iter().copied().enumerate() {
            let output = convolve.filter(sample);
            if index % 3 == 2 {
                expected[expected_len] = output;
                expected_len += 1;
            }
        }

        let mut output = [0; 3];
        assert_eq!(decimator.process(&input, &mut output), (9, 3));
        assert_eq!(output, expected);
    }

    #[test]
    fn process_leaves_phase_zero_input_unconsumed_when_output_is_full() {
        let mut decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });
        let input = [10, 20, 30, 40, 50, 60];
        let mut first_output = [0; 1];
        let mut second_output = [0; 1];

        assert_eq!(decimator.process(&input, &mut first_output), (5, 1));
        assert_eq!(first_output, [100]);

        assert_eq!(decimator.process(&input[5..], &mut second_output), (1, 1));
        assert_eq!(second_output, [560]);
    }

    #[test]
    fn empty_output_can_still_consume_until_output_boundary() {
        let mut decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });
        let mut output = [0; 1];

        assert_eq!(decimator.process(&[10, 20, 30, 40], &mut []), (2, 0));
        assert_eq!(decimator.process(&[30], &mut output), (1, 1));
        assert_eq!(output, [100]);
    }

    #[test]
    fn reset_clears_taps_and_phase() {
        let mut decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });
        let mut output = [0; 1];

        assert_eq!(decimator.process(&[10, 20], &mut []), (2, 0));
        let mut decimator = decimator.reset();

        assert_eq!(decimator.process(&[20, 30, 40], &mut output), (3, 1));
        assert_eq!(output, [160]);
    }

    #[test]
    #[should_panic(expected = "taps capacity must equal taps_per_phase")]
    fn from_parts_capacity_mismatch_panics() {
        let config = Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        };
        let taps = [
            circular_buffer::FixedCircularBuffer::<i32, 3>::new(),
            circular_buffer::FixedCircularBuffer::<i32, 3>::new(),
        ];
        let _ = PolyphaseDecimator::from_parts(config, taps);
    }

    #[test]
    fn guts_round_trip_preserves_phase() {
        let mut decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });
        assert_eq!(decimator.process(&[10, 20], &mut []), (2, 0));

        let guts = decimator.into_guts();
        let mut decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::from_guts(guts);
        let mut output = [0; 1];

        assert_eq!(decimator.process(&[30], &mut output), (1, 1));
        assert_eq!(output, [100]);
    }

    #[test]
    fn from_guts_accepts_existing_state() {
        let mut phase0 = circular_buffer::FixedCircularBuffer::<i32, 2>::new();
        let _ = phase0.push_back(10);
        let _ = phase0.push_back(40);
        let mut phase1 = circular_buffer::FixedCircularBuffer::<i32, 2>::new();
        let _ = phase1.push_back(0);
        let _ = phase1.push_back(30);
        let mut phase2 = circular_buffer::FixedCircularBuffer::<i32, 2>::new();
        let _ = phase2.push_back(0);
        let _ = phase2.push_back(20);

        let decimator = PolyphaseDecimatorArray::<i32, 6, 2, 3>::from_guts((
            Config {
                num_phases: 3,
                taps_per_phase: 2,
                coefficients: [1, 4, 2, 5, 3, 6],
            },
            State {
                taps: [phase0, phase1, phase2],
                phase: 2,
            },
        ));

        assert_eq!(decimator.state.phase, 2);
        assert!(decimator.state.taps[0].iter().copied().eq([10, 40]));
        assert!(decimator.state.taps[1].iter().copied().eq([0, 30]));
        assert!(decimator.state.taps[2].iter().copied().eq([0, 20]));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn vec_from_prototype_taps_decimates() {
        let mut decimator =
            super::PolyphaseDecimatorVec::<i32>::from_prototype_taps(3, &[1, 2, 3, 4, 5, 6]);
        let input = [10, 20, 30, 40, 50, 60];
        let mut output = [0; 2];

        assert_eq!(decimator.process(&input, &mut output), (6, 2));
        assert_eq!(output, [100, 560]);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn real_taps_complex_samples_match_independent_real_decimators() {
        use approx::assert_abs_diff_eq;

        use crate::complex::Complex32;

        let coefficients = [0.5_f32, 0.125, -0.25, 0.0625];
        let real_input = [1.0_f32, -2.0, 3.0, 5.0, -8.0];
        let imag_input = [13.0_f32, -21.0, 34.0, -55.0, 89.0];
        let complex_input = [
            Complex32::new(real_input[0], imag_input[0]),
            Complex32::new(real_input[1], imag_input[1]),
            Complex32::new(real_input[2], imag_input[2]),
            Complex32::new(real_input[3], imag_input[3]),
            Complex32::new(real_input[4], imag_input[4]),
        ];
        let mut real_decimator = PolyphaseDecimatorArray::<f32, 4, 2, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients,
        });
        let mut imag_decimator = PolyphaseDecimatorArray::<f32, 4, 2, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients,
        });
        let mut complex_decimator =
            PolyphaseDecimatorArray::<Complex32, 4, 2, 2, f32>::with_config(Config {
                num_phases: 2,
                taps_per_phase: 2,
                coefficients,
            });
        let mut real_output = [0.0; 2];
        let mut imag_output = [0.0; 2];
        let mut complex_output = [Complex32::new(0.0, 0.0); 2];

        assert_eq!(
            real_decimator.process(&real_input, &mut real_output),
            (5, 2)
        );
        assert_eq!(
            imag_decimator.process(&imag_input, &mut imag_output),
            (5, 2)
        );
        assert_eq!(
            complex_decimator.process(&complex_input, &mut complex_output),
            (5, 2)
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
}
