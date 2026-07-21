// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Polyphase FIR rational resampling.

use core::ops::{Add, Mul};

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::{Num, Zero};

use crate::storage::{zero_fill_ring, AsSlice, RingBuffer};
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, MultirateFilter, Reset, WithConfig,
};

use super::{
    filter_bank::Config as BankConfig,
    fir::{PolyphaseFir, PolyphaseFirArray},
};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "alloc")]
use super::fir::PolyphaseFirVec;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The rational resampler's configuration.
#[derive(Clone, Debug)]
pub struct Config<C> {
    /// Interpolation factor and number of coefficient phases.
    pub interpolation: usize,
    /// Decimation factor.
    pub decimation: usize,
    /// Number of coefficients in each phase branch.
    pub taps_per_phase: usize,
    /// Coefficients in the phase-major layout described by
    /// [`PolyphaseFilterBank`](super::filter_bank::PolyphaseFilterBank),
    /// with `interpolation` phases.
    pub coefficients: C,
}

/// The rational resampler's state.
#[derive(Clone, Debug)]
pub struct State {
    /// Phase accumulator.
    ///
    /// `phase < interpolation` means an output is pending for the current delay
    /// state. `phase >= interpolation` means input must be consumed before the
    /// next output can be produced.
    pub phase: usize,
}

/// A stateful `P`/`Q` polyphase rational resampler.
///
/// This type wraps [`PolyphaseFir`] and implements [`MultirateFilter`] using a
/// phase accumulator. The interpolation factor is the wrapped FIR's number of
/// phases. `decimation` is the phase increment after each produced output.
///
/// The first input sample is consumed before the first output is produced. This
/// matches the usual `upfirdn` timing where outputs are taken from indices
/// `0, Q, 2Q, ...` of the upsampled-and-filtered sequence.
///
/// Coefficients follow the phase-major ordering described by
/// [`PolyphaseFilterBank`](super::filter_bank::PolyphaseFilterBank).
///
/// # Prototype design
///
/// For this P/Q rational resampler, design dense prototype taps at
/// `input_rate * P`, the intermediate rate after interpolation and before
/// decimation. For low-pass anti-image and anti-aliasing prototypes, the lower
/// input/output Nyquist boundary at that intermediate rate is
/// `0.5 / max(P, Q)`.
///
/// # Gain
///
/// This type does not apply interpolation gain scaling. If a prototype designer
/// normalizes taps to unity passband gain, multiply the prototype coefficients
/// by `P` before polyphase construction when unity amplitude should be
/// preserved. This compensates for the interpolation step that inserts `P - 1`
/// zero-valued samples between input samples.
///
/// # Ratio reduction
///
/// The interpolation and decimation factors are used exactly as supplied; they
/// are not reduced by their greatest common divisor. If the factors share a
/// divisor, only the corresponding subset of phases is reached by the phase
/// accumulator, so the unreduced form may store redundant phase branches. Pass
/// reduced factors and coefficients packed for that reduced phase count when
/// minimal storage is required.
///
/// # Streaming
///
/// The [`MultirateFilter::process`] implementation supports streaming operation
/// with arbitrary input and output slice sizes. When output capacity is full,
/// the resampler can still consume input while advancing its phase accumulator
/// until the next output is pending. Once an output is pending, it consumes no
/// further input until that output is emitted. Pass the unconsumed input suffix
/// to the next call.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`RationalResamplerArray<T, N, H, K>`] for stack-allocated coefficients
///   and delay-line storage.
#[cfg_attr(
    feature = "alloc",
    doc = "- [`RationalResamplerVec<T, K>`] for heap-allocated coefficients and delay-line storage."
)]
#[cfg_attr(
    not(feature = "alloc"),
    doc = "- `RationalResamplerVec<T, K>` for heap-allocated coefficients and delay-line storage."
)]
/// - [`RationalResamplerRefMut<'_, T, C, K>`] for caller-owned delay-line
///   storage.
///
/// # Complexity
///
/// - **Time per sample:** O(H/P) amortized per output, where H is the total tap count and P the
///   interpolation factor; each output evaluates one phase branch of H/P taps, consuming on
///   average Q/P inputs (Q = decimation factor).
/// - **Space:** O(H) for coefficients plus O(H/P) for the shared delay line.
#[derive(Clone, Debug)]
pub struct RationalResampler<T, C, R, K = T> {
    fir: PolyphaseFir<T, C, R, K>,
    decimation: usize,
    state: State,
}

/// A rational resampler backed by fixed coefficient and delay-line storage.
///
/// `N` is the total coefficient count. `H` is the number of coefficients in
/// each phase branch and must match the configuration's `taps_per_phase`.
pub type RationalResamplerArray<T, const N: usize, const H: usize, K = T> =
    RationalResampler<T, [K; N], FixedCircularBuffer<T, H>, K>;

/// A rational resampler backed by heap-allocated storage.
///
/// Requires the `alloc` feature.
#[cfg(feature = "alloc")]
pub type RationalResamplerVec<T, K = T> =
    RationalResampler<T, alloc::vec::Vec<K>, HeapCircularBuffer<T>, K>;

/// A rational resampler that borrows caller-owned delay-line storage.
pub type RationalResamplerRefMut<'a, T, C, K = T> =
    RationalResampler<T, C, &'a mut CircularBuffer<T>, K>;

impl<T, C, R, K> RationalResampler<T, C, R, K>
where
    C: AsSlice<K>,
    R: RingBuffer<T>,
{
    /// Creates a [`RationalResampler`] from an already-constructed `config` and
    /// delay-line buffer.
    ///
    /// Use this constructor when the delay-line storage is caller-owned or must
    /// be constructed with a runtime capacity.
    ///
    /// The delay-line buffer is taken as-is with its current contents. It must
    /// contain `taps_per_phase` samples before the first output is produced.
    ///
    /// # Expected storage state
    ///
    /// For zero-padded cold-start behavior, prefill the buffer with
    /// `taps_per_phase` zeros before passing it here.
    ///
    /// # Panics
    ///
    /// Panics if `config.decimation` is zero, if `config.interpolation` or
    /// `config.taps_per_phase` is zero, if the coefficient count does not equal
    /// `config.interpolation * config.taps_per_phase`, or if the delay-line
    /// capacity does not equal `config.taps_per_phase`.
    pub fn from_parts(config: Config<C>, taps: R) -> Self {
        assert!(
            config.decimation > 0,
            "RationalResampler: decimation must be > 0"
        );
        let decimation = config.decimation;
        let fir = PolyphaseFir::from_parts(
            BankConfig {
                num_phases: config.interpolation,
                taps_per_phase: config.taps_per_phase,
                coefficients: config.coefficients,
            },
            taps,
        );
        Self::from_fir(fir, decimation)
    }
}

impl<T, C, R, K> RationalResampler<T, C, R, K> {
    fn from_fir(fir: PolyphaseFir<T, C, R, K>, decimation: usize) -> Self {
        assert!(decimation > 0, "RationalResampler: decimation must be > 0");
        Self {
            state: State {
                phase: fir.num_phases(),
            },
            fir,
            decimation,
        }
    }

    /// Returns the interpolation factor.
    #[must_use]
    pub fn interpolation(&self) -> usize {
        self.fir.num_phases()
    }

    /// Returns the decimation factor.
    #[must_use]
    pub fn decimation(&self) -> usize {
        self.decimation
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

#[cfg(feature = "alloc")]
impl<T, K> RationalResamplerVec<T, K>
where
    T: Zero,
    K: Clone + Zero,
{
    /// Creates a heap-backed rational resampler from dense prototype
    /// coefficients.
    ///
    /// This convenience constructor allocates coefficient and delay-line storage.
    /// `prototype` is passed to
    /// [`PolyphaseFilterBankVec::from_prototype_taps`](super::filter_bank::PolyphaseFilterBankVec::from_prototype_taps)
    /// using the supplied interpolation factor for ordering and padding
    /// behavior. The ratio is not reduced before packing.
    ///
    /// # Panics
    ///
    /// Panics if `interpolation` or `decimation` is zero, `prototype` is empty,
    /// or the padded coefficient count overflows `usize`.
    #[must_use]
    pub fn from_prototype_taps(interpolation: usize, decimation: usize, prototype: &[K]) -> Self {
        Self::from_fir(
            PolyphaseFirVec::from_prototype_taps(interpolation, prototype),
            decimation,
        )
    }
}

impl<T, C, R, K> ConfigTrait for RationalResampler<T, C, R, K> {
    type Config = Config<C>;
}

impl<T, const N: usize, const H: usize, K> WithConfig for RationalResamplerArray<T, N, H, K>
where
    T: Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let mut taps = FixedCircularBuffer::new();
        zero_fill_ring(&mut taps);
        Self::from_parts(config, taps)
    }
}

impl<T, C, R, K> ConfigClone for RationalResampler<T, C, R, K>
where
    PolyphaseFir<T, C, R, K>: ConfigClone<Config = BankConfig<C>>,
{
    fn config(&self) -> Self::Config {
        let bank = self.fir.config();
        Config {
            interpolation: bank.num_phases,
            decimation: self.decimation,
            taps_per_phase: bank.taps_per_phase,
            coefficients: bank.coefficients,
        }
    }
}

impl<T, C, R, K> HasGuts for RationalResampler<T, C, R, K>
where
    PolyphaseFir<T, C, R, K>: HasGuts,
{
    type Guts = (<PolyphaseFir<T, C, R, K> as HasGuts>::Guts, usize, State);
}

impl<T, C, R, K> FromGuts for RationalResampler<T, C, R, K>
where
    PolyphaseFir<T, C, R, K>: FromGuts + HasGuts,
{
    fn from_guts(guts: Self::Guts) -> Self {
        let (fir, decimation, state) = guts;
        assert!(decimation > 0, "RationalResampler: decimation must be > 0");
        Self {
            fir: PolyphaseFir::from_guts(fir),
            decimation,
            state,
        }
    }
}

impl<T, C, R, K> IntoGuts for RationalResampler<T, C, R, K>
where
    PolyphaseFir<T, C, R, K>: IntoGuts + HasGuts,
{
    fn into_guts(self) -> Self::Guts {
        (self.fir.into_guts(), self.decimation, self.state)
    }
}

impl<T, const N: usize, const H: usize, K> Reset for RationalResamplerArray<T, N, H, K>
where
    PolyphaseFirArray<T, N, H, K>: Reset,
{
    fn reset(self) -> Self {
        Self::from_fir(self.fir.reset(), self.decimation)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize, const H: usize, K> ResetMut for RationalResamplerArray<T, N, H, K> where
    Self: Reset
{
}

impl<T, C, R, K> MultirateFilter<T> for RationalResampler<T, C, R, K>
where
    T: Clone + Zero + Add<Output = T> + Mul<K, Output = T>,
    K: Clone,
    C: crate::storage::AsSlice<K>,
    R: RingBuffer<T>,
{
    type Output = T;

    fn process(&mut self, input: &[T], output: &mut [Self::Output]) -> (usize, usize) {
        let mut input_consumed = 0;
        let mut output_produced = 0;
        let interpolation = self.interpolation();

        loop {
            if self.state.phase >= interpolation {
                if input_consumed == input.len() {
                    break;
                }

                self.fir.push(input[input_consumed].clone());
                input_consumed += 1;
                self.state.phase -= interpolation;
                continue;
            }

            if output_produced == output.len() {
                break;
            }

            output[output_produced] = self.fir.execute(self.state.phase);
            output_produced += 1;
            self.state.phase += self.decimation;
        }

        (input_consumed, output_produced)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "alloc")]
    use super::RationalResamplerVec;
    use super::{Config, RationalResamplerArray};
    use crate::filters::fir::convolve::{Config as ConvolveConfig, ConvolveArray};
    use crate::traits::{
        guts::{FromGuts, IntoGuts},
        ConfigClone, Filter, MultirateFilter, Reset, WithConfig,
    };

    #[test]
    fn process_resamples_known_ratio() {
        let mut resampler = RationalResamplerArray::<i32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        let input = [10, 20, 30, 40];
        let mut output = [0; 3];

        assert_eq!(resampler.process(&input, &mut output), (4, 3));
        assert_eq!(output, [10, 80, 130]);
    }

    #[test]
    fn process_matches_upsample_convolve_then_downsample() {
        let mut resampler = RationalResamplerArray::<i32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        let mut convolve = ConvolveArray::<i32, 4>::with_config(ConvolveConfig {
            coefficients: [1, 2, 3, 4],
        });
        let input = [10, 20, 30, 40, 50, 60];
        let mut expected = [0; 4];
        let mut expected_len = 0;

        for (index, sample) in input
            .iter()
            .copied()
            .flat_map(|sample| [sample, 0])
            .enumerate()
        {
            let output = convolve.filter(sample);
            if index % 3 == 0 {
                expected[expected_len] = output;
                expected_len += 1;
            }
        }

        let mut output = [0; 4];
        assert_eq!(resampler.process(&input, &mut output), (6, 4));
        assert_eq!(expected_len, output.len());
        assert_eq!(output, expected);
    }

    #[test]
    fn process_can_resume_with_pending_output() {
        let mut resampler = RationalResamplerArray::<i32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        let input = [10, 20, 30, 40];
        let mut first_output = [0; 1];
        let mut second_output = [0; 1];
        let mut third_output = [0; 1];

        assert_eq!(resampler.process(&input, &mut first_output), (2, 1));
        assert_eq!(first_output, [10]);

        assert_eq!(resampler.process(&input[2..], &mut second_output), (2, 1));
        assert_eq!(second_output, [80]);

        assert_eq!(resampler.process(&[], &mut third_output), (0, 1));
        assert_eq!(third_output, [130]);
    }

    #[test]
    fn interpolation_only_matches_interpolator_timing() {
        let mut resampler = RationalResamplerArray::<i32, 3, 1>::with_config(Config {
            interpolation: 3,
            decimation: 1,
            taps_per_phase: 1,
            coefficients: [1, 2, 3],
        });
        let input = [10, 20];
        let mut output = [0; 6];

        assert_eq!(resampler.process(&input, &mut output), (2, 6));
        assert_eq!(output, [10, 20, 30, 20, 40, 60]);
    }

    #[test]
    fn decimation_only_uses_upfirdn_output_phase() {
        let mut resampler = RationalResamplerArray::<i32, 4, 4>::with_config(Config {
            interpolation: 1,
            decimation: 3,
            taps_per_phase: 4,
            coefficients: [1, 2, 3, 4],
        });
        let input = [10, 20, 30, 40, 50];
        let mut output = [0; 2];

        assert_eq!(resampler.process(&input, &mut output), (5, 2));
        assert_eq!(output, [10, 200]);
    }

    #[test]
    fn empty_output_can_consume_until_next_output_is_pending() {
        let mut resampler = RationalResamplerArray::<i32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        let mut output = [0; 1];

        assert_eq!(resampler.process(&[10], &mut []), (1, 0));
        assert_eq!(resampler.process(&[], &mut output), (0, 1));
        assert_eq!(output, [10]);
    }

    #[test]
    fn reset_clears_delay_line_and_phase() {
        let mut resampler = RationalResamplerArray::<i32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        let mut output = [0; 1];

        assert_eq!(resampler.process(&[10, 20], &mut output), (2, 1));
        let mut resampler = resampler.reset();

        assert_eq!(resampler.process(&[10], &mut output), (1, 1));
        assert_eq!(output, [10]);
    }

    #[test]
    fn guts_round_trip_preserves_pending_phase() {
        let mut resampler = RationalResamplerArray::<i32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        assert_eq!(resampler.process(&[10], &mut []), (1, 0));

        let guts = resampler.into_guts();
        let mut resampler = RationalResamplerArray::<i32, 4, 2>::from_guts(guts);
        let mut output = [0; 1];

        assert_eq!(resampler.process(&[], &mut output), (0, 1));
        assert_eq!(output, [10]);
    }

    #[test]
    fn config_clone_preserves_resampling_config() {
        let resampler = RationalResamplerArray::<i32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });

        let config = resampler.config();
        assert_eq!(config.interpolation, 2);
        assert_eq!(config.decimation, 3);
        assert_eq!(config.taps_per_phase, 2);
        assert_eq!(config.coefficients, [1, 3, 2, 4]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn vec_from_prototype_taps_resamples() {
        let mut resampler = RationalResamplerVec::<i32>::from_prototype_taps(2, 3, &[1, 2, 3, 4]);
        let input = [10, 20, 30, 40];
        let mut output = [0; 3];

        assert_eq!(resampler.process(&input, &mut output), (4, 3));
        assert_eq!(output, [10, 80, 130]);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn real_taps_complex_samples_match_independent_real_resamplers() {
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
        let mut real_resampler = RationalResamplerArray::<f32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients,
        });
        let mut imag_resampler = RationalResamplerArray::<f32, 4, 2>::with_config(Config {
            interpolation: 2,
            decimation: 3,
            taps_per_phase: 2,
            coefficients,
        });
        let mut complex_resampler =
            RationalResamplerArray::<Complex32, 4, 2, f32>::with_config(Config {
                interpolation: 2,
                decimation: 3,
                taps_per_phase: 2,
                coefficients,
            });
        let mut real_output = [0.0; 4];
        let mut imag_output = [0.0; 4];
        let mut complex_output = [Complex32::new(0.0, 0.0); 4];

        assert_eq!(
            real_resampler.process(&real_input, &mut real_output),
            (5, 4)
        );
        assert_eq!(
            imag_resampler.process(&imag_input, &mut imag_output),
            (5, 4)
        );
        assert_eq!(
            complex_resampler.process(&complex_input, &mut complex_output),
            (5, 4)
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
