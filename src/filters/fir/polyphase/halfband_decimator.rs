// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Half-band FIR decimation by two.
//!
//! A half-band low-pass of `4 J - 1` taps cuts off at a quarter of its sample rate, is zero at
//! every even offset from its center tap other than the center itself, and mirrors around the
//! center. Split into the two polyphase branches of a 2-to-1 decimator, one branch is the center
//! tap alone, a gain and a delay, and the other holds the `2 J` odd-offset taps, mirrored pairs
//! that share a multiplication. Each output therefore takes `J + 1` multiplications.
//!
//! A windowed-sinc low-pass with a cutoff of 0.25 cycles per sample and `4 J - 1` taps, such as
//! [`kaiser::lowpass_with_beta_vec`](crate::filters::fir::design::windowed_sinc::kaiser::lowpass_with_beta_vec),
//! is a half-band.

use core::marker::PhantomData;
use core::ops::{Add, Mul};

use num_traits::{Num, Zero};

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, MultirateFilter, Reset, WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The half-band decimator's configuration.
#[derive(Clone, Debug)]
pub struct Config<C, K> {
    /// The center tap.
    pub center: K,
    /// The taps at the odd offsets `1, 3, ..., 2 J - 1` from the center, nearest first. Each
    /// one multiplies the two samples that far before and after the center sample.
    pub coefficients: C,
}

/// Blocks of outputs that [`HalfbandDecimatorVec`] sizes its windows for.
#[cfg(feature = "alloc")]
const VEC_BLOCK: usize = 256;

/// The half-band decimator's state.
#[derive(Clone, Debug)]
pub struct State<S> {
    /// The even window then the odd window, `2 J - 1 + B` samples each for blocks of `B`
    /// outputs: the `2 J - 1` newest even or odd input samples, oldest first, then room for `B`
    /// more.
    pub history: S,
    /// Whether the next input sample completes an output. The even sample it completes is the
    /// first one after the even window's `2 J - 1`.
    pub completes_output: bool,
}

/// A stateful 2-to-1 half-band decimator.
///
/// The output is the convolution of the input with a half-band low-pass of `4 J - 1` taps,
/// keeping the outputs at the second, fourth, and every further even-numbered input. A
/// half-band's taps are zero at every even offset from its center except the center itself,
/// and mirror around it, so each output takes `J + 1` multiplications: the center sample times
/// the center tap, and each mirrored pair of samples, added first, times their shared tap.
///
/// # Streaming
///
/// The [`MultirateFilter::process`] implementation accepts arbitrary input and output slice
/// sizes, with the same conventions as
/// [`PolyphaseDecimator`](super::decimator::PolyphaseDecimator): the first output follows two
/// inputs, and when the output slice is full, the input that would complete the next output is
/// left unconsumed. The result does not depend on how the input is split into calls.
///
/// The input goes through in blocks of up to `B` outputs. Each block's even and odd input
/// samples are copied into two windows after the newest `2 J - 1` earlier ones, so every loop
/// over the block's outputs reads contiguous samples, then the windows slide. Larger blocks
/// spend less per output on the slides.
///
/// # Type aliases
///
/// - [`HalfbandDecimatorArray<T, J, H, K>`] for stack-allocated coefficients and history.
#[cfg_attr(
    feature = "alloc",
    doc = "- [`HalfbandDecimatorVec<T, K>`] for heap-allocated coefficients and history."
)]
#[cfg_attr(
    not(feature = "alloc"),
    doc = "- `HalfbandDecimatorVec<T, K>` for heap-allocated coefficients and history."
)]
/// - [`HalfbandDecimatorRefMut`] for caller-owned history.
///
/// # Complexity
///
/// - **Time per output:** O(J), with `J + 1` multiplications and `2 J` additions.
/// - **Space:** O(J + B), with `J + 1` coefficients and `4 J - 2 + 2 B` samples of history.
#[derive(Clone, Debug)]
pub struct HalfbandDecimator<T, C, S, K = T> {
    config: Config<C, K>,
    state: State<S>,
    _pd: PhantomData<T>,
}

/// A half-band decimator backed by arrays: `J` coefficients and `H` samples of history, where
/// `H` must equal `2 (2 J - 1 + B)` for blocks of `B >= 1` outputs.
pub type HalfbandDecimatorArray<T, const J: usize, const H: usize, K = T> =
    HalfbandDecimator<T, [K; J], [T; H], K>;

/// A half-band decimator backed by heap-allocated coefficients and history.
///
/// Requires the `alloc` feature.
#[cfg(feature = "alloc")]
pub type HalfbandDecimatorVec<T, K = T> =
    HalfbandDecimator<T, alloc::vec::Vec<K>, alloc::vec::Vec<T>, K>;

/// A half-band decimator that borrows its history.
///
/// Construct via [`HalfbandDecimator::from_parts`].
pub type HalfbandDecimatorRefMut<'a, T, C, K = T> = HalfbandDecimator<T, C, &'a mut [T], K>;

impl<T, C, S, K> HalfbandDecimator<T, C, S, K>
where
    C: AsSlice<K>,
    S: AsSlice<T>,
{
    /// Creates a [`HalfbandDecimator`] from a `config` and `history` of `2 (2 J - 1 + B)`
    /// samples for blocks of `B` outputs, laid out as [`State::history`] and taken as-is.
    ///
    /// Zero-fill `history` for the usual zero-padded cold start. The first output follows two
    /// inputs.
    ///
    /// # Panics
    ///
    /// Panics if `config.coefficients` is empty, or if `history` does not hold
    /// `2 (2 J - 1 + B)` samples for some `B >= 1`.
    pub fn from_parts(config: Config<C, K>, history: S) -> Self {
        let pairs = config.coefficients.as_slice().len();
        assert!(
            pairs > 0,
            "HalfbandDecimator: coefficients must not be empty"
        );
        let len = history.as_slice().len();
        assert!(
            len.is_multiple_of(2) && len / 2 > 2 * pairs - 1,
            "HalfbandDecimator: history must hold 2 (2 J - 1 + B) samples for J coefficients and B >= 1"
        );
        Self {
            config,
            state: State {
                history,
                completes_output: false,
            },
            _pd: PhantomData,
        }
    }

    /// Returns the prototype's tap count, `4 J - 1`.
    #[must_use]
    pub fn num_taps(&self) -> usize {
        4 * self.config.coefficients.as_slice().len() - 1
    }
}

impl<T, C, S, K> ConfigTrait for HalfbandDecimator<T, C, S, K> {
    type Config = Config<C, K>;
}

impl<T, const J: usize, const H: usize, K> WithConfig for HalfbandDecimatorArray<T, J, H, K>
where
    T: Num,
{
    type Output = Self;

    /// Creates a [`HalfbandDecimatorArray`] with zero-filled history.
    ///
    /// # Panics
    ///
    /// Panics if `J` is zero or `H` is not `2 (2 J - 1 + B)` for some `B >= 1`.
    fn with_config(config: Self::Config) -> Self::Output {
        Self::from_parts(config, core::array::from_fn(|_| T::zero()))
    }
}

#[cfg(feature = "alloc")]
impl<T, K> WithConfig for HalfbandDecimatorVec<T, K>
where
    T: Clone + Zero,
{
    type Output = Self;

    /// Creates a [`HalfbandDecimatorVec`] with zero-filled history sized from the coefficients,
    /// for blocks of 256 outputs.
    ///
    /// # Panics
    ///
    /// Panics if `config.coefficients` is empty.
    fn with_config(config: Self::Config) -> Self::Output {
        let pairs = config.coefficients.len();
        assert!(
            pairs > 0,
            "HalfbandDecimator: coefficients must not be empty"
        );
        Self::from_parts(
            config,
            alloc::vec![T::zero(); 2 * (2 * pairs - 1 + VEC_BLOCK)],
        )
    }
}

#[cfg(feature = "alloc")]
impl<T, K> HalfbandDecimatorVec<T, K>
where
    T: Clone + Zero,
    K: Clone,
{
    /// Creates a heap-backed half-band decimator from a whole half-band `prototype` of
    /// `4 J - 1` taps, with zero-filled history.
    ///
    /// Only the center tap and the odd-offset taps after it are read. The prototype is taken to
    /// mirror them, with zeros at the other even offsets. Windowed-sinc designs can differ from
    /// their mirror image, or from zero, in the last bits, which this ignores.
    ///
    /// # Panics
    ///
    /// Panics if `prototype.len()` is not `4 J - 1` for some `J >= 1`.
    #[must_use]
    pub fn from_prototype_taps(prototype: &[K]) -> Self {
        let len = prototype.len();
        assert!(
            len >= 3 && (len + 1).is_multiple_of(4),
            "HalfbandDecimator: prototype must have 4 J - 1 taps (got {len})"
        );
        let center = len / 2;
        let coefficients = prototype[center + 1..].iter().step_by(2).cloned().collect();
        Self::with_config(Config {
            center: prototype[center].clone(),
            coefficients,
        })
    }
}

impl<T, C, S, K> ConfigRef for HalfbandDecimator<T, C, S, K> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C, S, K> ConfigClone for HalfbandDecimator<T, C, S, K>
where
    Config<C, K>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C, S, K> HasGuts for HalfbandDecimator<T, C, S, K> {
    type Guts = (Config<C, K>, State<S>);
}

impl<T, C, S, K> FromGuts for HalfbandDecimator<T, C, S, K> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self {
            config,
            state,
            _pd: PhantomData,
        }
    }
}

impl<T, C, S, K> IntoGuts for HalfbandDecimator<T, C, S, K> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, C, S, K> Reset for HalfbandDecimator<T, C, S, K>
where
    T: Zero,
    S: AsSlice<T>,
{
    /// Restores the zero-padded cold start: zero history, and the first output after two
    /// inputs.
    fn reset(mut self) -> Self {
        self.state.history.as_mut_slice().fill_with(T::zero);
        self.state.completes_output = false;
        self
    }
}

#[cfg(feature = "derive")]
impl<T, C, S, K> ResetMut for HalfbandDecimator<T, C, S, K> where Self: Reset {}

// Decimates the `output.len()` samples after the `2 J - 1` kept at the front of each window,
// then keeps the newest `2 J - 1` there. Output `t` is the center `even[t + J]` times `center`,
// then each mirrored pair `odd[t + J - 1 - i]` and `odd[t + J + i]`, nearest first, added and
// times its tap.
fn decimate<T, K>(center: &K, taps: &[K], even: &mut [T], odd: &mut [T], output: &mut [T])
where
    T: Clone + Add<Output = T> + Mul<K, Output = T>,
    K: Clone,
{
    let j = taps.len();
    let kept = 2 * j - 1;
    assert!(even.len() >= kept + output.len() && odd.len() >= kept + output.len());
    let samples = even[j..].iter().zip(&odd[j - 1..]).zip(&odd[j..]);
    for (y, ((c, a), b)) in output.iter_mut().zip(samples) {
        *y = c.clone() * center.clone() + (a.clone() + b.clone()) * taps[0].clone();
    }
    for (i, tap) in taps.iter().enumerate().skip(1) {
        let samples = odd[j - 1 - i..].iter().zip(&odd[j + i..]);
        for (y, (a, b)) in output.iter_mut().zip(samples) {
            *y = y.clone() + (a.clone() + b.clone()) * tap.clone();
        }
    }
    slide(even, kept, output.len());
    slide(odd, kept, output.len());
}

// Moves the newest `kept` of `window[..kept + n]` to the front.
fn slide<T: Clone>(window: &mut [T], kept: usize, n: usize) {
    if n >= kept {
        let (front, back) = window.split_at_mut(n);
        front[..kept].clone_from_slice(&back[..kept]);
    } else {
        window[..kept + n].rotate_left(n);
    }
}

impl<T, C, S, K> MultirateFilter<T> for HalfbandDecimator<T, C, S, K>
where
    T: Clone + Add<Output = T> + Mul<K, Output = T>,
    K: Clone,
    C: AsSlice<K>,
    S: AsSlice<T>,
{
    type Output = T;

    fn process(&mut self, input: &[T], output: &mut [Self::Output]) -> (usize, usize) {
        let center = &self.config.center;
        let taps = self.config.coefficients.as_slice();
        let history = self.state.history.as_mut_slice();
        let (even, odd) = history.split_at_mut(history.len() / 2);
        let kept = 2 * taps.len() - 1;
        let block = even.len() - kept;

        let (mut consumed, mut produced) = (0, 0);
        if self.state.completes_output {
            let (Some(sample), Some(y)) = (input.first(), output.first_mut()) else {
                return (0, 0);
            };
            odd[kept] = sample.clone();
            decimate(center, taps, even, odd, core::slice::from_mut(y));
            self.state.completes_output = false;
            (consumed, produced) = (1, 1);
        }

        let (pairs, _) = input[consumed..].as_chunks::<2>();
        for (pairs, output) in pairs
            .chunks(block)
            .zip(output[produced..].chunks_mut(block))
        {
            let pairs = &pairs[..pairs.len().min(output.len())];
            for (e, [x, _]) in even[kept..].iter_mut().zip(pairs) {
                *e = x.clone();
            }
            for (o, [_, x]) in odd[kept..].iter_mut().zip(pairs) {
                *o = x.clone();
            }
            decimate(center, taps, even, odd, &mut output[..pairs.len()]);
            consumed += 2 * pairs.len();
            produced += pairs.len();
        }

        // An even sample waits for the odd one that completes its output.
        if let Some(sample) = input.get(consumed) {
            even[kept] = sample.clone();
            self.state.completes_output = true;
            consumed += 1;
        }

        (consumed, produced)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::filters::fir::convolve::{Config as ConvolveConfig, ConvolveArray};
    use crate::traits::Filter;

    // A half-band prototype with distinct integer taps: center 16, pairs 9, -2, 1.
    const PROTOTYPE: [i64; 11] = [1, 0, -2, 0, 9, 16, 9, 0, -2, 0, 1];

    // Blocks of 6 outputs, so the windows slide both by more and by fewer than the 5 samples
    // they keep.
    fn decimator() -> HalfbandDecimatorArray<i64, 3, 22> {
        HalfbandDecimatorArray::with_config(Config {
            center: 16,
            coefficients: [9, -2, 1],
        })
    }

    // Every second output of the full convolution, at the second, fourth, ... input.
    fn reference(input: &[i64]) -> alloc::vec::Vec<i64> {
        let mut convolve = ConvolveArray::<i64, 11>::with_config(ConvolveConfig {
            coefficients: PROTOTYPE,
        });
        input
            .iter()
            .map(|&x| convolve.filter(x))
            .skip(1)
            .step_by(2)
            .collect()
    }

    fn input() -> [i64; 41] {
        core::array::from_fn(|k| (k as i64 * 37) % 23 - 11)
    }

    #[test]
    fn matches_convolve_then_every_second_output() {
        let input = input();
        let mut output = [0; 20];
        assert_eq!(decimator().process(&input, &mut output), (41, 20));
        assert_eq!(output.as_slice(), reference(&input).as_slice());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn every_pair_count_matches_convolve_then_every_second_output() {
        use crate::filters::fir::convolve::ConvolveVec;

        let input = input();
        for pairs in [1, 2, 5] {
            let mut prototype = alloc::vec![0_i64; 4 * pairs - 1];
            let center = 2 * pairs - 1;
            prototype[center] = 32;
            for j in 0..pairs {
                let tap = 7 - 3 * j as i64;
                prototype[center - 1 - 2 * j] = tap;
                prototype[center + 1 + 2 * j] = tap;
            }
            let mut convolve = ConvolveVec::with_config(ConvolveConfig {
                coefficients: prototype.clone(),
            });
            let expected: alloc::vec::Vec<i64> = input
                .iter()
                .map(|&x| convolve.filter(x))
                .skip(1)
                .step_by(2)
                .collect();
            let mut output = [0; 20];
            let mut decimator = HalfbandDecimatorVec::<i64>::from_prototype_taps(&prototype);
            assert_eq!(decimator.process(&input, &mut output), (41, 20));
            assert_eq!(output.as_slice(), expected.as_slice(), "{pairs} pairs");
        }
    }

    #[test]
    fn any_chunking_and_output_space_gives_the_same_stream() {
        let input = input();
        let expected = reference(&input);
        for chunk in [1, 2, 3, 5, 8, 13] {
            for space in [0, 1, 2, 3, 7] {
                let mut decimator = decimator();
                let (mut position, mut produced) = (0, alloc::vec::Vec::new());
                let mut stalls = 0;
                while position < input.len() {
                    let end = (position + chunk).min(input.len());
                    let mut output = [0; 7];
                    // Zero output space can only consume up to the next completing input.
                    let space = if stalls > 0 { 1 } else { space };
                    let (consumed, emitted) =
                        decimator.process(&input[position..end], &mut output[..space]);
                    // With output space, every call consumes or produces.
                    assert!(
                        space == 0 || consumed + emitted > 0,
                        "no progress at {position}"
                    );
                    produced.extend_from_slice(&output[..emitted]);
                    stalls = if consumed == 0 { stalls + 1 } else { 0 };
                    position += consumed;
                }
                assert_eq!(produced, expected, "chunk {chunk}, space {space}");
            }
        }
    }

    #[test]
    fn full_output_leaves_the_completing_input_unconsumed() {
        let mut decimator = decimator();
        let input = input();
        let expected = reference(&input);
        let mut output = [0; 1];
        // Input 1 completes the first output, input 3 the next, which waits for space.
        assert_eq!(decimator.process(&input, &mut output), (3, 1));
        assert_eq!(output[0], expected[0]);
        assert_eq!(decimator.process(&input[3..], &mut []), (0, 0));
        assert_eq!(decimator.process(&input[3..], &mut output), (2, 1));
        assert_eq!(output[0], expected[1]);
    }

    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    fn float_output_does_not_depend_on_the_chunking() {
        let input: [f32; 64] = core::array::from_fn(|k| (k as f32 * 0.77).sin());
        let taps: alloc::vec::Vec<f32> =
            crate::filters::fir::design::windowed_sinc::kaiser::lowpass_with_beta_vec(
                31, 5.0, 0.25,
            );
        let mut whole = HalfbandDecimatorVec::<f32>::from_prototype_taps(&taps);
        let mut expected = [0.0; 32];
        assert_eq!(whole.process(&input, &mut expected), (64, 32));
        for chunk in [1, 3, 7] {
            let mut decimator = HalfbandDecimatorVec::<f32>::from_prototype_taps(&taps);
            let mut produced = alloc::vec::Vec::new();
            for part in input.chunks(chunk) {
                let mut output = [0.0; 8];
                let (consumed, emitted) = decimator.process(part, &mut output);
                assert_eq!(consumed, part.len());
                produced.extend_from_slice(&output[..emitted]);
            }
            let bits = |v: &[f32]| {
                v.iter()
                    .map(|x| x.to_bits())
                    .collect::<alloc::vec::Vec<_>>()
            };
            assert_eq!(bits(&produced), bits(&expected), "chunk {chunk}");
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn vec_from_prototype_taps_matches_the_array() {
        let input = input();
        let mut array_output = [0; 20];
        let mut vec_output = [0; 20];
        decimator().process(&input, &mut array_output);
        HalfbandDecimatorVec::<i64>::from_prototype_taps(&PROTOTYPE)
            .process(&input, &mut vec_output);
        assert_eq!(array_output, vec_output);
    }

    #[test]
    fn reset_restores_the_cold_start() {
        let input = input();
        let mut first = [0; 20];
        let mut again = [0; 20];
        let mut decimator = decimator();
        decimator.process(&input[..7], &mut [0; 4]);
        let mut decimator = decimator.reset();
        decimator.process(&input, &mut again);
        HalfbandDecimatorArray::<i64, 3, 22>::with_config(Config {
            center: 16,
            coefficients: [9, -2, 1],
        })
        .process(&input, &mut first);
        assert_eq!(first, again);
    }

    #[test]
    #[should_panic(expected = "history must hold 2 (2 J - 1 + B) samples")]
    fn wrong_history_length_panics() {
        let _ = HalfbandDecimatorArray::<i64, 3, 10>::with_config(Config {
            center: 16,
            coefficients: [9, -2, 1],
        });
    }
}
