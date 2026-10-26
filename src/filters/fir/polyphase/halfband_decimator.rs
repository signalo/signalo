// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Half-band FIR decimation by two.
//!
//! A half-band low-pass of `4 J - 1` taps cuts off at a quarter of its sample rate, and is zero
//! at every even offset from its center tap other than the center itself. Split into the two
//! polyphase branches of a 2-to-1 decimator, one branch is the center tap alone, a gain and a
//! delay, and the other holds the `2 J` taps at odd offsets from the center.
//!
//! A windowed-sinc low-pass with a cutoff of 0.25 cycles per sample and `4 J - 1` taps, such as
//! [`kaiser::lowpass_with_beta_vec`](crate::filters::fir::design::windowed_sinc::kaiser::lowpass_with_beta_vec),
//! is a half-band.

use core::marker::PhantomData;
use core::ops::{Add, Mul};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;
use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::Zero;

use crate::storage::{dot_product_reversed_coefficients, AsSlice, RingBuffer};
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
    /// The `2 J` taps at the odd offsets `-(2 J - 1), ..., -1, 1, ..., 2 J - 1` from the center,
    /// in prototype order.
    pub coefficients: C,
}

/// The half-band decimator's state.
#[derive(Clone, Debug)]
pub struct State<D, R> {
    /// The `J` newest even input samples. The oldest is the newest output's center sample.
    pub even: D,
    /// The `2 J` newest odd input samples.
    pub odd: R,
}

/// A stateful 2-to-1 half-band decimator.
///
/// The output is the convolution of the input with a half-band low-pass of `4 J - 1` taps,
/// keeping the outputs at the second, fourth, and every further even-numbered input. Each output
/// is the even input `J - 1` pairs back times the center tap, plus the dot product of the `2 J`
/// newest odd inputs with the odd-offset taps, `2 J + 1` multiplications in all.
///
/// # Streaming
///
/// [`MultirateFilter::process`] consumes the input in pairs, an even and then an odd sample,
/// and produces one output per pair. A trailing sample without its pair is left unconsumed, as
/// are the pairs that find the output slice full, so the caller passes them again with the next
/// input. The result does not depend on how the input is split into calls. The rings start out
/// zero-filled, so the first outputs carry the usual zero-padded start-up transient.
///
/// # Type aliases
///
/// - [`HalfbandDecimatorArray<T, J, H, K>`] for stack-allocated coefficients and rings.
#[cfg_attr(
    feature = "alloc",
    doc = "- [`HalfbandDecimatorVec<T, K>`] for heap-allocated coefficients and rings."
)]
#[cfg_attr(
    not(feature = "alloc"),
    doc = "- `HalfbandDecimatorVec<T, K>` for heap-allocated coefficients and rings."
)]
/// - [`HalfbandDecimatorRefMut`] for caller-owned rings.
///
/// # Complexity
///
/// - **Time per output:** O(J), with `2 J + 1` multiplications and as many additions.
/// - **Space:** O(J), with `2 J + 1` coefficients and `3 J` samples in the rings.
#[derive(Clone, Debug)]
pub struct HalfbandDecimator<T, C, D, R, K = T> {
    config: Config<C, K>,
    state: State<D, R>,
    _pd: PhantomData<T>,
}

/// A half-band decimator backed by arrays: `H = 2 J` coefficients, a ring of `J` even samples
/// and a ring of `H` odd samples.
pub type HalfbandDecimatorArray<T, const J: usize, const H: usize, K = T> =
    HalfbandDecimator<T, [K; H], FixedCircularBuffer<T, J>, FixedCircularBuffer<T, H>, K>;

/// A half-band decimator backed by heap-allocated coefficients and rings.
///
/// Requires the `alloc` feature.
#[cfg(feature = "alloc")]
pub type HalfbandDecimatorVec<T, K = T> =
    HalfbandDecimator<T, alloc::vec::Vec<K>, HeapCircularBuffer<T>, HeapCircularBuffer<T>, K>;

/// A half-band decimator that borrows its rings.
///
/// Construct via [`HalfbandDecimator::from_parts`].
pub type HalfbandDecimatorRefMut<'a, T, C, K = T> =
    HalfbandDecimator<T, C, &'a mut CircularBuffer<T>, &'a mut CircularBuffer<T>, K>;

impl<T, C, D, R, K> HalfbandDecimator<T, C, D, R, K>
where
    C: AsSlice<K>,
    D: RingBuffer<T>,
    R: RingBuffer<T>,
{
    /// Creates a [`HalfbandDecimator`] from a `config` and two full rings, taken as-is: `even`
    /// of capacity `J` and `odd` of capacity `2 J`.
    ///
    /// Zero-filled rings give the usual zero-padded cold start. The first output follows two
    /// inputs.
    ///
    /// # Panics
    ///
    /// Panics if `config.coefficients` does not hold `2 J` taps for some `J >= 1`, or if the
    /// rings are not full with those capacities.
    pub fn from_parts(config: Config<C, K>, even: D, odd: R) -> Self {
        let taps = config.coefficients.as_slice().len();
        assert!(
            taps > 0 && taps.is_multiple_of(2),
            "HalfbandDecimator: coefficients must hold 2 J taps for some J >= 1"
        );
        assert!(
            even.capacity() == taps / 2
                && odd.capacity() == taps
                && even.is_full()
                && odd.is_full(),
            "HalfbandDecimator: rings must be full, even with J samples and odd with 2 J"
        );
        Self {
            config,
            state: State { even, odd },
            _pd: PhantomData,
        }
    }

    /// Returns the prototype's tap count, `4 J - 1`.
    #[must_use]
    pub fn num_taps(&self) -> usize {
        2 * self.config.coefficients.as_slice().len() - 1
    }
}

impl<T, C, D, R, K> ConfigTrait for HalfbandDecimator<T, C, D, R, K> {
    type Config = Config<C, K>;
}

impl<T, const J: usize, const H: usize, K> WithConfig for HalfbandDecimatorArray<T, J, H, K>
where
    T: Zero,
{
    type Output = Self;

    /// Creates a [`HalfbandDecimatorArray`] with zero-filled rings.
    ///
    /// # Panics
    ///
    /// Panics if `J` is zero or `H` is not `2 J`.
    fn with_config(config: Self::Config) -> Self::Output {
        let mut even = FixedCircularBuffer::<T, J>::new();
        let mut odd = FixedCircularBuffer::<T, H>::new();
        RingBuffer::fill_with(&mut even, T::zero);
        RingBuffer::fill_with(&mut odd, T::zero);
        Self::from_parts(config, even, odd)
    }
}

#[cfg(feature = "alloc")]
impl<T, K> WithConfig for HalfbandDecimatorVec<T, K>
where
    T: Zero,
{
    type Output = Self;

    /// Creates a [`HalfbandDecimatorVec`] with zero-filled rings sized from the coefficients.
    ///
    /// # Panics
    ///
    /// Panics if `config.coefficients` does not hold `2 J` taps for some `J >= 1`.
    fn with_config(config: Self::Config) -> Self::Output {
        let taps = config.coefficients.len();
        assert!(
            taps > 0 && taps.is_multiple_of(2),
            "HalfbandDecimator: coefficients must hold 2 J taps for some J >= 1"
        );
        let ring = |capacity| {
            let mut ring = HeapCircularBuffer::with_capacity(capacity);
            RingBuffer::fill_with(&mut ring, T::zero);
            ring
        };
        Self::from_parts(config, ring(taps / 2), ring(taps))
    }
}

#[cfg(feature = "alloc")]
impl<T, K> HalfbandDecimatorVec<T, K>
where
    T: Zero,
    K: Clone,
{
    /// Creates a heap-backed half-band decimator from a whole half-band `prototype` of
    /// `4 J - 1` taps, with zero-filled rings.
    ///
    /// It reads the center tap and the `2 J` taps at odd offsets from it, and skips the zeros at
    /// the other even offsets. Windowed-sinc designs can differ from zero there in the last bits,
    /// which this ignores.
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
        Self::with_config(Config {
            center: prototype[len / 2].clone(),
            coefficients: prototype.iter().step_by(2).cloned().collect(),
        })
    }
}

impl<T, C, D, R, K> ConfigRef for HalfbandDecimator<T, C, D, R, K> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C, D, R, K> ConfigClone for HalfbandDecimator<T, C, D, R, K>
where
    Config<C, K>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C, D, R, K> HasGuts for HalfbandDecimator<T, C, D, R, K> {
    type Guts = (Config<C, K>, State<D, R>);
}

impl<T, C, D, R, K> FromGuts for HalfbandDecimator<T, C, D, R, K> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self {
            config,
            state,
            _pd: PhantomData,
        }
    }
}

impl<T, C, D, R, K> IntoGuts for HalfbandDecimator<T, C, D, R, K> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, C, D, R, K> Reset for HalfbandDecimator<T, C, D, R, K>
where
    T: Zero,
    D: RingBuffer<T>,
    R: RingBuffer<T>,
{
    /// Restores the zero-padded cold start.
    fn reset(mut self) -> Self {
        self.state.even.fill_with(T::zero);
        self.state.odd.fill_with(T::zero);
        self
    }
}

#[cfg(feature = "derive")]
impl<T, C, D, R, K> ResetMut for HalfbandDecimator<T, C, D, R, K> where Self: Reset {}

impl<T, C, D, R, K, O> MultirateFilter<T> for HalfbandDecimator<T, C, D, R, K>
where
    T: Clone + Zero + Mul<K, Output = O>,
    K: Clone,
    C: AsSlice<K>,
    D: RingBuffer<T>,
    R: RingBuffer<T>,
    O: Zero + Add<Output = O>,
{
    type Output = O;

    fn process(&mut self, input: &[T], output: &mut [Self::Output]) -> (usize, usize) {
        let taps = self.config.coefficients.as_slice();
        let (pairs, _) = input.as_chunks::<2>();
        let mut produced = 0;
        for ([even, odd], y) in pairs.iter().zip(output) {
            let _ = self.state.even.push_back(even.clone());
            let _ = self.state.odd.push_back(odd.clone());
            // The oldest of the `J` newest even samples is the center, `J - 1` pairs back.
            let center = self.state.even.front().cloned().unwrap_or_else(T::zero);
            *y = center * self.config.center.clone()
                + dot_product_reversed_coefficients(self.state.odd.as_slices(), taps);
            produced += 1;
        }
        (2 * produced, produced)
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

    // Its center and odd-offset taps, `J = 3`.
    fn config() -> Config<[i64; 6], i64> {
        Config {
            center: 16,
            coefficients: [1, -2, 9, 9, -2, 1],
        }
    }

    fn decimator() -> HalfbandDecimatorArray<i64, 3, 6> {
        HalfbandDecimatorArray::with_config(config())
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
        assert_eq!(decimator().process(&input, &mut output), (40, 20));
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
            assert_eq!(decimator.process(&input, &mut output), (40, 20));
            assert_eq!(output.as_slice(), expected.as_slice(), "{pairs} pairs");
        }
    }

    // Feeds `input` in chunks of `chunk` into an output slice of `space`, keeping whatever the
    // decimator leaves unconsumed in front of the next chunk, as a caller would.
    fn stream<T: Clone>(
        decimator: &mut impl MultirateFilter<T, Output = T>,
        input: &[T],
        chunk: usize,
        space: usize,
        zero: T,
    ) -> (usize, alloc::vec::Vec<T>) {
        let (mut position, mut end, mut produced) = (0, 0, alloc::vec::Vec::new());
        loop {
            end = (end + chunk).min(input.len());
            let mut output = alloc::vec![zero.clone(); space];
            let (consumed, emitted) = decimator.process(&input[position..end], &mut output);
            produced.extend_from_slice(&output[..emitted]);
            position += consumed;
            if end == input.len() && consumed == 0 {
                return (position, produced);
            }
        }
    }

    #[test]
    fn any_chunking_and_output_space_gives_the_same_stream() {
        let input = input();
        let expected = reference(&input);
        for chunk in [1, 2, 3, 5, 8, 13] {
            for space in [1, 2, 3, 7] {
                let (consumed, produced) = stream(&mut decimator(), &input, chunk, space, 0);
                // The last of the 41 inputs has no pair, so it stays with the caller.
                assert_eq!(consumed, 40, "chunk {chunk}, space {space}");
                assert_eq!(produced, expected, "chunk {chunk}, space {space}");
            }
        }
    }

    #[test]
    fn unpaired_and_unplaced_inputs_stay_unconsumed() {
        let mut decimator = decimator();
        let input = input();
        let expected = reference(&input);
        let mut output = [0; 1];
        assert_eq!(decimator.process(&input[..1], &mut output), (0, 0));
        assert_eq!(decimator.process(&input, &mut []), (0, 0));
        // One output's worth of space takes one pair and leaves the next.
        assert_eq!(decimator.process(&input, &mut output), (2, 1));
        assert_eq!(output[0], expected[0]);
        assert_eq!(decimator.process(&input[2..5], &mut [0; 4]), (2, 1));
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
            let (consumed, produced) = stream(&mut decimator, &input, chunk, 8, 0.0);
            assert_eq!(consumed, 64);
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
    fn borrowed_rings_match_the_array() {
        let input = input();
        let mut expected = [0; 20];
        decimator().process(&input, &mut expected);
        let mut even = FixedCircularBuffer::<i64, 3>::new();
        let mut odd = FixedCircularBuffer::<i64, 6>::new();
        RingBuffer::fill_with(&mut even, || 0);
        RingBuffer::fill_with(&mut odd, || 0);
        let mut output = [0; 20];
        HalfbandDecimatorRefMut::from_parts(config(), &mut *even, &mut *odd)
            .process(&input, &mut output);
        assert_eq!(output, expected);
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
        self::decimator().process(&input, &mut first);
        assert_eq!(first, again);
    }

    #[test]
    #[should_panic(expected = "rings must be full")]
    fn wrong_even_ring_length_panics() {
        // One slot too many would make the oldest even sample a pair too old.
        let _ = HalfbandDecimatorArray::<i64, 4, 6>::with_config(config());
    }
}
