// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Convolution filters.

use core::marker::PhantomData;
use core::ops::{Add, Mul};

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::{Num, Zero};

use crate::storage::{zero_filled_fixed_ring, AsSlice, RingBuffer};
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

pub mod differentiator;
pub mod lagrange;
pub mod moving_sum;
pub mod windowed_sinc;

/// The convolution filter's configuration.
///
/// Holds the coefficient storage `C`, which must implement [`AsSlice<K>`]
/// on relevant impls. Use [`ConvolveArray`] for stack-allocated coefficients
/// or [`ConvolveVec`] for heap-allocated ones.
#[derive(Clone, Debug)]
pub struct Config<C> {
    /// The convolution coefficients.
    pub coefficients: C,
}

/// The convolution filter's state.
///
/// Holds the tap ring-buffer `R`, which must implement [`RingBuffer<T>`]
/// on relevant impls.
#[derive(Clone, Debug)]
pub struct State<R> {
    /// The filter's taps (i.e. buffered input).
    pub taps: R,
}

/// A convolution filter generic over sample type `T`, coefficient storage `C`,
/// tap storage `R`, and coefficient type `K`.
///
/// # Coefficient ordering
///
/// Coefficients `h[k]` pair with taps `x[n−k]` so that `h[0]` multiplies the
/// newest sample and `h[N−1]` the oldest. The dot product computes
/// `y[n] = Σ_{k=0}^{N−1} h[k]·x[n−k]` using zero-padding for negative
/// time indices. This convention is verified by the `coefficient_ordering`
/// test.
///
/// # Complexity
///
/// - **Time per sample:** O(N); dot product of N taps with N coefficients.
/// - **Space:** O(N); circular tap buffer of N elements plus N coefficient array.
///
/// # Cold-start behaviour
///
/// On construction, the tap buffer is pre-filled with `N` zeros. The first
/// `N − 1` outputs therefore reflect implicit zero-padding `x[n] = 0` for
/// `n < 0`, as verified by `cold_start_is_zero_padded_partial_convolution`.
/// Discard the warm-up window if zero-padding bias is unacceptable
/// for your application.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`ConvolveArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`ConvolveVec<T>`] — heap-allocated, requires the `alloc` feature.
#[derive(Clone, Debug)]
pub struct Convolve<T, C, R, K = T> {
    config: Config<C>,
    state: State<R>,
    _pd: PhantomData<(T, K)>,
}

/// A convolution filter backed by a const-generic array of coefficients and a
/// [`FixedCircularBuffer`] tap buffer.
///
/// This alias is the `no_std`-friendly, zero-allocation form. Both the
/// coefficient array and the tap ring-buffer live entirely on the stack.
pub type ConvolveArray<T, const N: usize, K = T> =
    Convolve<T, [K; N], FixedCircularBuffer<T, N>, K>;

/// A convolution filter backed by heap-allocated [`Vec`](alloc::vec::Vec) coefficients
/// and a [`HeapCircularBuffer`] tap buffer.
///
/// Requires the `alloc` feature. Use [`Convolve::from_parts`] to construct
/// this variant, since the tap buffer cannot be `Default`-constructed without
/// knowing the desired capacity at compile time.
#[cfg(feature = "alloc")]
pub type ConvolveVec<T, K = T> = Convolve<T, alloc::vec::Vec<K>, HeapCircularBuffer<T>, K>;

/// A convolution filter that borrows a [`CircularBuffer`] tap buffer.
///
/// This alias allows sharing a caller-owned ring buffer without taking
/// ownership of it. The coefficient storage `C` remains generic — use
/// any type implementing [`AsSlice<K>`](crate::storage::AsSlice).
/// Construct via [`Convolve::from_parts`], passing
/// a `&mut CircularBuffer<T>` for the tap buffer.
pub type ConvolveRefMut<'a, T, C, K = T> = Convolve<T, C, &'a mut CircularBuffer<T>, K>;

impl<T, C, R, K> Convolve<T, C, R, K>
where
    C: AsSlice<K>,
    R: RingBuffer<T>,
{
    /// Creates a [`Convolve`] filter from an already-constructed `config` and
    /// `taps` ring-buffer.
    ///
    /// Use this constructor when the tap storage is not `Default`-constructible,
    /// e.g. for [`ConvolveVec`] whose capacity must be known at runtime.
    ///
    /// The `taps` buffer is taken as-is with their current contents. If it contains
    /// non-zero initial content, the first `N − 1` outputs will not reflect implicit
    /// zero-padding (`x[n] = 0` for `n < 0`).
    ///
    /// # Expected storage state
    ///
    /// For the idiomatic zero-padded cold-start, pre-fill the buffer with
    /// `N` zeros before passing.
    ///
    /// # Panics
    ///
    /// Panics if `config.coefficients.as_slice().len()` does not equal
    /// `taps.capacity()`, or if that length is zero.
    pub fn from_parts(config: Config<C>, taps: R) -> Self {
        let n = config.coefficients.as_slice().len();
        assert!(n > 0, "Convolve: window size N must be > 0");
        assert_eq!(
            n,
            taps.capacity(),
            "Convolve: coefficients length ({n}) must equal taps capacity ({})",
            taps.capacity()
        );
        Self {
            config,
            state: State { taps },
            _pd: PhantomData,
        }
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
impl<T, const N: usize> ConvolveArray<T, N>
where
    T: num_traits::Float + core::fmt::Debug,
{
    /// Creates a new [`ConvolveArray`] filter with given `coefficients`,
    /// normalizing them to unity DC gain.
    ///
    /// This constructor is float-only. For integer types, use
    /// [`with_config`](WithConfig::with_config) directly with manually pre-scaled
    /// coefficients.
    ///
    /// # Behaviour
    ///
    /// If `sum == 0` (exact), normalisation is skipped — this is the documented
    /// DC-blocker escape hatch. Otherwise the sum must be finite and its
    /// magnitude must be at or above `T::min_positive_value().sqrt()`; smaller
    /// denominators (near-zero) panic.
    pub fn normalized(mut config: Config<[T; N]>) -> Self {
        let sum = config
            .coefficients
            .as_slice()
            .iter()
            .copied()
            .fold(T::zero(), |a, b| a + b);
        if !sum.is_zero() {
            // Exact zero is treated as an explicit DC-blocker request; near-zero
            // is treated as numerical error and rejected by safe_normalise_divisor.
            let denom =
                crate::math::safe_normalise_divisor(sum, "Convolve::normalized: coefficient sum");
            for coeff in config.coefficients.as_mut_slice() {
                *coeff = *coeff / denom;
            }
        }
        Self::with_config(config)
    }
}

impl<T, C, R, K> ConfigTrait for Convolve<T, C, R, K> {
    type Config = Config<C>;
}

impl<T, C, R, K> StateTrait for Convolve<T, C, R, K> {
    type State = State<R>;
}

impl<T, const N: usize, K> WithConfig for ConvolveArray<T, N, K>
where
    T: Num,
{
    type Output = Self;

    /// Creates a [`ConvolveArray`] from a configuration.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero or if the coefficients length does not equal `N`.
    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Convolve: window size N must be > 0");
        assert_eq!(
            config.coefficients.as_slice().len(),
            N,
            "Convolve: coefficients length must equal N"
        );
        let state = {
            let taps = zero_filled_fixed_ring::<T, N>();
            State { taps }
        };
        Self {
            config,
            state,
            _pd: PhantomData,
        }
    }
}

impl<T, C, R, K> ConfigRef for Convolve<T, C, R, K> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C, R, K> ConfigClone for Convolve<T, C, R, K>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C, R, K> StateMut for Convolve<T, C, R, K> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C, R, K> HasGuts for Convolve<T, C, R, K> {
    type Guts = (Config<C>, State<R>);
}

impl<T, C, R, K> FromGuts for Convolve<T, C, R, K> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self {
            config,
            state,
            _pd: PhantomData,
        }
    }
}

impl<T, C, R, K> IntoGuts for Convolve<T, C, R, K> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize, K> Reset for ConvolveArray<T, N, K>
where
    T: Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize, K> ResetMut for ConvolveArray<T, N, K> where Self: Reset {}

impl<T, C, R, K> Filter<T> for Convolve<T, C, R, K>
where
    T: Clone + Zero + Add<Output = T> + Mul<K, Output = T>,
    K: Clone,
    C: AsSlice<K>,
    R: RingBuffer<T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        self.state.taps.push_back(input);

        let state_iter = self.state.taps.iter();
        // See "Coefficient ordering" in the struct-level documentation.
        // coeff_iter.rev(): state iterates oldest->newest; reversing pairs h[N-1] with oldest, h[0] with newest.
        let coeff_iter = self.config.coefficients.as_slice().iter().rev();

        state_iter
            .zip(coeff_iter)
            .fold(T::zero(), |sum, (state, coeff)| {
                sum + ((*state).clone() * (*coeff).clone())
            })
    }
}

#[cfg(test)]
mod tests;
