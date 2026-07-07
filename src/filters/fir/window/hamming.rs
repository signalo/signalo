// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Hamming window.

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The Hamming window's configuration with precomputed weights.
///
/// Weights use the original 0.54 / 0.46 coefficients (as opposed to the
/// optimised 0.53836 / 0.46164 variant). Optimised coefficients improve
/// scalloping loss by moving the first null; the original coefficients are
/// used here to match the classic definition.
///
/// The storage backend `C` may be any type that implements [`AsSlice<T>`],
/// e.g. `[T; N]` (stack-allocated) or `Vec<T>` (heap-allocated, requires the
/// `alloc` feature). Prefer the [`HammingArray`] and [`HammingVec`] type
/// aliases over naming this type directly.
///
/// # Periodicity warning
///
/// Applied periodically: the k-th tap returned is `w[k mod N]`, not tied to
/// input sample index. This means the same coefficient sequence repeats
/// every N calls.
#[derive(Clone, Debug)]
pub struct Config<C> {
    /// Precomputed window weights.
    pub weights: C,
}

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(
    clippy::new_without_default,
    clippy::unwrap_used,
    clippy::missing_panics_doc
)]
impl<T: num_traits::Float, const N: usize> Config<[T; N]> {
    /// Create a window configuration with precomputed weights.
    ///
    /// Computes `w[k] = 0.54 − 0.46 · cos(2πk / (N−1))` for each tap `k`.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    #[must_use]
    pub fn new() -> Self {
        use crate::filters::util::window::hamming;
        assert!(N > 0, "Hamming: window size N must be > 0");
        let mut weights = [T::zero(); N];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = hamming::<T>(k, N);
        }
        Self { weights }
    }
}

/// The Hamming window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A Hamming window generic over its weight storage `C`.
///
/// Each tap coefficient `w[k] = 0.54 − 0.46 · cos(2πk / (N−1))`, using the
/// original coefficients from R. W. Hamming's 1977 *Digital Filters*.
/// Sidelobe attenuation is approximately −42.5 dB.
///
/// # Periodicity
///
/// Applied periodically: the k-th tap returned is w[k mod N], not tied to input
/// sample index. This means the same coefficient sequence repeats every N calls.
///
/// # Complexity
///
/// - **Time per sample:** O(1)
/// - **Space:** O(N · `sizeof::<T>()`)
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`HammingArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`HammingVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Examples
///
/// ```rust
/// # #[cfg(any(feature = "libm", feature = "std"))] {
/// use signalo::filters::fir::window::hamming::{Config, HammingArray};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = HammingArray::<f32, 4>::with_config(Config::<[f32; 4]>::new());
/// let output = window.filter(1.0);
/// assert!((output - 0.08).abs() < 1e-5); // w[0] = 0.54 - 0.46 = 0.08
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Hamming<T, C> {
    config: Config<C>,
    state: State,
    _pd: core::marker::PhantomData<T>,
}

/// A Hamming window backed by a const-generic array of weights.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The weight
/// array lives entirely on the stack.
pub type HammingArray<T, const N: usize> = Hamming<T, [T; N]>;

/// A Hamming window backed by a heap-allocated [`Vec`](alloc::vec::Vec) of weights.
///
/// Requires the `alloc` feature. Use [`Hamming::from_parts`] to construct
/// this variant, since the size is not known at compile time.
#[cfg(feature = "alloc")]
pub type HammingVec<T> = Hamming<T, alloc::vec::Vec<T>>;

/// A Hamming window that borrows a caller-owned weights slice.
///
/// This alias allows sharing precomputed window weights without taking
/// ownership. Construct via [`Hamming::from_parts`], passing a
/// `Config { weights: &mut weights_array }`.
pub type HammingRefMut<'a, T> = Hamming<T, &'a mut [T]>;

impl<T, C> Hamming<T, C>
where
    C: AsSlice<T>,
{
    /// Creates a [`Hamming`] window from a pre-built config.
    ///
    /// Use this constructor when the weight storage is not
    /// `Default`-constructible, e.g. for [`HammingVec`] whose size is only
    /// known at runtime.
    ///
    /// # Panics
    ///
    /// Panics if `config.weights.as_slice().len()` is zero.
    pub fn from_parts(config: Config<C>) -> Self {
        let n = config.weights.as_slice().len();
        assert!(n > 0, "Hamming: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigTrait for Hamming<T, C> {
    type Config = Config<C>;
}

impl<T, C> StateTrait for Hamming<T, C> {
    type State = State;
}

impl<T, const N: usize> WithConfig for HammingArray<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Hamming: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigRef for Hamming<T, C> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C> ConfigClone for Hamming<T, C>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C> StateMut for Hamming<T, C> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C> HasGuts for Hamming<T, C> {
    type Guts = (Config<C>, State);
}

impl<T, C> FromGuts for Hamming<T, C>
where
    C: AsSlice<T>,
{
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        let n = config.weights.as_slice().len();
        assert!(n > 0, "Hamming: window size N must be > 0");
        Self {
            config,
            state,
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> IntoGuts for Hamming<T, C> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for HammingArray<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for HammingArray<T, N> where Self: Reset {}

impl<T, C> Filter<T> for Hamming<T, C>
where
    T: Clone + core::ops::Mul<Output = T>,
    C: AsSlice<T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let weights = self.config.weights.as_slice();
        let n = weights.len();
        let w = weights[self.state.k].clone();
        self.state.k = (self.state.k + 1) % n;
        input * w
    }
}

#[cfg(test)]
mod tests;
