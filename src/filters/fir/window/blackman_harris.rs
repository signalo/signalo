// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! 4-term Blackman-Harris window.

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The Blackman-Harris window's configuration with precomputed weights.
///
/// Weights use the 4-term coefficients `a0 = 0.35875`, `a1 = 0.48829`,
/// `a2 = 0.14128`, `a3 = 0.01168`.
///
/// The storage backend `C` may be any type that implements [`AsSlice<T>`],
/// e.g. `[T; N]` (stack-allocated) or `Vec<T>` (heap-allocated, requires the
/// `alloc` feature). Prefer the [`BlackmanHarrisArray`] and [`BlackmanHarrisVec`]
/// type aliases over naming this type directly.
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

/// Fill a slice with Blackman-Harris window weights.
#[cfg(any(feature = "libm", feature = "std"))]
pub fn window<T: num_traits::Float>(weights: &mut [T]) {
    super::fill(weights, crate::filters::util::window::blackman_harris);
}

/// Create heap-backed Blackman-Harris window weights.
#[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
#[must_use]
pub fn window_vec<T: num_traits::Float>(num_taps: usize) -> alloc::vec::Vec<T> {
    super::to_vec(num_taps, crate::filters::util::window::blackman_harris)
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
    /// Computes `w[k] = a0 − a1·cos(α) + a2·cos(2α) − a3·cos(3α)` with
    /// `α = 2πk/(N−1)` for each tap `k`.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    #[must_use]
    pub fn new() -> Self {
        assert!(N > 0, "BlackmanHarris: window size N must be > 0");
        let mut weights = [T::zero(); N];
        window(&mut weights);
        Self { weights }
    }
}

/// The Blackman-Harris window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A 4-term Blackman-Harris window generic over its weight storage `C`.
///
/// Each tap coefficient `w[k] = a0 − a1·cos(α) + a2·cos(2α) − a3·cos(3α)`
/// with `α = 2πk/(N−1)`, `a0 = 0.35875`, `a1 = 0.48829`, `a2 = 0.14128`,
/// `a3 = 0.01168`. This is the 4-term Blackman-Harris window. Sidelobe
/// attenuation is approximately −92 dB.
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
/// - [`BlackmanHarrisArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`BlackmanHarrisVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Examples
///
/// ```rust
/// # #[cfg(any(feature = "libm", feature = "std"))] {
/// use signalo::filters::fir::window::blackman_harris::{Config, BlackmanHarrisArray};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = BlackmanHarrisArray::<f32, 4>::with_config(
///     Config::<[f32; 4]>::new(),
/// );
/// let output = window.filter(1.0);
/// // w[0] = a0 - a1 + a2 - a3 ≈ 0.00006
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct BlackmanHarris<T, C> {
    config: Config<C>,
    state: State,
    _pd: core::marker::PhantomData<T>,
}

/// A Blackman-Harris window backed by a const-generic array of weights.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The weight
/// array lives entirely on the stack.
pub type BlackmanHarrisArray<T, const N: usize> = BlackmanHarris<T, [T; N]>;

/// A Blackman-Harris window backed by a heap-allocated [`Vec`](alloc::vec::Vec) of weights.
///
/// Requires the `alloc` feature. Use [`BlackmanHarris::from_parts`] to construct
/// this variant, since the size is not known at compile time.
#[cfg(feature = "alloc")]
pub type BlackmanHarrisVec<T> = BlackmanHarris<T, alloc::vec::Vec<T>>;

/// A Blackman-Harris window that borrows a caller-owned weights slice.
///
/// This alias allows sharing precomputed window weights without taking
/// ownership. Construct via [`BlackmanHarris::from_parts`], passing a
/// `Config { weights: &mut weights_array }`.
pub type BlackmanHarrisRefMut<'a, T> = BlackmanHarris<T, &'a mut [T]>;

impl<T, C> BlackmanHarris<T, C>
where
    C: AsSlice<T>,
{
    /// Creates a [`BlackmanHarris`] window from a pre-built config.
    ///
    /// Use this constructor when the weight storage is not
    /// `Default`-constructible, e.g. for [`BlackmanHarrisVec`] whose size is
    /// only known at runtime.
    ///
    /// # Panics
    ///
    /// Panics if `config.weights.as_slice().len()` is zero.
    pub fn from_parts(config: Config<C>) -> Self {
        let n = config.weights.as_slice().len();
        assert!(n > 0, "BlackmanHarris: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigTrait for BlackmanHarris<T, C> {
    type Config = Config<C>;
}

impl<T, C> StateTrait for BlackmanHarris<T, C> {
    type State = State;
}

impl<T, const N: usize> WithConfig for BlackmanHarrisArray<T, N> {
    type Output = Self;

    /// Creates a [`BlackmanHarrisArray`] from a configuration.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "BlackmanHarris: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigRef for BlackmanHarris<T, C> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C> ConfigClone for BlackmanHarris<T, C>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C> StateMut for BlackmanHarris<T, C> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C> HasGuts for BlackmanHarris<T, C> {
    type Guts = (Config<C>, State);
}

impl<T, C> FromGuts for BlackmanHarris<T, C>
where
    C: AsSlice<T>,
{
    /// Reconstructs a [`BlackmanHarris`] filter from its decomposed state.
    ///
    /// # Panics
    ///
    /// Panics if the weights slice is empty.
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        let n = config.weights.as_slice().len();
        assert!(n > 0, "BlackmanHarris: window size N must be > 0");
        Self {
            config,
            state,
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> IntoGuts for BlackmanHarris<T, C> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for BlackmanHarrisArray<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for BlackmanHarrisArray<T, N> where Self: Reset {}

impl<T, C> Filter<T> for BlackmanHarris<T, C>
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
