// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Hann window.

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The Hann window's configuration with precomputed weights.
///
/// Weights are computed at construction time using the formula
/// `w[k] = 0.5 · (1 − cos(2πk / (N−1)))`.
///
/// The storage backend `C` may be any type that implements [`AsSlice<T>`],
/// e.g. `[T; N]` (stack-allocated) or `Vec<T>` (heap-allocated, requires the
/// `alloc` feature). Prefer the [`HannArray`] and [`HannVec`] type aliases
/// over naming this type directly.
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

/// Fill a slice with Hann window weights.
#[cfg(any(feature = "libm", feature = "std"))]
pub fn window<T: num_traits::Float>(weights: &mut [T]) {
    super::fill(weights, crate::filters::util::window::hann);
}

/// Create heap-backed Hann window weights.
#[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
#[must_use]
pub fn window_vec<T: num_traits::Float>(num_taps: usize) -> alloc::vec::Vec<T> {
    super::to_vec(num_taps, crate::filters::util::window::hann)
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
    /// Computes `w[k] = 0.5 · (1 − cos(2πk / (N−1)))` for each tap `k`.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    #[must_use]
    pub fn new() -> Self {
        assert!(N > 0, "Hann: window size N must be > 0");
        let mut weights = [T::zero(); N];
        window(&mut weights);
        Self { weights }
    }
}

/// The Hann window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A Hann window generic over its weight storage `C`.
///
/// Each tap coefficient `w[k] = 0.5 · (1 − cos(2πk / (N−1)))`, producing a
/// smooth cosine-tapered sequence with zero at both endpoints. Sidelobe
/// attenuation is approximately −31.5 dB.
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
/// - [`HannArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`HannVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Examples
///
/// ```rust
/// # #[cfg(any(feature = "libm", feature = "std"))] {
/// use signalo::filters::fir::window::hann::{Config, HannArray};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = HannArray::<f32, 4>::with_config(Config::<[f32; 4]>::new());
/// let output = window.filter(1.0);
/// assert!((output - 0.0).abs() < 1e-5); // w[0] = 0
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Hann<T, C> {
    config: Config<C>,
    state: State,
    _pd: core::marker::PhantomData<T>,
}

/// A Hann window backed by a const-generic array of weights.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The weight
/// array lives entirely on the stack.
pub type HannArray<T, const N: usize> = Hann<T, [T; N]>;

/// A Hann window backed by a heap-allocated [`Vec`](alloc::vec::Vec) of weights.
///
/// Requires the `alloc` feature. Use [`Hann::from_parts`] to construct
/// this variant, since the size is not known at compile time.
#[cfg(feature = "alloc")]
pub type HannVec<T> = Hann<T, alloc::vec::Vec<T>>;

/// A Hann window that borrows a caller-owned weights slice.
///
/// This alias allows sharing precomputed window weights without taking
/// ownership. Construct via [`Hann::from_parts`], passing a
/// `Config { weights: &mut weights_array }`.
pub type HannRefMut<'a, T> = Hann<T, &'a mut [T]>;

impl<T, C> Hann<T, C>
where
    C: AsSlice<T>,
{
    /// Creates a [`Hann`] window from a pre-built config.
    ///
    /// Use this constructor when the weight storage is not
    /// `Default`-constructible, e.g. for [`HannVec`] whose size is only
    /// known at runtime.
    ///
    /// # Panics
    ///
    /// Panics if `config.weights.as_slice().len()` is zero.
    pub fn from_parts(config: Config<C>) -> Self {
        let n = config.weights.as_slice().len();
        assert!(n > 0, "Hann: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigTrait for Hann<T, C> {
    type Config = Config<C>;
}

impl<T, C> StateTrait for Hann<T, C> {
    type State = State;
}

impl<T, const N: usize> WithConfig for HannArray<T, N> {
    type Output = Self;

    /// Creates a [`HannArray`] from a configuration.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Hann: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigRef for Hann<T, C> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C> ConfigClone for Hann<T, C>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C> StateMut for Hann<T, C> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C> HasGuts for Hann<T, C> {
    type Guts = (Config<C>, State);
}

impl<T, C> FromGuts for Hann<T, C>
where
    C: AsSlice<T>,
{
    /// Reconstructs a [`Hann`] filter from its decomposed state.
    ///
    /// # Panics
    ///
    /// Panics if the weights slice is empty.
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        let n = config.weights.as_slice().len();
        assert!(n > 0, "Hann: window size N must be > 0");
        Self {
            config,
            state,
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> IntoGuts for Hann<T, C> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for HannArray<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for HannArray<T, N> where Self: Reset {}

impl<T, C> Filter<T> for Hann<T, C>
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
