// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Classical (truncated 3-term) Blackman window.

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The Blackman window's configuration with precomputed weights.
///
/// Weights use the classical (truncated) 3-term Blackman form rather than
/// the "exact Blackman" form. The exact Blackman adjusts the coefficients
/// to force the first and last samples to zero (`a0 = 7938/18608 ≈ 0.42659`);
/// the classical form keeps the simpler `a0 = 0.42` at the cost of a slight
/// discontinuity at the endpoints.
///
/// The storage backend `C` may be any type that implements [`AsSlice<T>`],
/// e.g. `[T; N]` (stack-allocated) or `Vec<T>` (heap-allocated, requires the
/// `alloc` feature). Prefer the [`BlackmanArray`] and [`BlackmanVec`] type
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
    /// Computes `w[k] = 0.42 − 0.5·cos(α) + 0.08·cos(2α)` with
    /// `α = 2πk/(N−1)` for each tap `k`.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    #[must_use]
    pub fn new() -> Self {
        use crate::filters::util::window::blackman;
        assert!(N > 0, "Blackman: window size N must be > 0");
        let mut weights = [T::zero(); N];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = blackman::<T>(k, N);
        }
        Self { weights }
    }
}

/// The Blackman window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A classical (truncated 3-term) Blackman window generic over its weight
/// storage `C`.
///
/// Each tap coefficient `w[k] = 0.42 − 0.5·cos(α) + 0.08·cos(2α)` with
/// `α = 2πk/(N−1)`. This is the classical three-term Blackman window
/// (not the "exact Blackman" variant). Sidelobe attenuation is approximately
/// −58 dB.
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
/// - [`BlackmanArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`BlackmanVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Examples
///
/// ```rust
/// # #[cfg(any(feature = "libm", feature = "std"))] {
/// use signalo::filters::fir::window::blackman::{Config, BlackmanArray};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = BlackmanArray::<f32, 4>::with_config(Config::<[f32; 4]>::new());
/// let output = window.filter(1.0);
/// assert!((output - 0.0).abs() < 1e-5); // w[0] = 0
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Blackman<T, C> {
    config: Config<C>,
    state: State,
    _pd: core::marker::PhantomData<T>,
}

/// A Blackman window backed by a const-generic array of weights.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The weight
/// array lives entirely on the stack.
pub type BlackmanArray<T, const N: usize> = Blackman<T, [T; N]>;

/// A Blackman window backed by a heap-allocated [`Vec`](alloc::vec::Vec) of weights.
///
/// Requires the `alloc` feature. Use [`Blackman::from_parts`] to construct
/// this variant, since the size is not known at compile time.
#[cfg(feature = "alloc")]
pub type BlackmanVec<T> = Blackman<T, alloc::vec::Vec<T>>;

/// A Blackman window that borrows a caller-owned weights slice.
///
/// This alias allows sharing precomputed window weights without taking
/// ownership. Construct via [`Blackman::from_parts`], passing a
/// `Config { weights: &mut weights_array }`.
pub type BlackmanRefMut<'a, T> = Blackman<T, &'a mut [T]>;

impl<T, C> Blackman<T, C>
where
    C: AsSlice<T>,
{
    /// Creates a [`Blackman`] window from a pre-built config.
    ///
    /// Use this constructor when the weight storage is not
    /// `Default`-constructible, e.g. for [`BlackmanVec`] whose size is only
    /// known at runtime.
    ///
    /// # Panics
    ///
    /// Panics if `config.weights.as_slice().len()` is zero.
    pub fn from_parts(config: Config<C>) -> Self {
        let n = config.weights.as_slice().len();
        assert!(n > 0, "Blackman: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigTrait for Blackman<T, C> {
    type Config = Config<C>;
}

impl<T, C> StateTrait for Blackman<T, C> {
    type State = State;
}

impl<T, const N: usize> WithConfig for BlackmanArray<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Blackman: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigRef for Blackman<T, C> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C> ConfigClone for Blackman<T, C>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C> StateMut for Blackman<T, C> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C> HasGuts for Blackman<T, C> {
    type Guts = (Config<C>, State);
}

impl<T, C> FromGuts for Blackman<T, C>
where
    C: AsSlice<T>,
{
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        let n = config.weights.as_slice().len();
        assert!(n > 0, "Blackman: window size N must be > 0");
        Self {
            config,
            state,
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> IntoGuts for Blackman<T, C> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for BlackmanArray<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for BlackmanArray<T, N> where Self: Reset {}

impl<T, C> Filter<T> for Blackman<T, C>
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
