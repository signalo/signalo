// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Flat-top window (SRS / Matlab `flattopwin`, 5-term).

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The flat-top window's configuration with precomputed weights.
///
/// Weights use the 5-term SRS / Matlab `flattopwin` ("Bruce Hartmann")
/// coefficients: `a0 = 0.21557895`, `a1 = 0.41663158`,
/// `a2 = 0.277263158`, `a3 = 0.083578947`, `a4 = 0.006947368`.
///
/// This window is optimised for amplitude accuracy, not frequency resolution.
/// The passband is exceptionally flat, making it the preferred choice for
/// calibration and precision amplitude measurements.
///
/// Note: IEC 61000-4-7 (harmonic measurement standard) specifies a
/// rectangular window over an integer number of mains cycles plus Hann
/// smoothing, not this 5-term flat-top filter.
///
/// The storage backend `C` may be any type that implements [`AsSlice<T>`],
/// e.g. `[T; N]` (stack-allocated) or `Vec<T>` (heap-allocated, requires the
/// `alloc` feature). Prefer the [`FlatTopArray`] and [`FlatTopVec`] type
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
    /// Computes the 5-term SRS / Matlab `flattopwin` formula for each tap `k`.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    #[must_use]
    pub fn new() -> Self {
        use crate::filters::util::window::flat_top;
        assert!(N > 0, "FlatTop: window size N must be > 0");
        let mut weights = [T::zero(); N];
        for (k, w) in weights.iter_mut().enumerate() {
            *w = flat_top::<T>(k, N);
        }
        Self { weights }
    }
}

/// The flat-top window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A flat-top window (SRS / Matlab flattopwin, 5-term Bruce Hartmann) generic
/// over its weight storage `C`.
///
/// Each tap coefficient uses the 5-term SRS / Matlab flattopwin (Bruce Hartmann) formula:
/// `w[k] = a0 − a1·cos(α) + a2·cos(2α) − a3·cos(3α) + a4·cos(4α)`
/// with `α = 2πk/(N−1)`.
///
/// This window is optimised for amplitude accuracy — the passband ripple
/// is nearly zero — rather than frequency resolution. It is the standard
/// choice for precision amplitude measurements and calibration. Sidelobe
/// attenuation is approximately −88 dB.
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
/// - [`FlatTopArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`FlatTopVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Examples
///
/// ```rust
/// # #[cfg(any(feature = "libm", feature = "std"))] {
/// use signalo::filters::fir::window::flat_top::{Config, FlatTopArray};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = FlatTopArray::<f32, 4>::with_config(Config::<[f32; 4]>::new());
/// let output = window.filter(1.0);
/// // w[0] = a0 - a1 + a2 - a3 + a4 ≈ -0.00042
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct FlatTop<T, C> {
    config: Config<C>,
    state: State,
    _pd: core::marker::PhantomData<T>,
}

/// A flat-top window backed by a const-generic array of weights.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The weight
/// array lives entirely on the stack.
pub type FlatTopArray<T, const N: usize> = FlatTop<T, [T; N]>;

/// A flat-top window backed by a heap-allocated [`Vec`](alloc::vec::Vec) of weights.
///
/// Requires the `alloc` feature. Use [`FlatTop::from_parts`] to construct
/// this variant, since the size is not known at compile time.
#[cfg(feature = "alloc")]
pub type FlatTopVec<T> = FlatTop<T, alloc::vec::Vec<T>>;

impl<T, C> FlatTop<T, C>
where
    C: AsSlice<T>,
{
    /// Creates a [`FlatTop`] window from a pre-built config.
    ///
    /// Use this constructor when the weight storage is not
    /// `Default`-constructible, e.g. for [`FlatTopVec`] whose size is only
    /// known at runtime.
    ///
    /// # Panics
    ///
    /// Panics if `config.weights.as_slice().len()` is zero.
    pub fn from_parts(config: Config<C>) -> Self {
        let n = config.weights.as_slice().len();
        assert!(n > 0, "FlatTop: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigTrait for FlatTop<T, C> {
    type Config = Config<C>;
}

impl<T, C> StateTrait for FlatTop<T, C> {
    type State = State;
}

impl<T, const N: usize> WithConfig for FlatTopArray<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "FlatTop: window size N must be > 0");
        Self {
            config,
            state: State::default(),
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> ConfigRef for FlatTop<T, C> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C> ConfigClone for FlatTop<T, C>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C> StateMut for FlatTop<T, C> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C> HasGuts for FlatTop<T, C> {
    type Guts = (Config<C>, State);
}

impl<T, C> FromGuts for FlatTop<T, C>
where
    C: AsSlice<T>,
{
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        let n = config.weights.as_slice().len();
        assert!(n > 0, "FlatTop: window size N must be > 0");
        Self {
            config,
            state,
            _pd: core::marker::PhantomData,
        }
    }
}

impl<T, C> IntoGuts for FlatTop<T, C> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for FlatTopArray<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for FlatTopArray<T, N> where Self: Reset {}

impl<T, C> Filter<T> for FlatTop<T, C>
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
