// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Kaiser window.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The Kaiser window's configuration with precomputed weights and beta parameter.
///
/// `C` is the storage container for the precomputed window weights and must
/// implement [`AsSlice<T>`]. Use [`KaiserArray`] for stack-allocated, const-generic
/// storage or [`KaiserVec`] (with the `alloc` feature) for heap-allocated storage.
///
/// Weights are computed at construction time using the formula
/// `w[k] = I₀(β·√(1 − (2k/(N−1) − 1)²)) / I₀(β)`.
///
/// # Periodicity warning
///
/// Applied periodically: the k-th tap returned is `w[k mod N]`, not tied to
/// input sample index. This means the same coefficient sequence repeats
/// every N calls.
#[derive(Clone, Debug)]
pub struct Config<T, C> {
    /// Shape parameter β (higher = stronger attenuation, wider main lobe).
    pub beta: T,
    /// Precomputed window weights.
    pub weights: C,
}

/// Fill a slice with Kaiser window weights.
///
/// # Panics
///
/// Panics if β is negative or NaN and `weights` is not empty.
#[cfg(any(feature = "libm", feature = "std"))]
pub fn window<T: Float + core::fmt::Debug>(weights: &mut [T], beta: T) {
    if weights.is_empty() {
        return;
    }
    assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
    super::fill(weights, crate::filters::util::window::kaiser(beta));
}

/// Create heap-backed Kaiser window weights.
///
/// # Panics
///
/// Panics if β is negative or NaN and `num_taps` is not zero.
#[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
#[must_use]
pub fn window_vec<T: Float + core::fmt::Debug>(num_taps: usize, beta: T) -> alloc::vec::Vec<T> {
    if num_taps == 0 {
        return alloc::vec::Vec::new();
    }
    assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
    let window = crate::filters::util::window::kaiser(beta);
    super::to_vec(num_taps, window)
}

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
impl<T: Float + core::fmt::Debug, const N: usize> Config<T, [T; N]> {
    /// Create a Kaiser window configuration with the given shape parameter β.
    ///
    /// A larger β produces stronger sidelobe attenuation and a wider main lobe.
    /// Typical values: 5.0–8.0 for general-purpose use (~50–70 dB attenuation).
    ///
    /// When β = 0 the Kaiser window reduces to a rectangular (Dirichlet) window
    /// because `I₀(0) = 1` and all weights equal 1. This is mathematically
    /// consistent but may be surprising; see [`Config::beta_for_attenuation`]
    /// to choose β from a desired stopband attenuation.
    ///
    /// # Panics
    ///
    /// Panics if β is negative or NaN, or if N is 0.
    #[must_use]
    pub fn new(beta: T) -> Self {
        assert!(N > 0, "Kaiser: window size N must be > 0");
        let mut weights = [T::zero(); N];
        window(&mut weights, beta);
        Self { beta, weights }
    }

    /// Compute Kaiser β for a desired sidelobe attenuation in dB.
    ///
    /// Uses the Kaiser empirical formula:
    /// - A ≤ 21 dB → β = 0
    /// - 21 < A < 50 → β = 0.5842·(A−21)^0.4 + 0.07886·(A−21)
    /// - A ≥ 50 → β = 0.1102·(A−8.7)
    ///
    /// The two analytic arms join with a small (~0.02) discontinuity at
    /// `A = 50` dB; this is inherent to Kaiser & Reed's empirical fit and
    /// not a defect.
    #[must_use]
    pub fn beta_for_attenuation(atten_db: T) -> T {
        let zero = T::zero();
        let twenty_one = T::from(21.0).unwrap();
        let fifty = T::from(50.0).unwrap();

        if atten_db <= twenty_one {
            zero
        } else if atten_db < fifty {
            let a_minus_21 = atten_db - twenty_one;
            T::from(0.5842).unwrap() * a_minus_21.powf(T::from(0.4).unwrap())
                + T::from(0.07886).unwrap() * a_minus_21
        } else {
            T::from(0.1102).unwrap() * (atten_db - T::from(8.7).unwrap())
        }
    }
}

/// The Kaiser window's state.
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Current tap index.
    k: usize,
}

/// A Kaiser window backed by a generic flat storage container.
///
/// `T` is the numeric type; `C` is the weights container and must implement
/// [`AsSlice<T>`]. Use the [`KaiserArray`] type alias for fixed-size, stack-allocated
/// storage or [`KaiserVec`] for heap-allocated, runtime-sized storage.
///
/// Each tap coefficient `w[k] = I₀(β·√(1 − (2k/(N−1) − 1)²)) / I₀(β)`,
/// producing a flexible window parametrized by β that smoothly trades off
/// main-lobe width against sidelobe attenuation.
///
/// The Kaiser window is near-optimal in the sense of maximising energy in the
/// main lobe for a given sidelobe level.
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
/// # Examples
///
/// ```rust
/// # #[cfg(any(feature = "libm", feature = "std"))] {
/// use signalo::filters::fir::window::kaiser::{Config as KaiserConfig, KaiserArray};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = KaiserArray::<f64, 4>::with_config(KaiserConfig::<f64, [f64; 4]>::new(6.0));
/// let output = window.filter(1.0);
/// assert!(output > 0.0);
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Kaiser<T, C> {
    config: Config<T, C>,
    state: State,
}

/// A [`Kaiser`] window backed by a fixed-size array `[T; N]`.
///
/// Provides stack-allocated, `no_std`-friendly storage. Use [`KaiserVec`]
/// when the number of taps is only known at runtime.
pub type KaiserArray<T, const N: usize> = Kaiser<T, [T; N]>;

/// A [`Kaiser`] window backed by a heap-allocated `Vec<T>`.
///
/// Requires the `alloc` feature. Use [`KaiserArray`] for `no_std` contexts
/// where the number of taps is known at compile time.
#[cfg(feature = "alloc")]
pub type KaiserVec<T> = Kaiser<T, alloc::vec::Vec<T>>;

/// A Kaiser window that borrows a caller-owned weights slice.
///
/// This alias allows sharing precomputed Kaiser window weights without taking
/// ownership. Construct via [`Kaiser::from_parts`], passing a
/// `Config { beta, weights: &mut weights_array }`.
pub type KaiserRefMut<'a, T> = Kaiser<T, &'a mut [T]>;

impl<T, C> ConfigTrait for Kaiser<T, C> {
    type Config = Config<T, C>;
}

impl<T, C> StateTrait for Kaiser<T, C> {
    type State = State;
}

impl<T, const N: usize> WithConfig for KaiserArray<T, N> {
    type Output = Self;

    /// Creates a [`KaiserArray`] from a configuration.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Kaiser: window size N must be > 0");
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, C> ConfigRef for Kaiser<T, C> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, C> ConfigClone for Kaiser<T, C>
where
    Config<T, C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, C> StateMut for Kaiser<T, C> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C> HasGuts for Kaiser<T, C> {
    type Guts = (Config<T, C>, State);
}

impl<T, C> FromGuts for Kaiser<T, C> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, C> IntoGuts for Kaiser<T, C> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for KaiserArray<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for KaiserArray<T, N> where Self: Reset {}

impl<T, C> Filter<T> for Kaiser<T, C>
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

impl<T, C> Kaiser<T, C>
where
    C: AsSlice<T>,
{
    /// Creates a [`Kaiser`] from a pre-built [`Config`].
    ///
    /// This constructor is intended for storage containers whose size is not
    /// known at compile time (e.g. [`KaiserVec`]). For array-backed storage,
    /// prefer [`WithConfig::with_config`] on [`KaiserArray`].
    ///
    /// # Panics
    ///
    /// Panics if the weights slice is empty.
    pub fn from_parts(config: Config<T, C>) -> Self {
        assert!(
            !config.weights.as_slice().is_empty(),
            "Kaiser: window size N must be > 0"
        );
        Self {
            config,
            state: State::default(),
        }
    }
}

#[cfg(test)]
mod tests;
