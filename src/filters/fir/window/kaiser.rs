// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Kaiser window.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

#[cfg(any(feature = "libm", feature = "std"))]
pub(crate) use crate::math::bessel_i0;

/// The Kaiser window's configuration with precomputed weights and beta parameter.
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
pub struct Config<T, const N: usize> {
    /// Shape parameter β (higher = stronger attenuation, wider main lobe).
    pub beta: T,
    /// Precomputed window weights.
    pub weights: [T; N],
}

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
impl<T: Float + core::fmt::Debug, const N: usize> Config<T, N> {
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
        assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
        assert!(N > 0, "Kaiser: window size N must be > 0");
        let one = T::one();
        let mut weights = [T::zero(); N];
        if N == 1 {
            weights[0] = one;
            return Self { beta, weights };
        }
        let i0_beta = bessel_i0(beta);
        let n_minus_1 = T::from(N - 1).unwrap();
        let two = T::from(2.0).unwrap();
        for (k, weight) in weights.iter_mut().enumerate() {
            let k_f = T::from(k).unwrap();
            let arg = two * k_f / n_minus_1 - one;
            let root = (one - arg * arg).max(T::zero()).sqrt();
            *weight = bessel_i0(beta * root) / i0_beta;
        }
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

/// A Kaiser window.
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
/// use signalo::filters::fir::window::kaiser::{Config as KaiserConfig, Kaiser};
/// use signalo::traits::{Filter, WithConfig};
///
/// let mut window = Kaiser::<f64, 4>::with_config(KaiserConfig::<f64, 4>::new(6.0));
/// let output = window.filter(1.0);
/// assert!(output > 0.0);
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Kaiser<T, const N: usize> {
    config: Config<T, N>,
    state: State,
}

impl<T, const N: usize> ConfigTrait for Kaiser<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for Kaiser<T, N> {
    type State = State;
}

impl<T, const N: usize> WithConfig for Kaiser<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Kaiser: window size N must be > 0");
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, const N: usize> ConfigRef for Kaiser<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Kaiser<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Kaiser<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Kaiser<T, N> {
    type Guts = (Config<T, N>, State);
}

impl<T, const N: usize> FromGuts for Kaiser<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        assert!(N > 0, "Kaiser: window size N must be > 0");
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Kaiser<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Kaiser<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Kaiser<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Kaiser<T, N>
where
    T: Clone + core::ops::Mul<Output = T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let w = self.config.weights[self.state.k].clone();
        self.state.k = (self.state.k + 1) % N;
        input * w
    }
}

#[cfg(test)]
mod tests;
