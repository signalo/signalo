// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Windowed-sinc FIR filter design via the Fourier-transform method.
//!
//! This module provides a family of windowed-sinc (also called "window method")
//! linear-phase FIR filter designs. Each design starts from the ideal infinite
//! impulse response of a low-pass filter (a sinc function), truncates it to a
//! finite length `N`, and shapes it with a window to control the Gibbs
//! phenomenon.
//!
//! # When to prefer each window
//!
//! Values from Harris, F. J. (1978). "On the Use of Windows for Harmonic Analysis
//! with the Discrete Fourier Transform." *Proceedings of the IEEE*, 66(1).
//!
//! | Window          | Peak sidelobe | Transition width | Best for |
//! |-----------------|---------------|------------------|----------|
//! | Rectangular     | −13 dB        | Narrowest        | Testing / baseline |
//! | Triangular      | −25 dB        | Narrow           | Simplicity |
//! | Hann            | −31.5 dB      | Moderate         | General-purpose |
//! | Hamming         | −42.5 dB      | Moderate         | Lower sidelobes |
//! | Blackman        | −58 dB        | Wide             | High attenuation |
//! | Blackman-Harris | −92 dB        | Wide             | Very high attenuation |
//! | Flat-top        | ≥ −90 dB      | Wide             | Amplitude accuracy |
//! | Kaiser (β=6.0)  | ~−57 dB       | Tunable          | Flexible trade-off |
//!
//! # Usage
//!
//! Each window family is a newtype wrapper (e.g. [`HannSinc`], [`BlackmanSinc`])
//! implementing [`WindowedSinc`]. Construct a filter and convert to [`Convolve`]:
//!
//! ```rust
//! # #[cfg(any(feature = "libm", feature = "std"))] {
//! use signalo::filters::fir::convolve::Convolve;
//! use signalo::filters::fir::convolve::windowed_sinc::{HannSinc, WindowedSinc};
//! use signalo::traits::Filter;
//!
//! // 1 kHz low-pass at 44.1 kHz sample rate
//! let mut lp = HannSinc::<Convolve<f64, 65>>::lowpass_hz(44_100.0, 1_000.0);
//! let output = lp.filter(1.0);
//! # }
//! ```
//!
//! # Related
//!
//! - [`super::savitzky_golay`] for polynomial smoothing filters
//! - [`super::Convolve::normalized`] for general coefficient normalisation
//!
//! # Coefficient ordering
//!
//! `h[k]` pairs with tap `x[n-k]` (`h[0]` = newest sample).
//! Verified by `convolve::tests::coefficient_ordering`.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

pub(crate) use crate::filters::fir::convolve::Convolve;

#[allow(unused_imports)]
pub(crate) use crate::traits::WithConfig;

#[allow(unused_imports)]
pub(crate) use crate::traits::{ConfigRef, Filter};

#[cfg(any(feature = "libm", feature = "std"))]
use super::Config;

#[cfg(any(feature = "libm", feature = "std"))]
pub(crate) use crate::filters::util::window::*;

pub(crate) mod kernel;

#[cfg(any(feature = "libm", feature = "std"))]
use self::kernel::{default_kaiser, sinc_bandpass, sinc_bandstop, sinc_highpass, sinc_lowpass};

#[cfg(all(any(feature = "libm", feature = "std"), test))]
use self::kernel::gain_at_freq;

// MARK: - WindowedSinc trait

/// Trait for constructing windowed-sinc filters.
///
/// Each method constructs a filter of the given type, normalised to unity
/// gain in the passband.  Unnormalized variants return raw windowed-sinc
/// coefficients.
///
/// # Group delay
///
/// Group delay is `(N−1)/2` samples.  For even `N` this is a half-integer
/// (the filter is implicitly a fractional-delay filter); for integer delay
/// use odd `N`.
pub trait WindowedSinc<T, const N: usize>: Sized {
    /// Lowpass filter with cutoff `fc` (normalised, 0 < fc < 0.5).
    ///
    /// # Panics
    ///
    /// Panics if `N < 2` (N=1 is a degenerate identity filter).
    fn lowpass(fc: T) -> Self;

    /// Highpass filter with cutoff `fc`.  Requires odd N.
    fn highpass(fc: T) -> Self;

    /// Bandpass filter with lower/upper edges `f_lo`, `f_hi`.
    fn bandpass(f_lo: T, f_hi: T) -> Self;

    /// Bandstop filter with lower/upper edges `f_lo`, `f_hi`.  Requires odd N.
    fn bandstop(f_lo: T, f_hi: T) -> Self;

    /// Unnormalized lowpass (raw windowed-sinc coefficients).
    fn lowpass_unnormalized(fc: T) -> Self;

    /// Unnormalized highpass (raw windowed-sinc coefficients).
    fn highpass_unnormalized(fc: T) -> Self;

    /// Unnormalized bandpass (raw windowed-sinc coefficients).
    fn bandpass_unnormalized(f_lo: T, f_hi: T) -> Self;

    /// Unnormalized bandstop (raw windowed-sinc coefficients).
    fn bandstop_unnormalized(f_lo: T, f_hi: T) -> Self;

    /// Lowpass with frequency in Hz (converts to normalised).
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate ≤ 0`, `freq ≤ 0`, `freq ≥ Nyquist`, or `N < 2`.
    fn lowpass_hz(sample_rate: T, freq: T) -> Self;

    /// Highpass with frequency in Hz.
    fn highpass_hz(sample_rate: T, freq: T) -> Self;

    /// Bandpass with frequencies in Hz.
    fn bandpass_hz(sample_rate: T, f_lo: T, f_hi: T) -> Self;

    /// Bandstop with frequencies in Hz.
    fn bandstop_hz(sample_rate: T, f_lo: T, f_hi: T) -> Self;
}

// MARK: - Newtype wrappers

/// Rectangular (Dirichlet) windowed sinc filter.
pub struct RectangularSinc<C>(pub C);

/// Triangular (Bartlett) windowed sinc filter.
pub struct TriangularSinc<C>(pub C);

/// Hann windowed sinc filter.
pub struct HannSinc<C>(pub C);

/// Hamming windowed sinc filter.
pub struct HammingSinc<C>(pub C);

/// Blackman windowed sinc filter.
pub struct BlackmanSinc<C>(pub C);

/// Blackman-Harris windowed sinc filter.
pub struct BlackmanHarrisSinc<C>(pub C);

/// Flat top windowed sinc filter.
pub struct FlatTopSinc<C>(pub C);

/// Kaiser windowed sinc filter (β = 6.0).
///
/// # Behaviour
///
/// All [`WindowedSinc`] trait methods on this type use a fixed β value of 6.0
/// (approximately −57 dB sidelobe attenuation). For custom β, use
/// [`KaiserSinc::lowpass_with_beta`] or [`KaiserSinc::lowpass_with_beta_hz`] instead.
pub struct KaiserSinc<C>(pub C);

// MARK: - Deref / DerefMut / Into

macro_rules! impl_into_inner {
    ($ty:ident) => {
        impl<T, const N: usize> core::ops::Deref for $ty<Convolve<T, N>> {
            type Target = Convolve<T, N>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<T, const N: usize> core::ops::DerefMut for $ty<Convolve<T, N>> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<T, const N: usize> From<$ty<Convolve<T, N>>> for Convolve<T, N> {
            fn from(wrapper: $ty<Convolve<T, N>>) -> Self {
                wrapper.0
            }
        }
    };
}

impl_into_inner!(RectangularSinc);
impl_into_inner!(TriangularSinc);
impl_into_inner!(HannSinc);
impl_into_inner!(HammingSinc);
impl_into_inner!(BlackmanSinc);
impl_into_inner!(BlackmanHarrisSinc);
impl_into_inner!(FlatTopSinc);
impl_into_inner!(KaiserSinc);

// MARK: - KaiserSinc with custom β

#[cfg(any(feature = "libm", feature = "std"))]
impl<T: Float + core::fmt::Debug, const N: usize> KaiserSinc<Convolve<T, N>> {
    /// Create a lowpass Kaiser-windowed sinc filter with custom shape parameter β.
    ///
    /// A larger β produces stronger stopband attenuation and a wider transition band.
    /// Typical values: 5.0–8.0 for general-purpose use (~50–70 dB attenuation).
    /// Use [`Config::beta_for_attenuation`](crate::filters::window::kaiser::Config::beta_for_attenuation)
    /// to compute β from a desired stopband attenuation.
    ///
    /// # Panics
    ///
    /// Panics if `fc` is not in (0, 0.5), `N < 2`, or `beta` is negative.
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn lowpass_with_beta(beta: T, fc: T) -> Self {
        assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
        let coeffs = sinc_lowpass::<T, N>(fc, kaiser(beta), true);
        Self(Convolve::with_config(Config {
            coefficients: coeffs,
        }))
    }

    /// Deprecated: use [`lowpass_with_beta`](Self::lowpass_with_beta) instead.
    #[deprecated(since = "0.3.0", note = "use lowpass_with_beta instead")]
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn with_beta(beta: T, fc: T) -> Self {
        Self::lowpass_with_beta(beta, fc)
    }

    /// Create a highpass Kaiser-windowed sinc filter with custom shape parameter β.
    ///
    /// Requires odd N. See [`WindowedSinc::highpass`] for semantics.
    ///
    /// # Panics
    ///
    /// Panics if `fc` is not in (0, 0.5), N is even, or N is 0.
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn highpass_with_beta(beta: T, fc: T) -> Self {
        assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
        let coeffs = sinc_highpass::<T, N>(fc, kaiser(beta), true);
        Self(Convolve::with_config(Config {
            coefficients: coeffs,
        }))
    }

    /// Create a bandpass Kaiser-windowed sinc filter with custom shape parameter β.
    ///
    /// See [`WindowedSinc::bandpass`] for semantics.
    ///
    /// # Panics
    ///
    /// Panics if `0 < f_lo < f_hi < 0.5` is not satisfied or if N is 0.
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn bandpass_with_beta(beta: T, f_lo: T, f_hi: T) -> Self {
        assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
        let coeffs = sinc_bandpass::<T, N>(f_lo, f_hi, kaiser(beta), true);
        Self(Convolve::with_config(Config {
            coefficients: coeffs,
        }))
    }

    /// Create a bandstop Kaiser-windowed sinc filter with custom shape parameter β.
    ///
    /// Requires odd N. See [`WindowedSinc::bandstop`] for semantics.
    ///
    /// # Panics
    ///
    /// Panics if `0 < f_lo < f_hi < 0.5` is not satisfied, N is even, or N is 0.
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn bandstop_with_beta(beta: T, f_lo: T, f_hi: T) -> Self {
        assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
        let coeffs = sinc_bandstop::<T, N>(f_lo, f_hi, kaiser(beta), true);
        Self(Convolve::with_config(Config {
            coefficients: coeffs,
        }))
    }

    /// Create a lowpass Kaiser-windowed sinc filter with custom β, specifying
    /// frequencies in Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is ≤ 0, `freq` is ≤ 0, `freq` is ≥ Nyquist,
    /// `N < 2`, or `beta` is negative.
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn lowpass_with_beta_hz(beta: T, sample_rate: T, freq: T) -> Self {
        assert!(sample_rate > T::zero(), "sample_rate must be > 0");
        assert!(freq > T::zero(), "frequency must be > 0");
        assert!(
            freq < sample_rate / T::from(2.0).unwrap(),
            "frequency must be < Nyquist"
        );
        Self::lowpass_with_beta(beta, freq / sample_rate)
    }

    /// Deprecated: use [`lowpass_with_beta_hz`](Self::lowpass_with_beta_hz) instead.
    #[deprecated(since = "0.3.0", note = "use lowpass_with_beta_hz instead")]
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn with_beta_hz(beta: T, sample_rate: T, freq: T) -> Self {
        Self::lowpass_with_beta_hz(beta, sample_rate, freq)
    }

    /// Create a highpass Kaiser-windowed sinc filter with custom β, specifying
    /// frequencies in Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is ≤ 0, `freq` is ≤ 0, or `freq` is ≥ Nyquist.
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn highpass_with_beta_hz(beta: T, sample_rate: T, freq: T) -> Self {
        assert!(sample_rate > T::zero(), "sample_rate must be > 0");
        assert!(freq > T::zero(), "frequency must be > 0");
        assert!(
            freq < sample_rate / T::from(2.0).unwrap(),
            "frequency must be < Nyquist"
        );
        Self::highpass_with_beta(beta, freq / sample_rate)
    }

    /// Create a bandpass Kaiser-windowed sinc filter with custom β, specifying
    /// frequencies in Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is ≤ 0, `f_lo` is ≤ 0, `f_lo ≥ f_hi`, or
    /// `f_hi` is ≥ Nyquist.
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn bandpass_with_beta_hz(beta: T, sample_rate: T, f_lo: T, f_hi: T) -> Self {
        assert!(sample_rate > T::zero(), "sample_rate must be > 0");
        assert!(f_lo > T::zero(), "f_lo must be > 0");
        assert!(f_lo < f_hi, "f_lo must be < f_hi");
        assert!(
            f_hi < sample_rate / T::from(2.0).unwrap(),
            "f_hi must be < Nyquist"
        );
        Self::bandpass_with_beta(beta, f_lo / sample_rate, f_hi / sample_rate)
    }

    /// Create a bandstop Kaiser-windowed sinc filter with custom β, specifying
    /// frequencies in Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is ≤ 0, `f_lo` is ≤ 0, `f_lo ≥ f_hi`, or
    /// `f_hi` is ≥ Nyquist.
    #[must_use]
    #[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
    pub fn bandstop_with_beta_hz(beta: T, sample_rate: T, f_lo: T, f_hi: T) -> Self {
        assert!(sample_rate > T::zero(), "sample_rate must be > 0");
        assert!(f_lo > T::zero(), "f_lo must be > 0");
        assert!(f_lo < f_hi, "f_lo must be < f_hi");
        assert!(
            f_hi < sample_rate / T::from(2.0).unwrap(),
            "f_hi must be < Nyquist"
        );
        Self::bandstop_with_beta(beta, f_lo / sample_rate, f_hi / sample_rate)
    }
}

// MARK: - WindowedSinc implementations

macro_rules! impl_windowed_sinc {
    ($ty:ident, $win:expr) => {
        #[cfg(any(feature = "libm", feature = "std"))]
        impl<T: Float + core::fmt::Debug, const N: usize> WindowedSinc<T, N>
            for $ty<Convolve<T, N>>
        {
            fn lowpass(fc: T) -> Self {
                let coeffs = sinc_lowpass::<T, N>(fc, $win, true);
                Self(Convolve::with_config(Config {
                    coefficients: coeffs,
                }))
            }

            fn lowpass_unnormalized(fc: T) -> Self {
                let coeffs = sinc_lowpass::<T, N>(fc, $win, false);
                Self(Convolve::with_config(Config {
                    coefficients: coeffs,
                }))
            }

            fn highpass(fc: T) -> Self {
                let coeffs = sinc_highpass::<T, N>(fc, $win, true);
                Self(Convolve::with_config(Config {
                    coefficients: coeffs,
                }))
            }

            fn highpass_unnormalized(fc: T) -> Self {
                let coeffs = sinc_highpass::<T, N>(fc, $win, false);
                Self(Convolve::with_config(Config {
                    coefficients: coeffs,
                }))
            }

            fn bandpass(f_lo: T, f_hi: T) -> Self {
                let coeffs = sinc_bandpass::<T, N>(f_lo, f_hi, $win, true);
                Self(Convolve::with_config(Config {
                    coefficients: coeffs,
                }))
            }

            fn bandpass_unnormalized(f_lo: T, f_hi: T) -> Self {
                let coeffs = sinc_bandpass::<T, N>(f_lo, f_hi, $win, false);
                Self(Convolve::with_config(Config {
                    coefficients: coeffs,
                }))
            }

            fn bandstop(f_lo: T, f_hi: T) -> Self {
                let coeffs = sinc_bandstop::<T, N>(f_lo, f_hi, $win, true);
                Self(Convolve::with_config(Config {
                    coefficients: coeffs,
                }))
            }

            fn bandstop_unnormalized(f_lo: T, f_hi: T) -> Self {
                let coeffs = sinc_bandstop::<T, N>(f_lo, f_hi, $win, false);
                Self(Convolve::with_config(Config {
                    coefficients: coeffs,
                }))
            }

            fn lowpass_hz(sample_rate: T, freq: T) -> Self {
                assert!(sample_rate > T::zero(), "sample_rate must be > 0");
                assert!(freq > T::zero(), "frequency must be > 0");
                assert!(
                    freq < sample_rate / T::from(2.0).unwrap(),
                    "frequency must be < Nyquist"
                );
                Self::lowpass(freq / sample_rate)
            }

            fn highpass_hz(sample_rate: T, freq: T) -> Self {
                assert!(sample_rate > T::zero(), "sample_rate must be > 0");
                assert!(freq > T::zero(), "frequency must be > 0");
                assert!(
                    freq < sample_rate / T::from(2.0).unwrap(),
                    "frequency must be < Nyquist"
                );
                Self::highpass(freq / sample_rate)
            }

            fn bandpass_hz(sample_rate: T, f_lo: T, f_hi: T) -> Self {
                assert!(sample_rate > T::zero(), "sample_rate must be > 0");
                assert!(f_lo > T::zero(), "f_lo must be > 0");
                assert!(f_lo < f_hi, "f_lo must be < f_hi");
                assert!(
                    f_hi < sample_rate / T::from(2.0).unwrap(),
                    "f_hi must be < Nyquist"
                );
                Self::bandpass(f_lo / sample_rate, f_hi / sample_rate)
            }

            fn bandstop_hz(sample_rate: T, f_lo: T, f_hi: T) -> Self {
                assert!(sample_rate > T::zero(), "sample_rate must be > 0");
                assert!(f_lo > T::zero(), "f_lo must be > 0");
                assert!(f_lo < f_hi, "f_lo must be < f_hi");
                assert!(
                    f_hi < sample_rate / T::from(2.0).unwrap(),
                    "f_hi must be < Nyquist"
                );
                Self::bandstop(f_lo / sample_rate, f_hi / sample_rate)
            }
        }
    };
}

impl_windowed_sinc!(RectangularSinc, |k: usize, n: usize| {
    rectangular::<T>(k, n)
});
impl_windowed_sinc!(TriangularSinc, |k: usize, n: usize| {
    triangular::<T>(k, n)
});
impl_windowed_sinc!(HannSinc, |k: usize, n: usize| { hann::<T>(k, n) });
impl_windowed_sinc!(HammingSinc, |k: usize, n: usize| { hamming::<T>(k, n) });
impl_windowed_sinc!(BlackmanSinc, |k: usize, n: usize| { blackman::<T>(k, n) });
impl_windowed_sinc!(BlackmanHarrisSinc, |k: usize, n: usize| {
    blackman_harris::<T>(k, n)
});
impl_windowed_sinc!(FlatTopSinc, |k: usize, n: usize| { flat_top::<T>(k, n) });
impl_windowed_sinc!(KaiserSinc, |k: usize, n: usize| {
    (default_kaiser::<T>())(k, n)
});

#[cfg(test)]
mod tests;
