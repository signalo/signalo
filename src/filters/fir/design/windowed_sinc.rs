// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Windowed-sinc FIR tap generation.
//!
//! Frequencies are normalized cycles per sample, so Nyquist is `0.5`.
//! Window-specific helpers, such as [`hann`] and [`kaiser`], generate the
//! window internally while using the same centralized tap generators.

use num_traits::Float;

use crate::filters::util::window as helpers;

use super::{normalize, normalize_by_divisor, Normalization};

macro_rules! windowed_sinc_module {
    ($module:ident, $window:expr, $label:literal) => {
        windowed_sinc_module!($module, $window, $label, {});
    };
    ($module:ident, $window:expr, $label:literal, { $($extra:item)* }) => {
        #[doc = concat!($label, "-windowed sinc tap helpers.")]
        pub mod $module {
            use num_traits::Float;

            /// Fill a slice with low-pass windowed-sinc taps.
            ///
            /// `fc` is normalized cutoff frequency in cycles per sample
            /// (`0 < fc < 0.5`). Taps are normalized to unit DC/passband gain
            /// (`sum(h) == 1`).
            ///
            /// # Panics
            ///
            /// Panics if `weights.len() < 2`, `fc` is outside `(0, 0.5)`, or
            /// passband-gain normalization fails.
            pub fn lowpass<T>(weights: &mut [T], fc: T)
            where
                T: Float + core::fmt::Debug,
            {
                lowpass_unnormalized(weights, fc);
                super::normalize(weights, super::Normalization::PassbandGain);
            }

            pub(crate) fn lowpass_unnormalized<T>(weights: &mut [T], fc: T)
            where
                T: Float + core::fmt::Debug,
            {
                super::lowpass_unnormalized(weights, fc, $window);
            }

            /// Fill a slice with high-pass windowed-sinc taps.
            ///
            /// `fc` is normalized cutoff frequency in cycles per sample
            /// (`0 < fc < 0.5`). Taps are normalized to unit gain at Nyquist.
            ///
            /// # Panics
            ///
            /// Panics if `weights.len() < 3`, `weights.len()` is even, `fc` is
            /// outside `(0, 0.5)`, or gain normalization fails.
            pub fn highpass<T>(weights: &mut [T], fc: T)
            where
                T: Float + core::fmt::Debug,
            {
                highpass_unnormalized(weights, fc);
                super::normalize_highpass(weights);
            }

            pub(crate) fn highpass_unnormalized<T>(weights: &mut [T], fc: T)
            where
                T: Float + core::fmt::Debug,
            {
                super::highpass_unnormalized(weights, fc, $window);
            }

            /// Fill a slice with band-pass windowed-sinc taps.
            ///
            /// `f_lo` and `f_hi` are normalized band edges in cycles per sample.
            /// Taps are normalized to unit gain at the band center.
            ///
            /// # Panics
            ///
            /// Panics if `weights` is empty, `0 < f_lo < f_hi < 0.5` is not
            /// satisfied, or gain normalization fails.
            pub fn bandpass<T>(weights: &mut [T], f_lo: T, f_hi: T)
            where
                T: Float + core::fmt::Debug,
            {
                bandpass_unnormalized(weights, f_lo, f_hi);
                super::normalize_bandpass(weights, f_lo, f_hi);
            }

            pub(crate) fn bandpass_unnormalized<T>(weights: &mut [T], f_lo: T, f_hi: T)
            where
                T: Float + core::fmt::Debug,
            {
                super::bandpass_unnormalized(weights, f_lo, f_hi, $window);
            }

            /// Fill a slice with band-stop windowed-sinc taps.
            ///
            /// `f_lo` and `f_hi` are normalized stop-band edges in cycles per
            /// sample. Taps are normalized to unit DC/passband gain.
            ///
            /// # Panics
            ///
            /// Panics if `weights.len() < 3`, `weights.len()` is even,
            /// `0 < f_lo < f_hi < 0.5` is not satisfied, or passband-gain
            /// normalization fails.
            pub fn bandstop<T>(weights: &mut [T], f_lo: T, f_hi: T)
            where
                T: Float + core::fmt::Debug,
            {
                bandstop_unnormalized(weights, f_lo, f_hi);
                super::normalize(weights, super::Normalization::PassbandGain);
            }

            pub(crate) fn bandstop_unnormalized<T>(weights: &mut [T], f_lo: T, f_hi: T)
            where
                T: Float + core::fmt::Debug,
            {
                super::bandstop_unnormalized(weights, f_lo, f_hi, $window);
            }

            /// Create heap-backed low-pass windowed-sinc taps.
            ///
            /// Taps are normalized to unit DC/passband gain (`sum(h) == 1`).
            #[cfg(feature = "alloc")]
            #[must_use]
            pub fn lowpass_vec<T>(num_taps: usize, fc: T) -> alloc::vec::Vec<T>
            where
                T: Float + core::fmt::Debug,
            {
                let mut weights = alloc::vec![T::zero(); num_taps];
                lowpass(&mut weights, fc);
                weights
            }

            /// Create heap-backed high-pass windowed-sinc taps.
            ///
            /// Taps are normalized to unit gain at Nyquist.
            #[cfg(feature = "alloc")]
            #[must_use]
            pub fn highpass_vec<T>(num_taps: usize, fc: T) -> alloc::vec::Vec<T>
            where
                T: Float + core::fmt::Debug,
            {
                let mut weights = alloc::vec![T::zero(); num_taps];
                highpass(&mut weights, fc);
                weights
            }

            /// Create heap-backed band-pass windowed-sinc taps.
            ///
            /// Taps are normalized to unit gain at the band center.
            #[cfg(feature = "alloc")]
            #[must_use]
            pub fn bandpass_vec<T>(num_taps: usize, f_lo: T, f_hi: T) -> alloc::vec::Vec<T>
            where
                T: Float + core::fmt::Debug,
            {
                let mut weights = alloc::vec![T::zero(); num_taps];
                bandpass(&mut weights, f_lo, f_hi);
                weights
            }

            /// Create heap-backed band-stop windowed-sinc taps.
            ///
            /// Taps are normalized to unit DC/passband gain.
            #[cfg(feature = "alloc")]
            #[must_use]
            pub fn bandstop_vec<T>(num_taps: usize, f_lo: T, f_hi: T) -> alloc::vec::Vec<T>
            where
                T: Float + core::fmt::Debug,
            {
                let mut weights = alloc::vec![T::zero(); num_taps];
                bandstop(&mut weights, f_lo, f_hi);
                weights
            }

            $($extra)*
        }
    };
}

windowed_sinc_module!(rectangular, super::helpers::rectangular::<T>, "Rectangular");
windowed_sinc_module!(triangular, super::helpers::triangular::<T>, "Triangular");
windowed_sinc_module!(hann, super::helpers::hann::<T>, "Hann");
windowed_sinc_module!(hamming, super::helpers::hamming::<T>, "Hamming");
windowed_sinc_module!(blackman, super::helpers::blackman::<T>, "Blackman");
windowed_sinc_module!(
    blackman_harris,
    super::helpers::blackman_harris::<T>,
    "Blackman-Harris"
);
windowed_sinc_module!(flat_top, super::helpers::flat_top::<T>, "Flat-top");
windowed_sinc_module!(kaiser, super::default_kaiser::<T>(), "Kaiser (β=6.0)", {
    fn custom_kaiser<T>(beta: T) -> impl Fn(usize, usize) -> T
    where
        T: Float + core::fmt::Debug,
    {
        assert!(beta >= T::zero(), "Kaiser beta must be non-negative");
        super::helpers::kaiser(beta)
    }

    /// Fill a slice with Kaiser-windowed low-pass sinc taps using custom β.
    ///
    /// `fc` is normalized cutoff frequency in cycles per sample (`0 < fc < 0.5`).
    /// `beta` is the Kaiser shape parameter. Taps are normalized to unit
    /// DC/passband gain (`sum(h) == 1`).
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() < 2`, `fc` is outside `(0, 0.5)`, `beta` is
    /// negative, or passband-gain normalization fails.
    pub fn lowpass_with_beta<T>(weights: &mut [T], beta: T, fc: T)
    where
        T: Float + core::fmt::Debug,
    {
        super::lowpass_unnormalized(weights, fc, custom_kaiser(beta));
        super::normalize(weights, super::Normalization::PassbandGain);
    }

    /// Fill a slice with Kaiser-windowed low-pass sinc taps from Kaiser order parameters.
    ///
    /// `fc` is normalized cutoff frequency in cycles per sample (`0 < fc < 0.5`).
    /// Taps are normalized to unit DC/passband gain (`sum(h) == 1`).
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != order.num_taps`, or if [`lowpass_with_beta`]
    /// panics.
    pub fn lowpass_from_order<T>(weights: &mut [T], order: super::super::KaiserOrder<T>, fc: T)
    where
        T: Float + core::fmt::Debug,
    {
        assert!(
            weights.len() == order.num_taps,
            "Kaiser tap count mismatch: expected {}, got {}",
            order.num_taps,
            weights.len()
        );
        lowpass_with_beta(weights, order.beta, fc);
    }

    /// Fill a slice with Kaiser-windowed low-pass sinc taps from attenuation and transition width.
    ///
    /// `transition_width` is in cycles per sample, where Nyquist is `0.5`. This
    /// uses [`crate::filters::fir::design::kaiser_order`]. The computed tap
    /// count is not forced odd.
    ///
    /// # Panics
    ///
    /// Panics if [`crate::filters::fir::design::kaiser_order`] panics, if
    /// `weights.len() != order.num_taps`, or if [`lowpass_with_beta`] panics.
    pub fn lowpass_for_atten<T>(weights: &mut [T], atten_db: T, transition_width: T, fc: T)
    where
        T: Float + num_traits::ToPrimitive + core::fmt::Debug,
    {
        let order = super::super::kaiser_order(atten_db, transition_width);
        lowpass_from_order(weights, order, fc);
    }

    /// Fill a slice with Kaiser-windowed low-pass sinc taps from `SciPy`-style width.
    ///
    /// `width_nyq` is normalized to Nyquist, where `1.0` corresponds to
    /// π radians/sample. This uses [`crate::filters::fir::design::kaiser_order_nyq`]
    /// and matches `SciPy`'s `scipy.signal.kaiserord` width convention. The
    /// computed tap count is not forced odd.
    ///
    /// # Panics
    ///
    /// Panics if [`crate::filters::fir::design::kaiser_order_nyq`] panics, if
    /// `weights.len() != order.num_taps`, or if [`lowpass_with_beta`] panics.
    pub fn lowpass_for_atten_nyq<T>(weights: &mut [T], atten_db: T, width_nyq: T, fc: T)
    where
        T: Float + num_traits::ToPrimitive + core::fmt::Debug,
    {
        let order = super::super::kaiser_order_nyq(atten_db, width_nyq);
        lowpass_from_order(weights, order, fc);
    }

    /// Fill a slice with Kaiser-windowed high-pass sinc taps using custom β.
    ///
    /// Taps are normalized to unit gain at Nyquist.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() < 3`, `weights.len()` is even, `fc` is outside
    /// `(0, 0.5)`, `beta` is negative, or gain normalization fails.
    pub fn highpass_with_beta<T>(weights: &mut [T], beta: T, fc: T)
    where
        T: Float + core::fmt::Debug,
    {
        super::highpass_unnormalized(weights, fc, custom_kaiser(beta));
        super::normalize_highpass(weights);
    }

    /// Fill a slice with Kaiser-windowed band-pass sinc taps using custom β.
    ///
    /// Taps are normalized to unit gain at the band center.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty, `0 < f_lo < f_hi < 0.5` is not satisfied,
    /// `beta` is negative, or gain normalization fails.
    pub fn bandpass_with_beta<T>(weights: &mut [T], beta: T, f_lo: T, f_hi: T)
    where
        T: Float + core::fmt::Debug,
    {
        super::bandpass_unnormalized(weights, f_lo, f_hi, custom_kaiser(beta));
        super::normalize_bandpass(weights, f_lo, f_hi);
    }

    /// Fill a slice with Kaiser-windowed band-stop sinc taps using custom β.
    ///
    /// Taps are normalized to unit DC/passband gain.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() < 3`, `weights.len()` is even,
    /// `0 < f_lo < f_hi < 0.5` is not satisfied, `beta` is negative, or
    /// passband-gain normalization fails.
    pub fn bandstop_with_beta<T>(weights: &mut [T], beta: T, f_lo: T, f_hi: T)
    where
        T: Float + core::fmt::Debug,
    {
        super::bandstop_unnormalized(weights, f_lo, f_hi, custom_kaiser(beta));
        super::normalize(weights, super::Normalization::PassbandGain);
    }

    /// Create heap-backed Kaiser-windowed low-pass sinc taps.
    ///
    /// Taps are normalized to unit DC/passband gain (`sum(h) == 1`).
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn lowpass_with_beta_vec<T>(num_taps: usize, beta: T, fc: T) -> alloc::vec::Vec<T>
    where
        T: Float + core::fmt::Debug,
    {
        let mut weights = alloc::vec![T::zero(); num_taps];
        lowpass_with_beta(&mut weights, beta, fc);
        weights
    }

    /// Create heap-backed Kaiser-windowed low-pass sinc taps from Kaiser order parameters.
    ///
    /// Taps are normalized to unit DC/passband gain (`sum(h) == 1`).
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn lowpass_from_order_vec<T>(
        order: super::super::KaiserOrder<T>,
        fc: T,
    ) -> alloc::vec::Vec<T>
    where
        T: Float + core::fmt::Debug,
    {
        lowpass_with_beta_vec(order.num_taps, order.beta, fc)
    }

    /// Create heap-backed Kaiser-windowed low-pass sinc taps from attenuation and transition width.
    ///
    /// `transition_width` is in cycles per sample, where Nyquist is `0.5`. This
    /// uses [`crate::filters::fir::design::kaiser_order`]. The computed tap
    /// count is not forced odd.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn lowpass_for_atten_vec<T>(atten_db: T, transition_width: T, fc: T) -> alloc::vec::Vec<T>
    where
        T: Float + num_traits::ToPrimitive + core::fmt::Debug,
    {
        let order = super::super::kaiser_order(atten_db, transition_width);
        lowpass_from_order_vec(order, fc)
    }

    /// Create heap-backed Kaiser-windowed low-pass sinc taps from `SciPy`-style width.
    ///
    /// `width_nyq` is normalized to Nyquist, where `1.0` corresponds to
    /// π radians/sample. This uses [`crate::filters::fir::design::kaiser_order_nyq`]
    /// and matches `SciPy`'s `scipy.signal.kaiserord` width convention. The
    /// computed tap count is not forced odd.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn lowpass_for_atten_nyq_vec<T>(atten_db: T, width_nyq: T, fc: T) -> alloc::vec::Vec<T>
    where
        T: Float + num_traits::ToPrimitive + core::fmt::Debug,
    {
        let order = super::super::kaiser_order_nyq(atten_db, width_nyq);
        lowpass_from_order_vec(order, fc)
    }

    /// Create heap-backed Kaiser-windowed high-pass sinc taps using custom β.
    ///
    /// Taps are normalized to unit gain at Nyquist.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn highpass_with_beta_vec<T>(num_taps: usize, beta: T, fc: T) -> alloc::vec::Vec<T>
    where
        T: Float + core::fmt::Debug,
    {
        let mut weights = alloc::vec![T::zero(); num_taps];
        highpass_with_beta(&mut weights, beta, fc);
        weights
    }

    /// Create heap-backed Kaiser-windowed band-pass sinc taps using custom β.
    ///
    /// Taps are normalized to unit gain at the band center.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn bandpass_with_beta_vec<T>(
        num_taps: usize,
        beta: T,
        f_lo: T,
        f_hi: T,
    ) -> alloc::vec::Vec<T>
    where
        T: Float + core::fmt::Debug,
    {
        let mut weights = alloc::vec![T::zero(); num_taps];
        bandpass_with_beta(&mut weights, beta, f_lo, f_hi);
        weights
    }

    /// Create heap-backed Kaiser-windowed band-stop sinc taps using custom β.
    ///
    /// Taps are normalized to unit DC/passband gain.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn bandstop_with_beta_vec<T>(
        num_taps: usize,
        beta: T,
        f_lo: T,
        f_hi: T,
    ) -> alloc::vec::Vec<T>
    where
        T: Float + core::fmt::Debug,
    {
        let mut weights = alloc::vec![T::zero(); num_taps];
        bandstop_with_beta(&mut weights, beta, f_lo, f_hi);
        weights
    }
});

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
fn default_kaiser<T: Float + core::fmt::Debug>() -> impl Fn(usize, usize) -> T {
    helpers::kaiser(T::from(6.0).unwrap())
}

fn assert_lowpass_args<T>(len: usize, fc: T)
where
    T: Float,
{
    assert!(
        len >= 2,
        "windowed_sinc::lowpass: N=1 is degenerate; require at least 2 taps"
    );
    assert!(
        fc > T::zero() && fc < T::from(0.5).expect("0.5 is representable"),
        "windowed_sinc::lowpass: fc must be in (0, 0.5)"
    );
}

fn assert_highpass_args<T>(len: usize, fc: T)
where
    T: Float,
{
    assert!(len >= 3, "sinc_highpass: N must be >= 3");
    assert!(len % 2 == 1, "sinc_highpass: N must be odd (Type I FIR)");
    assert!(
        fc > T::zero() && fc < T::from(0.5).expect("0.5 is representable"),
        "sinc_highpass: fc must be in (0, 0.5)"
    );
}

fn assert_bandpass_args<T>(len: usize, f_lo: T, f_hi: T)
where
    T: Float,
{
    assert!(len > 0, "sinc_bandpass: N must be > 0");
    assert!(
        f_lo > T::zero() && f_hi > f_lo && f_hi < T::from(0.5).expect("0.5 is representable"),
        "sinc_bandpass: require 0 < f_lo < f_hi < 0.5"
    );
}

fn assert_bandstop_args<T>(len: usize, f_lo: T, f_hi: T)
where
    T: Float,
{
    assert!(len >= 3, "sinc_bandstop: N must be >= 3");
    assert!(len % 2 == 1, "sinc_bandstop: N must be odd (Type I FIR)");
    assert!(
        f_lo > T::zero() && f_hi > f_lo && f_hi < T::from(0.5).expect("0.5 is representable"),
        "sinc_bandstop: require 0 < f_lo < f_hi < 0.5"
    );
}

#[allow(clippy::unwrap_used)]
fn unnormalized_lowpass_tap<T>(index: usize, len: usize, fc: T, window: T) -> T
where
    T: Float,
{
    let two = T::from(2.0).unwrap();
    let pi = T::from(core::f64::consts::PI).unwrap();
    let center = T::from(len - 1).unwrap() / two;
    let offset = T::from(index).unwrap() - center;

    if offset.is_zero() {
        two * fc * window
    } else {
        (two * pi * fc * offset).sin() / (pi * offset) * window
    }
}

fn fill_lowpass_unnormalized<T>(weights: &mut [T], fc: T, window: impl Fn(usize, usize) -> T)
where
    T: Float,
{
    let len = weights.len();
    for (index, weight) in weights.iter_mut().enumerate() {
        *weight = unnormalized_lowpass_tap(index, len, fc, window(index, len));
    }
}

fn lowpass_unnormalized<T>(weights: &mut [T], fc: T, window: impl Fn(usize, usize) -> T)
where
    T: Float + core::fmt::Debug,
{
    assert_lowpass_args(weights.len(), fc);
    fill_lowpass_unnormalized(weights, fc, window);
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
fn highpass_unnormalized<T: Float + core::fmt::Debug>(
    weights: &mut [T],
    fc: T,
    window: impl Fn(usize, usize) -> T,
) {
    let one = T::one();
    let len = weights.len();

    assert_highpass_args(len, fc);
    fill_lowpass_unnormalized(weights, fc, window);

    let m = (len - 1) / 2;
    for (index, weight) in weights.iter_mut().enumerate() {
        if index == m {
            *weight = one - *weight;
        } else {
            *weight = -*weight;
        }
    }
}

fn normalize_highpass<T>(weights: &mut [T])
where
    T: Float + core::fmt::Debug,
{
    let alt_sum =
        weights.iter().enumerate().fold(
            T::zero(),
            |acc, (k, &hk)| {
                if k % 2 == 0 {
                    acc + hk
                } else {
                    acc - hk
                }
            },
        );
    normalize_by_divisor(weights, alt_sum, "sinc_highpass: alt_sum");
}

fn fill_bandpass_unnormalized<T: Float>(
    weights: &mut [T],
    f_lo: T,
    f_hi: T,
    window: impl Fn(usize, usize) -> T,
) {
    let len = weights.len();

    for (index, weight) in weights.iter_mut().enumerate() {
        let window_weight = window(index, len);
        *weight = unnormalized_lowpass_tap(index, len, f_hi, window_weight)
            - unnormalized_lowpass_tap(index, len, f_lo, window_weight);
    }
}

fn bandpass_unnormalized<T: Float + core::fmt::Debug>(
    weights: &mut [T],
    f_lo: T,
    f_hi: T,
    window: impl Fn(usize, usize) -> T,
) {
    assert_bandpass_args(weights.len(), f_lo, f_hi);
    fill_bandpass_unnormalized(weights, f_lo, f_hi, window);
}

#[allow(clippy::unwrap_used)]
fn normalize_bandpass<T>(weights: &mut [T], f_lo: T, f_hi: T)
where
    T: Float + core::fmt::Debug,
{
    let f_c = (f_lo + f_hi) / T::from(2.0).unwrap();
    let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
    let m = T::from(weights.len() - 1).unwrap() / T::from(2.0).unwrap();
    let a_fc = weights.iter().enumerate().fold(T::zero(), |acc, (k, &hk)| {
        acc + hk * (two_pi * f_c * (T::from(k).unwrap() - m)).cos()
    });
    normalize_by_divisor(weights, a_fc, "sinc_bandpass: A(f_c)");
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
fn bandstop_unnormalized<T: Float + core::fmt::Debug>(
    weights: &mut [T],
    f_lo: T,
    f_hi: T,
    window: impl Fn(usize, usize) -> T,
) {
    let one = T::one();
    let len = weights.len();

    assert_bandstop_args(len, f_lo, f_hi);
    fill_bandpass_unnormalized(weights, f_lo, f_hi, window);

    let m = (len - 1) / 2;
    for (index, weight) in weights.iter_mut().enumerate() {
        if index == m {
            *weight = one - *weight;
        } else {
            *weight = -*weight;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn gain_at_freq<T: Float>(h: &[T], f: T) -> T {
    let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
    let mut re = T::zero();
    let mut im = T::zero();
    for (k, &hk) in h.iter().enumerate() {
        let theta = two_pi * f * T::from(k).unwrap();
        re = re + hk * theta.cos();
        im = im - hk * theta.sin();
    }
    (re * re + im * im).sqrt()
}

#[cfg(test)]
mod tests;
