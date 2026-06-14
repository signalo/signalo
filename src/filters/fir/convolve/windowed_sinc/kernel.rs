// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

#[cfg(any(feature = "libm", feature = "std"))]
use super::helpers;

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn default_kaiser_window<T: Float + core::fmt::Debug>() -> impl Fn(usize, usize) -> T {
    helpers::kaiser_window(T::from(6.0).unwrap())
}

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn sinc_lowpass<T: Float + core::fmt::Debug, const N: usize>(
    fc: T,
    window: impl Fn(usize, usize) -> T,
    normalize: bool,
) -> [T; N] {
    let zero = T::zero();
    let two = T::from(2.0).unwrap();
    let pi = T::from(core::f64::consts::PI).unwrap();

    assert!(
        N >= 2,
        "sinc_lowpass: N=1 is degenerate (identity filter); require N >= 2"
    );
    assert!(
        fc > zero && fc < T::from(0.5).unwrap(),
        "sinc_lowpass: fc must be in (0, 0.5)"
    );

    let m_f = T::from(N - 1).unwrap() / two;

    let mut h = [zero; N];
    for (k, hk) in h.iter_mut().enumerate() {
        let w_k = window(k, N);
        let k_f = T::from(k).unwrap();
        let x = pi * two * fc * (k_f - m_f);
        if N % 2 == 1 && k == (N - 1) / 2 {
            *hk = two * fc * w_k;
        } else {
            *hk = two * fc * x.sin() / x * w_k;
        }
    }

    if normalize {
        let sum: T = h.iter().copied().fold(zero, |a, b| a + b);
        let denom = crate::math::safe_normalise_divisor(sum, "sinc_lowpass: coefficient sum");
        for coeff in &mut h {
            *coeff = *coeff / denom;
        }
    }

    h
}

#[cfg(all(any(feature = "libm", feature = "std"), test))]
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

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn sinc_highpass<T: Float + core::fmt::Debug, const N: usize>(
    fc: T,
    window: impl Fn(usize, usize) -> T,
    normalize: bool,
) -> [T; N] {
    let zero = T::zero();
    let one = T::one();

    assert!(N >= 3, "sinc_highpass: N must be >= 3");
    assert!(N % 2 == 1, "sinc_highpass: N must be odd (Type I FIR)");
    assert!(
        fc > zero && fc < T::from(0.5).unwrap(),
        "sinc_highpass: fc must be in (0, 0.5)"
    );

    let h_lp = sinc_lowpass::<T, N>(fc, window, false);

    let m = (N - 1) / 2;
    let mut h = [zero; N];
    for k in 0..N {
        if k == m {
            h[k] = one - h_lp[k];
        } else {
            h[k] = -h_lp[k];
        }
    }

    if normalize {
        let alt_sum: T =
            h.iter().enumerate().fold(
                zero,
                |acc, (k, &hk)| {
                    if k % 2 == 0 {
                        acc + hk
                    } else {
                        acc - hk
                    }
                },
            );
        let denom = crate::math::safe_normalise_divisor(alt_sum, "sinc_highpass: alt_sum");
        for coeff in &mut h {
            *coeff = *coeff / denom;
        }
    }

    h
}

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn sinc_bandpass<T: Float + core::fmt::Debug, const N: usize>(
    f_lo: T,
    f_hi: T,
    window: impl Fn(usize, usize) -> T,
    normalize: bool,
) -> [T; N] {
    let zero = T::zero();

    assert!(
        f_lo > zero && f_hi > f_lo && f_hi < T::from(0.5).unwrap(),
        "sinc_bandpass: require 0 < f_lo < f_hi < 0.5"
    );

    let h_hi = sinc_lowpass::<T, N>(f_hi, &window, false);
    let h_lo = sinc_lowpass::<T, N>(f_lo, &window, false);

    let mut h = [zero; N];
    for (k, hk) in h.iter_mut().enumerate() {
        *hk = h_hi[k] - h_lo[k];
    }

    if normalize {
        let f_c = (f_lo + f_hi) / T::from(2.0).unwrap();
        let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
        let m = T::from(N - 1).unwrap() / T::from(2.0).unwrap();
        let a_fc = h.iter().enumerate().fold(zero, |acc, (k, &hk)| {
            acc + hk * (two_pi * f_c * (T::from(k).unwrap() - m)).cos()
        });
        let denom = crate::math::safe_normalise_divisor(a_fc, "sinc_bandpass: A(f_c)");
        for coeff in &mut h {
            *coeff = *coeff / denom;
        }
    }

    h
}

#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn sinc_bandstop<T: Float + core::fmt::Debug, const N: usize>(
    f_lo: T,
    f_hi: T,
    window: impl Fn(usize, usize) -> T,
    normalize: bool,
) -> [T; N] {
    let zero = T::zero();
    let one = T::one();

    assert!(N >= 3, "sinc_bandstop: N must be >= 3");
    assert!(N % 2 == 1, "sinc_bandstop: N must be odd (Type I FIR)");
    assert!(
        f_lo > zero && f_hi > f_lo && f_hi < T::from(0.5).unwrap(),
        "sinc_bandstop: require 0 < f_lo < f_hi < 0.5"
    );

    let h_bp = sinc_bandpass::<T, N>(f_lo, f_hi, window, false);

    let m = (N - 1) / 2;
    let mut h = [zero; N];
    for k in 0..N {
        if k == m {
            h[k] = one - h_bp[k];
        } else {
            h[k] = -h_bp[k];
        }
    }

    if normalize {
        let sum: T = h.iter().copied().fold(zero, |a, b| a + b);
        let denom = crate::math::safe_normalise_divisor(sum, "sinc_bandstop: coefficient sum");
        for coeff in &mut h {
            *coeff = *coeff / denom;
        }
    }

    h
}
