// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use num_traits::Float;

pub(crate) use crate::math::bessel_i0;

pub(crate) fn rectangular_window<T: Float>(_: usize, _: usize) -> T {
    T::one()
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn triangular_window<T: Float>(k: usize, n: usize) -> T {
    if n == 1 {
        return T::one();
    }
    let n_f = T::from(n).unwrap();
    let k_f = T::from(k).unwrap();
    let two = T::from(2).unwrap();
    let one = T::one();
    let n_minus_1 = n_f - one;
    one - (two * k_f - n_minus_1).abs() / n_minus_1
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn hann_window<T: Float>(k: usize, n: usize) -> T {
    if n == 1 {
        return T::one();
    }
    let n_f = T::from(n).unwrap();
    let k_f = T::from(k).unwrap();
    let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
    T::from(0.5).unwrap() * (T::one() - (two_pi * k_f / (n_f - T::one())).cos())
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn hamming_window<T: Float>(k: usize, n: usize) -> T {
    if n == 1 {
        return T::one();
    }
    let n_f = T::from(n).unwrap();
    let k_f = T::from(k).unwrap();
    let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
    let theta = two_pi * k_f / (n_f - T::one());
    T::from(0.54).unwrap() - T::from(0.46).unwrap() * theta.cos()
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn blackman_window<T: Float>(k: usize, n: usize) -> T {
    if n == 1 {
        return T::one();
    }
    let n_f = T::from(n).unwrap();
    let k_f = T::from(k).unwrap();
    let one = T::one();
    let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
    let four_pi = T::from(4.0 * core::f64::consts::PI).unwrap();
    let theta0 = two_pi * k_f / (n_f - one);
    let theta1 = four_pi * k_f / (n_f - one);
    T::from(0.42).unwrap() - T::from(0.5).unwrap() * theta0.cos()
        + T::from(0.08).unwrap() * theta1.cos()
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn blackman_harris_window<T: Float>(k: usize, n: usize) -> T {
    if n == 1 {
        return T::one();
    }
    let n_f = T::from(n).unwrap();
    let k_f = T::from(k).unwrap();
    let one = T::one();
    let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
    let four_pi = T::from(4.0 * core::f64::consts::PI).unwrap();
    let six_pi = T::from(6.0 * core::f64::consts::PI).unwrap();
    let theta0 = two_pi * k_f / (n_f - one);
    let theta1 = four_pi * k_f / (n_f - one);
    let theta2 = six_pi * k_f / (n_f - one);
    T::from(0.35875).unwrap() - T::from(0.48829).unwrap() * theta0.cos()
        + T::from(0.14128).unwrap() * theta1.cos()
        - T::from(0.01168).unwrap() * theta2.cos()
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn flat_top_window<T: Float>(k: usize, n: usize) -> T {
    if n == 1 {
        return T::one();
    }
    let n_f = T::from(n).unwrap();
    let k_f = T::from(k).unwrap();
    let one = T::one();
    let two_pi = T::from(2.0 * core::f64::consts::PI).unwrap();
    let four_pi = T::from(4.0 * core::f64::consts::PI).unwrap();
    let six_pi = T::from(6.0 * core::f64::consts::PI).unwrap();
    let eight_pi = T::from(8.0 * core::f64::consts::PI).unwrap();
    let theta0 = two_pi * k_f / (n_f - one);
    let theta1 = four_pi * k_f / (n_f - one);
    let theta2 = six_pi * k_f / (n_f - one);
    let theta3 = eight_pi * k_f / (n_f - one);
    T::from(0.215_578_95).unwrap() - T::from(0.416_631_58).unwrap() * theta0.cos()
        + T::from(0.277_263_158).unwrap() * theta1.cos()
        - T::from(0.083_578_947).unwrap() * theta2.cos()
        + T::from(0.006_947_368).unwrap() * theta3.cos()
}

#[allow(clippy::unwrap_used, clippy::missing_panics_doc)]
pub(crate) fn kaiser_window<T: Float + core::fmt::Debug>(
    beta: T,
) -> impl Fn(usize, usize) -> T {
    let i0_beta = bessel_i0(beta);
    let two = T::from(2.0).unwrap();
    let one = T::one();
    move |k, n| {
        if n == 1 {
            return T::one();
        }
        let n_f = T::from(n).unwrap();
        let k_f = T::from(k).unwrap();
        let arg = two * k_f / (n_f - one) - one;
        let val = (one - arg * arg).max(T::zero()).sqrt();
        bessel_i0(beta * val) / i0_beta
    }
}
