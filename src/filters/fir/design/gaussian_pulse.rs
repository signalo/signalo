// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Gaussian frequency pulse taps for GMSK.

use num_traits::Float;

use super::{len_from_span, normalize, Normalization};
use crate::math::{erf, Erf};

/// Fill a slice with GMSK Gaussian frequency pulse taps and explicit normalization.
///
/// This is the BT-parameterized integrated Gaussian pulse used by GMSK
/// modulators, not a Gaussian window. `span` is the full pulse length in
/// symbols, not the per-side delay. The tap count must be `span * sps + 1`,
/// and `weights.len()` must match that value.
///
/// `sps` is the number of samples per symbol. `bt` is the bandwidth-time
/// product. `norm` selects the final tap normalization.
/// Typical GMSK deployments use `bt` in the range `[0.3, 0.5]`; values below
/// roughly `0.2` require a correspondingly larger `span` to avoid significant
/// truncation error.
///
/// # Panics
///
/// Panics if `bt <= 0`, `span == 0`, `sps < 2`, `span * sps` is odd,
/// `span * sps + 1` overflows, or the tap count is not `span * sps + 1`.
pub fn taps_with_norm<T>(weights: &mut [T], span: usize, sps: usize, bt: T, norm: Normalization)
where
    T: Float + Erf + core::fmt::Debug,
{
    let len = len_from_span(span, sps);
    assert_eq!(
        weights.len(),
        len,
        "gaussian_pulse: tap count must equal span * sps + 1"
    );
    let samples = len - 1;

    assert!(bt > T::zero(), "gaussian_pulse: BT must be > 0");

    let pi = T::from(core::f64::consts::PI).expect("π is representable");
    let sqrt_2 = T::from(core::f64::consts::SQRT_2).expect("sqrt(2) is representable");
    let sps_t = T::from(sps).expect("sps is representable");
    let two = T::from(2.0).expect("2 is representable");
    let half_scalar = T::from(0.5).expect("0.5 is representable");
    let sigma = T::ln(two).sqrt() / (two * pi * bt);
    let inv = T::one() / (sqrt_2 * sigma * sps_t);
    let half = sps_t * half_scalar;
    let Ok(center) = isize::try_from(samples / 2) else {
        panic!("gaussian_pulse: tap count must fit in isize");
    };

    for (index, weight) in weights.iter_mut().enumerate() {
        let Ok(index) = isize::try_from(index) else {
            panic!("gaussian_pulse: tap index must fit in isize");
        };
        let n = index - center;
        let t = T::from(n).expect("tap index is representable");
        *weight = half_scalar * (erf((t + half) * inv) - erf((t - half) * inv));
    }

    normalize(weights, norm);
}

/// Fill a slice with GMSK Gaussian frequency pulse taps.
///
/// Taps are passband-gain normalized. See [`taps_with_norm`] for the details.
pub fn taps<T>(weights: &mut [T], span: usize, sps: usize, bt: T)
where
    T: Float + Erf + core::fmt::Debug,
{
    taps_with_norm(weights, span, sps, bt, Normalization::PassbandGain);
}

/// Create heap-backed GMSK Gaussian frequency pulse taps with explicit normalization.
///
/// `span` is the full pulse length in symbols, not the per-side delay. The
/// returned vector has `span * sps + 1` taps. `span * sps` must be even so the
/// result has odd length and is centered on one tap.
///
/// This is the allocating equivalent of [`taps_with_norm`]. `sps` is the number
/// of samples per symbol, `bt` is the bandwidth-time product, and `norm` selects
/// the final tap normalization. See [`taps_with_norm`] for practical `bt`
/// range guidance.
///
/// # Panics
///
/// Panics if `span == 0`, `sps < 2`, `span * sps` is odd, `span * sps + 1`
/// overflows, or `bt <= 0`.
#[cfg(feature = "alloc")]
#[must_use]
pub fn taps_with_norm_vec<T>(
    span: usize,
    sps: usize,
    bt: T,
    norm: Normalization,
) -> alloc::vec::Vec<T>
where
    T: Float + Erf + core::fmt::Debug,
{
    let len = super::len_from_span(span, sps);
    let mut weights = alloc::vec![T::zero(); len];
    taps_with_norm(&mut weights, span, sps, bt, norm);
    weights
}

/// Create heap-backed GMSK Gaussian frequency pulse taps.
///
/// Taps are passband-gain normalized. See [`taps_with_norm_vec`] for the details.
#[cfg(feature = "alloc")]
#[must_use]
pub fn taps_vec<T>(span: usize, sps: usize, bt: T) -> alloc::vec::Vec<T>
where
    T: Float + Erf + core::fmt::Debug,
{
    taps_with_norm_vec(span, sps, bt, Normalization::PassbandGain)
}

#[cfg(test)]
mod tests;
