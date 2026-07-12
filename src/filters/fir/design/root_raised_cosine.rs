// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Square-root raised-cosine pulse-shaping taps.

use num_traits::Float;

use super::{len_from_span, normalize, Normalization};

/// Fill a slice with square-root raised-cosine taps and explicit normalization.
///
/// `span` is the full pulse length in symbols, not the per-side delay. The tap
/// count must be `span * sps + 1`, and `weights.len()` must match that value.
///
/// `sps` is the number of samples per symbol. `rolloff` is the excess bandwidth
/// factor in `[0, 1]`. `norm` selects the final tap normalization.
///
/// # Panics
///
/// Panics if `span == 0`, `sps < 2`, `span * sps` is odd, `span * sps + 1`
/// overflows, the tap count is not `span * sps + 1`, or `rolloff` is outside
/// `[0, 1]`.
pub fn taps_with_norm<T>(
    weights: &mut [T],
    span: usize,
    sps: usize,
    rolloff: T,
    norm: Normalization,
) where
    T: Float + core::fmt::Debug,
{
    let len = len_from_span(span, sps);
    assert_eq!(
        weights.len(),
        len,
        "root_raised_cosine: tap count must equal span * sps + 1"
    );
    let samples = len - 1;

    assert!(
        rolloff >= T::zero() && rolloff <= T::one(),
        "root_raised_cosine: rolloff must be in [0, 1]"
    );

    let pi = T::from(core::f64::consts::PI).expect("π is representable");
    let sps_t = T::from(sps).expect("sps is representable");
    let sqrt_2 = T::from(core::f64::consts::SQRT_2).expect("sqrt(2) is representable");
    let two = T::from(2.0).expect("2 is representable");
    let four = T::from(4.0).expect("4 is representable");
    let inv_sps = T::one() / sps_t;
    let sqrt_sps = sps_t.sqrt();
    let Ok(half) = isize::try_from(samples / 2) else {
        panic!("root_raised_cosine: tap count must fit in isize");
    };

    for (index, weight) in weights.iter_mut().enumerate() {
        let Ok(index) = isize::try_from(index) else {
            panic!("root_raised_cosine: tap index must fit in isize");
        };
        let offset = index - half;
        let offset_t = T::from(offset).expect("tap index is representable");
        let symbol_time = offset_t * inv_sps;
        let four_rolloff_time = four * rolloff * symbol_time;

        let tap = if symbol_time == T::zero() {
            (T::one() - rolloff + four * rolloff / pi) * sqrt_sps
        } else {
            let quad_denominator = T::one() - four_rolloff_time * four_rolloff_time;
            if rolloff > T::zero() && quad_denominator.abs() <= T::epsilon() * four {
                let s = (pi / (four * rolloff)).sin();
                let c = (pi / (four * rolloff)).cos();
                let g1 = T::one() + two / pi;
                let g3 = T::one() - two / pi;
                (rolloff / sqrt_2) * (g1 * s + g3 * c) * sqrt_sps
            } else {
                let low_side = (pi * (T::one() - rolloff) * symbol_time).sin();
                let high_side = four_rolloff_time * (pi * (T::one() + rolloff) * symbol_time).cos();
                let denominator = (pi * symbol_time) * quad_denominator;
                ((low_side + high_side) / denominator) * sqrt_sps
            }
        };

        *weight = tap;
    }

    normalize(weights, norm);
}

/// Fill a slice with square-root raised-cosine pulse-shaping taps.
///
/// Taps are unit-energy normalized. See [`taps_with_norm`] for the details.
pub fn taps<T>(weights: &mut [T], span: usize, sps: usize, rolloff: T)
where
    T: Float + core::fmt::Debug,
{
    taps_with_norm(weights, span, sps, rolloff, Normalization::UnitEnergy);
}

/// Create heap-backed square-root raised-cosine taps with explicit normalization.
///
/// `span` is the full pulse length in symbols, not the per-side delay. The
/// returned vector has `span * sps + 1` taps. `span * sps` must be even so the
/// result has odd length and is centered on one tap.
///
/// This is the allocating equivalent of [`taps_with_norm`]. `sps` is the number
/// samples per symbol, `rolloff` is the excess bandwidth factor in `[0, 1]`,
/// and `norm` selects the final tap normalization.
///
/// # Panics
///
/// Panics if `span == 0`, `sps < 2`, `span * sps` is odd, `span * sps + 1`
/// overflows, or `rolloff` is outside `[0, 1]`.
#[cfg(feature = "alloc")]
#[must_use]
pub fn taps_with_norm_vec<T>(
    span: usize,
    sps: usize,
    rolloff: T,
    norm: Normalization,
) -> alloc::vec::Vec<T>
where
    T: Float + core::fmt::Debug,
{
    let len = super::len_from_span(span, sps);
    let mut weights = alloc::vec![T::zero(); len];
    taps_with_norm(&mut weights, span, sps, rolloff, norm);
    weights
}

/// Create heap-backed square-root raised-cosine pulse-shaping taps.
///
/// Taps are unit-energy normalized. See [`taps_with_norm_vec`] for the details.
#[cfg(feature = "alloc")]
#[must_use]
pub fn taps_vec<T>(span: usize, sps: usize, rolloff: T) -> alloc::vec::Vec<T>
where
    T: Float + core::fmt::Debug,
{
    taps_with_norm_vec(span, sps, rolloff, Normalization::UnitEnergy)
}

#[cfg(test)]
mod tests;
