// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! FIR coefficient design helpers.
//!
//! This module contains two FIR tap-design families:
//!
//! - Pulse-shaping and matched-filter helpers, specified in symbol-time
//!   parameters such as span and samples per symbol.
//! - Windowed-sinc helpers, specified by normalized cutoff or band-edge
//!   frequencies and named window families.
//!
//! The helpers in this module generate coefficients into caller-provided
//! storage. With `alloc`, matching helpers can return `Vec`s. Use the FIR
//! convolution or polyphase modules to run those coefficients as filters.
//!
//! Here `span` is the full pulse length in symbols, not the per-side delay.
//! The generated time grid runs from `-span / 2` to `+span / 2` symbols,
//! sampled at `sps` samples per symbol, so the FIR has `span * sps + 1` taps.
//! This full-span convention matches what many SDR libraries use. Some APIs
//! instead use a per-side symbol delay and compute the length as
//! `2 * sps * delay + 1`; for those APIs, use `span = 2 * delay` here. APIs
//! that take `ntaps` directly do not expose this span/half-span distinction.
//!
//! Pulse-shaping helpers generate an odd number of taps. This keeps the pulse
//! centered on an actual tap and gives the symmetric FIR an integer group
//! delay, which is the usual linear-phase layout for these filters.
//!
//! Each pulse shape provides `taps` with a default normalization and
//! `taps_with_norm` for choosing a [`Normalization`] explicitly. With `alloc`,
//! the matching vector-returning helpers are `taps_vec` and
//! `taps_with_norm_vec`.
//!
//! Default normalizations:
//!
//! - Raised cosine: [`Normalization::UnitEnergy`].
//! - Square-root raised cosine: [`Normalization::UnitEnergy`].
//! - GMSK Gaussian pulse: [`Normalization::PassbandGain`].
//! - Windowed-sinc low-pass and band-stop: [`Normalization::PassbandGain`].
//!
//! # Feature flags
//!
//! Raised-cosine, square-root raised-cosine, Kaiser order estimation, and
//! windowed-sinc design require either `std` or `libm`. GMSK Gaussian pulse
//! design requires `libm` for error-function support.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::{Float, ToPrimitive};

#[cfg(feature = "libm")]
pub mod gaussian_pulse;
#[cfg(any(feature = "libm", feature = "std"))]
pub mod raised_cosine;
#[cfg(any(feature = "libm", feature = "std"))]
pub mod root_raised_cosine;
#[cfg(any(feature = "libm", feature = "std"))]
pub mod windowed_sinc;

/// Filter-design tap normalization mode.
#[cfg(any(feature = "libm", feature = "std"))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Normalization {
    /// Leave taps as computed.
    None,
    /// Scale taps so `sum(abs(h[k])^2) == 1`.
    #[default]
    UnitEnergy,
    /// Scale taps so `max(abs(h[k])) == 1`.
    UnitPeak,
    /// Scale taps so `sum(h[k]) == 1`.
    PassbandGain,
}

/// Kaiser FIR design parameters computed from a transition-width specification.
#[cfg(any(feature = "libm", feature = "std"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KaiserOrder<T = f32> {
    /// Number of taps.
    pub num_taps: usize,
    /// Kaiser shape parameter.
    pub beta: T,
}

/// Compute the Kaiser window shape parameter from stop-band attenuation in dB.
///
/// This follows the empirical formula used by `SciPy`'s `scipy.signal.kaiser_beta`:
///
/// - `A <= 21 dB`: `β = 0`
/// - `21 < A <= 50 dB`: `β = 0.5842 * (A - 21)^0.4 + 0.07886 * (A - 21)`
/// - `A > 50 dB`: `β = 0.1102 * (A - 8.7)`
///
/// The two analytic arms join with a small discontinuity at `A = 50 dB`; this
/// is inherent to the empirical fit and matches `SciPy`.
///
/// # References
///
/// Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.
///
/// # Panics
///
/// Panics if `atten_db` is not finite or is negative.
#[cfg(any(feature = "libm", feature = "std"))]
#[must_use]
#[allow(clippy::unwrap_used)]
pub fn kaiser_beta<T>(atten_db: T) -> T
where
    T: Float + core::fmt::Debug,
{
    assert!(
        atten_db.is_finite(),
        "Kaiser attenuation must be finite (got {atten_db:?})"
    );
    assert!(
        atten_db >= T::zero(),
        "Kaiser attenuation must be non-negative (got {atten_db:?})"
    );
    let twenty_one = T::from(21.0).unwrap();
    let fifty = T::from(50.0).unwrap();

    // Standard Kaiser beta approximation from Oppenheim/Schafer and SciPy.
    if atten_db > fifty {
        T::from(0.1102).unwrap() * (atten_db - T::from(8.7).unwrap())
    } else if atten_db > twenty_one {
        let a_minus_21 = atten_db - twenty_one;
        T::from(0.5842).unwrap() * a_minus_21.powf(T::from(0.4).unwrap())
            + T::from(0.07886).unwrap() * a_minus_21
    } else {
        T::zero()
    }
}

/// Estimate Kaiser FIR attenuation in dB from tap count and transition width.
///
/// `transition_width` is in cycles per sample, where Nyquist is `0.5`. This
/// uses the same formula as `SciPy`'s `scipy.signal.kaiser_atten`, but not the
/// same width convention. Use [`kaiser_atten_nyq`] for SciPy-compatible
/// Nyquist-normalized widths.
///
/// # Panics
///
/// Panics if `num_taps` is zero, or if `transition_width` is not finite or is
/// not in `(0, 0.5]`.
#[cfg(any(feature = "libm", feature = "std"))]
#[must_use]
#[allow(clippy::unwrap_used)]
pub fn kaiser_atten<T>(num_taps: usize, transition_width: T) -> T
where
    T: Float + core::fmt::Debug,
{
    assert_kaiser_transition_width(transition_width);

    kaiser_atten_nyq(num_taps, transition_width * T::from(2.0).unwrap())
}

/// Estimate Kaiser FIR attenuation in dB using `SciPy`'s width convention.
///
/// `width_nyq` is normalized to Nyquist, where `1.0` corresponds to
/// π radians/sample. This matches `SciPy`'s `scipy.signal.kaiser_atten`
/// convention.
///
/// Uses the empirical estimate `A = 2.285 * (num_taps - 1) * π * width_nyq + 7.95`,
/// the inverse of the tap-count estimate used by [`kaiser_order_nyq`].
///
/// # References
///
/// Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.
///
/// # Panics
///
/// Panics if `num_taps` is zero, or if `width_nyq` is not finite or is not in
/// `(0, 1]`.
#[cfg(any(feature = "libm", feature = "std"))]
#[must_use]
#[allow(clippy::unwrap_used)]
pub fn kaiser_atten_nyq<T>(num_taps: usize, width_nyq: T) -> T
where
    T: Float + core::fmt::Debug,
{
    assert!(num_taps > 0, "Kaiser tap count must be > 0");
    assert_kaiser_width_nyq(width_nyq);

    T::from(2.285).unwrap()
        * T::from(num_taps - 1).unwrap()
        * T::from(core::f64::consts::PI).unwrap()
        * width_nyq
        + T::from(7.95).unwrap()
}

/// Determine Kaiser FIR tap count and β from attenuation and transition width.
///
/// `transition_width` is in cycles per sample, where Nyquist is `0.5`. This
/// uses the same formula and tap-count behavior as `SciPy`'s
/// `scipy.signal.kaiserord`, but not the same width convention. Use
/// [`kaiser_order_nyq`] for SciPy-compatible Nyquist-normalized widths.
///
/// `atten_db` is interpreted by magnitude to match `SciPy`'s `kaiserord`
/// behavior.
///
/// The tap-count formula and even/odd behavior match `SciPy`: the computed count
/// is rounded up with `ceil` and is not forced odd.
///
/// # Panics
///
/// Panics if `atten_db` is not finite, if `abs(atten_db) < 8`, if
/// `transition_width` is not finite or is not in `(0, 0.5]`, or if the computed
/// tap count does not fit in `usize`.
#[cfg(any(feature = "libm", feature = "std"))]
#[must_use]
#[allow(clippy::unwrap_used)]
pub fn kaiser_order<T>(atten_db: T, transition_width: T) -> KaiserOrder<T>
where
    T: Float + ToPrimitive + core::fmt::Debug,
{
    assert_kaiser_transition_width(transition_width);

    kaiser_order_nyq(atten_db, transition_width * T::from(2.0).unwrap())
}

/// Determine Kaiser FIR tap count and β using `SciPy`'s width convention.
///
/// `width_nyq` is normalized to Nyquist, where `1.0` corresponds to
/// π radians/sample. This matches `SciPy`'s `scipy.signal.kaiserord` convention.
///
/// `atten_db` is interpreted by magnitude to match `SciPy`'s `kaiserord`
/// behavior.
///
/// The tap-count formula and even/odd behavior match `SciPy`: the computed count
/// is rounded up with `ceil` and is not forced odd.
///
/// # Panics
///
/// Panics if `atten_db` is not finite, if `abs(atten_db) < 8`, if `width_nyq`
/// is not finite or is not in `(0, 1]`, or if the computed tap count does not
/// fit in `usize`.
#[cfg(any(feature = "libm", feature = "std"))]
#[must_use]
#[allow(clippy::unwrap_used)]
pub fn kaiser_order_nyq<T>(atten_db: T, width_nyq: T) -> KaiserOrder<T>
where
    T: Float + ToPrimitive + core::fmt::Debug,
{
    assert!(
        atten_db.is_finite(),
        "Kaiser attenuation must be finite (got {atten_db:?})"
    );
    assert_kaiser_width_nyq(width_nyq);

    // Match SciPy `kaiserord`: interpret the ripple/attenuation argument by
    // magnitude, in case callers pass a negative decibel value.
    let attenuation = atten_db.abs();
    assert!(
        attenuation >= T::from(8.0).unwrap(),
        "Kaiser attenuation must be at least 8 dB (got {atten_db:?})"
    );

    let num_taps = ((attenuation - T::from(7.95).unwrap())
        / (T::from(2.285).unwrap() * T::from(core::f64::consts::PI).unwrap() * width_nyq)
        + T::one())
    .ceil();
    let num_taps = num_taps.to_usize().unwrap_or_else(|| {
        panic!(
            "Kaiser tap count {num_taps:?} does not fit in usize \
             (attenuation {attenuation:?} dB, width_nyq {width_nyq:?}); \
             reduce attenuation or increase transition width"
        )
    });

    KaiserOrder {
        num_taps,
        beta: kaiser_beta(attenuation),
    }
}

/// Determine Kaiser FIR design parameters from a transition width in Hz.
///
/// This is a convenience wrapper around [`kaiser_order`]. `sample_rate` and
/// `transition_width_hz` must use the same units.
///
/// # Panics
///
/// Panics if `sample_rate` is not finite or not positive, if
/// `transition_width_hz` is not finite or not positive, or if [`kaiser_order`]
/// rejects the normalized transition width.
#[cfg(any(feature = "libm", feature = "std"))]
#[must_use]
#[allow(clippy::unwrap_used)]
pub fn kaiser_order_hz<T>(atten_db: T, transition_width_hz: T, sample_rate: T) -> KaiserOrder<T>
where
    T: Float + ToPrimitive + core::fmt::Debug,
{
    assert!(
        sample_rate.is_finite(),
        "sample_rate must be finite (got {sample_rate:?})"
    );
    assert!(
        sample_rate > T::zero(),
        "sample_rate must be > 0 (got {sample_rate:?})"
    );
    assert!(
        transition_width_hz.is_finite(),
        "transition_width_hz must be finite (got {transition_width_hz:?})"
    );
    assert!(
        transition_width_hz > T::zero(),
        "transition_width_hz must be > 0 (got {transition_width_hz:?})"
    );

    kaiser_order(atten_db, transition_width_hz / sample_rate)
}

#[cfg(any(feature = "libm", feature = "std"))]
fn assert_kaiser_transition_width<T>(transition_width: T)
where
    T: Float + core::fmt::Debug,
{
    assert!(
        transition_width.is_finite(),
        "Kaiser transition width must be finite (got {transition_width:?})"
    );
    assert!(
        transition_width > T::zero()
            && transition_width <= T::from(0.5).expect("0.5 is representable"),
        "Kaiser transition width must be in (0, 0.5] cycles/sample (got {transition_width:?})"
    );
}

#[cfg(any(feature = "libm", feature = "std"))]
fn assert_kaiser_width_nyq<T>(width_nyq: T)
where
    T: Float + core::fmt::Debug,
{
    assert!(
        width_nyq.is_finite(),
        "Kaiser Nyquist-normalized width must be finite (got {width_nyq:?})"
    );
    assert!(
        width_nyq > T::zero() && width_nyq <= T::one(),
        "Kaiser Nyquist-normalized width must be in (0, 1] (got {width_nyq:?})"
    );
}

#[cfg(any(feature = "libm", feature = "std"))]
/// Return the odd tap count for a pulse with full-symbol `span` and samples per
/// symbol `sps`.
///
/// The length is `span * sps + 1`. `span * sps` must be even so the resulting
/// impulse response has odd length and can be centered on one tap.
///
/// # Panics
///
/// Panics if `span == 0`, `sps < 2`, `span * sps` is odd, `span * sps`
/// overflows, or `span * sps + 1` overflows.
pub(crate) const fn len_from_span(span: usize, sps: usize) -> usize {
    assert!(span > 0, "span must be > 0");
    assert!(sps >= 2, "samples per symbol must be >= 2");
    let samples = span.checked_mul(sps).expect("span * sps overflowed");
    assert!(
        samples.is_multiple_of(2),
        "span * sps must be even for a symmetric pulse"
    );
    samples.checked_add(1).expect("tap count overflowed")
}

#[cfg(any(feature = "libm", feature = "std"))]
/// Apply the requested normalization to generated taps in place.
///
/// `Normalization::None` leaves the taps unchanged. Other modes scale all taps
/// by a single gain factor computed from the current slice contents.
///
/// # Panics
///
/// Panics if the required normalization divisor is not finite or is too small
/// for safe division.
pub(crate) fn normalize<T>(taps: &mut [T], mode: Normalization)
where
    T: Float + core::fmt::Debug,
{
    match mode {
        Normalization::None => {}
        Normalization::UnitEnergy => normalize_unit_energy(taps),
        Normalization::UnitPeak => normalize_unit_peak(taps),
        Normalization::PassbandGain => normalize_passband_gain(taps),
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
fn normalize_unit_energy<T>(taps: &mut [T])
where
    T: Float + core::fmt::Debug,
{
    let sum = taps.iter().fold(T::zero(), |sum, tap| sum + (*tap * *tap));
    if !sum.is_zero() {
        normalize_by_divisor(taps, sum.sqrt(), "unit-energy normalization");
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
fn normalize_unit_peak<T>(taps: &mut [T])
where
    T: Float + core::fmt::Debug,
{
    let max = taps.iter().fold(T::zero(), |max, tap| max.max(tap.abs()));
    if !max.is_zero() {
        normalize_by_divisor(taps, max, "unit-peak normalization");
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
fn normalize_passband_gain<T>(taps: &mut [T])
where
    T: Float + core::fmt::Debug,
{
    let sum = taps.iter().fold(T::zero(), |sum, tap| sum + *tap);
    if !sum.is_zero() {
        normalize_by_divisor(taps, sum, "passband-gain normalization");
    }
}

#[cfg(any(feature = "libm", feature = "std"))]
pub(in crate::filters::fir::design) fn normalize_by_divisor<T>(
    taps: &mut [T],
    divisor: T,
    context: &'static str,
) where
    T: Float + core::fmt::Debug,
{
    let denom = crate::math::safe_normalise_divisor(divisor, context);
    let gain = T::one() / denom;
    for tap in taps {
        *tap = *tap * gain;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn kaiser_beta_matches_scipy_examples() {
        assert_abs_diff_eq!(kaiser_beta(65.0_f64), 6.20426, epsilon = 1e-5);
        assert_abs_diff_eq!(kaiser_beta(40.0_f64), 3.395_321_052, epsilon = 1e-9);
        assert_abs_diff_eq!(kaiser_beta(21.0_f64), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn kaiser_atten_nyq_matches_scipy_example() {
        let attenuation = kaiser_atten_nyq(211, 0.0375_f64);

        assert_abs_diff_eq!(attenuation, 64.480_996_305_939_83, epsilon = 1e-12);
    }

    #[test]
    fn kaiser_atten_converts_cycles_per_sample_width() {
        let cycles_per_sample = kaiser_atten(211, 9.0_f64 / 480.0);
        let nyquist_normalized = kaiser_atten_nyq(211, 0.0375);

        assert_abs_diff_eq!(cycles_per_sample, nyquist_normalized, epsilon = 1e-12);
    }

    #[test]
    fn kaiser_order_nyq_matches_scipy_example() {
        let order = kaiser_order_nyq(65.0_f64, 24.0 / 500.0);

        assert_eq!(order.num_taps, 167);
        assert_abs_diff_eq!(order.beta, 6.20426, epsilon = 1e-5);
    }

    #[test]
    fn kaiser_order_converts_cycles_per_sample_width() {
        let cycles_per_sample = kaiser_order(65.0_f64, 24.0 / 1000.0);
        let nyquist_normalized = kaiser_order_nyq(65.0_f64, 24.0 / 500.0);

        assert_eq!(cycles_per_sample, nyquist_normalized);
    }

    #[test]
    fn kaiser_order_does_not_force_odd_tap_count() {
        let order = kaiser_order(40.0_f64, 0.015);
        let order_nyq = kaiser_order_nyq(40.0_f64, 0.03);

        assert_eq!(order.num_taps, 150);
        assert_eq!(order_nyq.num_taps, 150);
    }

    #[test]
    fn kaiser_order_hz_matches_normalized_width() {
        let normalized = kaiser_order(65.0_f64, 24.0 / 1000.0);
        let hz = kaiser_order_hz(65.0_f64, 24.0, 1000.0);

        assert_eq!(hz, normalized);
    }

    #[test]
    #[should_panic(expected = "at least 8 dB")]
    fn kaiser_order_rejects_too_little_attenuation() {
        let _ = kaiser_order(7.9_f64, 0.1);
    }
}
