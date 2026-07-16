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
//! Raised-cosine and square-root raised-cosine design require either `std` or
//! `libm`. GMSK Gaussian pulse design requires `libm` for error-function
//! support.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

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
