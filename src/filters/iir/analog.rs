// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Analog (s-domain) second-order section design data.
//!
//! An [`AnalogBiquad`] represents a normalized continuous-time second-order transfer
//! function in the Laplace variable `s`:
//!
//! ```text
//! H(s) = (b[2] s² + b[1] s + b[0]) / (a[2] s² + a[1] s + a[0])
//! ```
//!
//! This is design-time data, not a runnable filter: unlike [`biquad::Biquad`](super::biquad),
//! [`AnalogBiquad`] holds no state and has no [`Filter`](crate::traits::Filter) implementation,
//! because an s-domain transfer function cannot be evaluated sample-by-sample directly. Classic
//! analog prototypes (Butterworth, Chebyshev, Bessel, and so on) are naturally expressed in the
//! s-domain, where their pole/zero placement is simplest to reason about.
//!
//! To turn an [`AnalogBiquad`] into something that can process a signal, discretize it (for
//! example via the bilinear transform) into a digital [`biquad::Config`](super::biquad::Config),
//! which drives the [`biquad::Biquad`](super::biquad::Biquad) filter.

/// A normalized analog (s-domain) second-order section:
/// `H(s) = (b[2] s² + b[1] s + b[0]) / (a[2] s² + a[1] s + a[0])`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalogBiquad<T> {
    /// Numerator coefficients `[b0, b1, b2]`.
    pub b: [T; 3],
    /// Denominator coefficients `[a0, a1, a2]`.
    pub a: [T; 3],
}

impl<T> AnalogBiquad<T> {
    /// Creates a section from numerator `b = [b0,b1,b2]` and denominator `a = [a0,a1,a2]`.
    pub const fn new(b: [T; 3], a: [T; 3]) -> Self {
        Self { b, a }
    }
}

/// Yields the normalized (`ωc = 1`) lowpass Butterworth prototype as second-order sections.
///
/// The Butterworth prototype of a given `order` has poles spaced evenly around the left half of
/// the unit circle in the s-plane, at angles `s_k = exp(i·π·(2k + order + 1) / (2·order))` for
/// `k = 0, …, order - 1`. Complex-conjugate pole pairs `(s_k, s_k*)` combine into a real quadratic
/// factor `s² - 2·Re(s_k)·s + 1`, since `|s_k| = 1`; this iterator yields one [`AnalogBiquad`] per
/// such pair. For an odd `order`, one real pole at `s = -1` remains unpaired and is yielded as a
/// trailing first-order section, encoded with `a = [1, 1, 0]` and `b = [1, 0, 0]`.
///
/// An `order` of zero yields an empty iterator.
///
/// # Panics
///
/// Panics while iterating if `order` or a pole-pair index does not fit into `T`, which cannot
/// happen for any of `num_traits::Float`'s built-in implementors (`f32`, `f64`) and a realistic
/// `order`.
///
/// # Examples
///
/// ```
/// # use signalo::filters::iir::analog::butterworth_lowpass;
/// assert_eq!(butterworth_lowpass::<f64>(2).count(), 1);
/// ```
pub fn butterworth_lowpass<T: num_traits::Float>(
    order: usize,
) -> impl Iterator<Item = AnalogBiquad<T>> {
    PoleSections {
        pair_count: order / 2,
        pair_index: 0,
        first_order_count: order % 2,
        pair: move |pair_index: usize| {
            let one = T::one();
            let two = one + one;
            let four = two + two;
            let pi = four * T::atan(one);

            let n = T::from(order).expect("order fits into T");
            let k = T::from(pair_index).expect("pair_index fits into T");
            let angle = pi * (two * k + n + one) / (two * n);
            let re = angle.cos();

            AnalogBiquad::new([one, T::zero(), T::zero()], [one, -two * re, one])
        },
        first_order: || {
            let one = T::one();

            AnalogBiquad::new([one, T::zero(), T::zero()], [one, one, T::zero()])
        },
        _phantom: core::marker::PhantomData,
    }
}

/// Transforms a normalized lowpass [`AnalogBiquad`] to a lowpass with cutoff `wc`.
///
/// Applies the substitution `s → s/wc` to `section`'s transfer function
/// `H(s) = (b[2] s² + b[1] s + b[0]) / (a[2] s² + a[1] s + a[0])`. Clearing the resulting
/// `1/wc²` factors from both numerator and denominator scales each coefficient by a power of
/// `wc` matching its order: the constant term (`b[0]`/`a[0]`) picks up `wc²`, the linear term
/// (`b[1]`/`a[1]`) picks up `wc`, and the quadratic term (`b[2]`/`a[2]`) is unchanged.
///
/// # Examples
///
/// ```
/// # use signalo::filters::iir::analog::{AnalogBiquad, lp_to_lp};
/// let section = AnalogBiquad::new([1.0_f64, 0.0, 0.0], [1.0, 2.0f64.sqrt(), 1.0]);
/// let scaled = lp_to_lp(section, 2.0);
/// assert_eq!(scaled.a, [4.0, 2.0 * 2.0f64.sqrt(), 1.0]);
/// ```
pub fn lp_to_lp<T: num_traits::Float>(section: AnalogBiquad<T>, wc: T) -> AnalogBiquad<T> {
    let wc2 = wc * wc;

    AnalogBiquad::new(
        [section.b[0] * wc2, section.b[1] * wc, section.b[2]],
        [section.a[0] * wc2, section.a[1] * wc, section.a[2]],
    )
}

/// Transforms a normalized lowpass [`AnalogBiquad`] to a highpass with cutoff `wc`.
///
/// Applies the substitution `s → wc/s` to `section`'s transfer function
/// `H(s) = (b[2] s² + b[1] s + b[0]) / (a[2] s² + a[1] s + a[0])`. Multiplying numerator and
/// denominator through by `s²/wc²` clears the negative powers of `s` and swaps the roles of the
/// constant and quadratic terms: the former constant term (`b[0]`/`a[0]`) becomes the new
/// quadratic term, the former quadratic term (`b[2]`/`a[2]`) becomes the new constant term scaled
/// by `wc²`, and the linear term (`b[1]`/`a[1]`) is scaled by `wc`.
///
/// # Examples
///
/// ```
/// # use signalo::filters::iir::analog::{AnalogBiquad, lp_to_hp};
/// let section = AnalogBiquad::new([1.0_f64, 0.0, 0.0], [1.0, 2.0f64.sqrt(), 1.0]);
/// let highpass = lp_to_hp(section, 1.0);
/// assert_eq!(highpass.b, [0.0, 0.0, 1.0]);
/// ```
pub fn lp_to_hp<T: num_traits::Float>(section: AnalogBiquad<T>, wc: T) -> AnalogBiquad<T> {
    let wc2 = wc * wc;

    AnalogBiquad::new(
        [section.b[2] * wc2, section.b[1] * wc, section.b[0]],
        [section.a[2] * wc2, section.a[1] * wc, section.a[0]],
    )
}

/// Bilinear-transforms an analog section to digital biquad coefficients
/// `[b0, b1, b2, a1, a2]` (the layout used by [`biquad`](super::biquad)).
///
/// Substitutes `s = 2·sample_rate·(1 − z⁻¹)/(1 + z⁻¹)` into `section`'s transfer function
/// `H(s) = (b[2] s² + b[1] s + b[0]) / (a[2] s² + a[1] s + a[0])`, then clears the negative
/// powers of `z` by multiplying through by `(1 + z⁻¹)²` and collecting `z⁻¹` powers.
/// This is the standard bilinear transform constant `K = 2·sample_rate`, used as-is: the
/// frequency warping inherent to the bilinear transform is compensated exactly once, upstream,
/// by scaling `section`'s analog cutoff to `2·sample_rate·tan(π·f/sample_rate)` for the desired
/// digital cutoff `f` — for example via [`lp_to_lp`] — before calling this function. Composing
/// that pre-warped cutoff with the plain substitution constant here reproduces the reference RBJ
/// Audio EQ Cookbook biquad recipe exactly; re-applying the tangent warp to `K` itself would
/// double-warp the result.
///
/// The output is normalized so `a0 == 1`, matching the convention of
/// [`Butterworth::lowpass`](super::biquad::coefficients::Butterworth::lowpass) and other
/// [`biquad::coefficients`](super::biquad::coefficients) factories.
///
/// # Examples
///
/// ```
/// # use signalo::filters::iir::analog::{butterworth_lowpass, lp_to_lp, bilinear};
/// let fs = 48000.0_f64;
/// let fc = 1000.0_f64;
/// let proto = butterworth_lowpass::<f64>(2).next().unwrap();
/// let wc = 2.0 * fs * (core::f64::consts::PI * fc / fs).tan();
/// let warped = lp_to_lp(proto, wc);
/// let coeffs = bilinear(warped, fs);
/// assert!(coeffs[0] > 0.0);
/// ```
///
/// This function performs no clamping or validation of `section` — it is a pure coefficient
/// transform. The only source of non-finite output is `section` itself, which propagates
/// through unmodified if, for instance, an upstream pre-warp computation such as
/// `2·sample_rate·tan(π·f/sample_rate)` was evaluated at `f ≥ sample_rate / 2` (a pole of `tan`)
/// before being passed into [`lp_to_lp`] or an equivalent scaling step.
pub fn bilinear<T: num_traits::Float>(section: AnalogBiquad<T>, sample_rate: T) -> [T; 5] {
    let two = T::one() + T::one();
    let k = two * sample_rate;
    let k2 = k * k;

    let b0 = section.b[2] * k2 + section.b[1] * k + section.b[0];
    let b1 = two * (section.b[0] - section.b[2] * k2);
    let b2 = section.b[2] * k2 - section.b[1] * k + section.b[0];

    let a0 = section.a[2] * k2 + section.a[1] * k + section.a[0];
    let a1 = two * (section.a[0] - section.a[2] * k2);
    let a2 = section.a[2] * k2 - section.a[1] * k + section.a[0];

    [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
}

/// Lazily computes normalized Butterworth lowpass sections one at a time.
///
/// Shared by analog lowpass prototypes (Butterworth, Chebyshev, …) whose poles decompose into
/// complex-conjugate pairs plus, for odd order, one unpaired real pole. `pair` computes the
/// quadratic section for pole-pair index `k` (`0 <= k < pair_count`); `first_order` computes the
/// trailing real-pole section and is called `first_order_count` times.
struct PoleSections<T, Pair, First> {
    pair_count: usize,
    pair_index: usize,
    first_order_count: usize,
    pair: Pair,
    first_order: First,
    _phantom: core::marker::PhantomData<T>,
}

impl<T, Pair, First> Iterator for PoleSections<T, Pair, First>
where
    Pair: FnMut(usize) -> AnalogBiquad<T>,
    First: FnMut() -> AnalogBiquad<T>,
{
    type Item = AnalogBiquad<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pair_index < self.pair_count {
            let section = (self.pair)(self.pair_index);
            self.pair_index += 1;

            return Some(section);
        }

        if self.first_order_count > 0 {
            self.first_order_count -= 1;

            return Some((self.first_order)());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analog_biquad_new_stores_coeffs() {
        let s = AnalogBiquad::new([1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(s.b, [1.0, 2.0, 3.0]);
        assert_eq!(s.a, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn butterworth_order2_is_canonical_section() {
        let sections: alloc::vec::Vec<_> = butterworth_lowpass::<f64>(2).collect();
        assert_eq!(sections.len(), 1);
        approx::assert_abs_diff_eq!(sections[0].a[0], 1.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(sections[0].a[1], 2.0f64.sqrt(), epsilon = 1e-9);
        approx::assert_abs_diff_eq!(sections[0].a[2], 1.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(sections[0].b[0], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn butterworth_order0_yields_empty_iterator() {
        let sections: alloc::vec::Vec<_> = butterworth_lowpass::<f64>(0).collect();
        assert_eq!(sections.len(), 0);
    }

    #[test]
    fn lp_to_lp_identity_at_unit_cutoff() {
        let s = butterworth_lowpass::<f64>(2).next().unwrap();
        let t = lp_to_lp(s, 1.0);
        for i in 0..3 {
            approx::assert_abs_diff_eq!(t.a[i], s.a[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn lp_to_lp_scales_cutoff() {
        let s = butterworth_lowpass::<f64>(2).next().unwrap();
        let wc = 5.0;
        let t = lp_to_lp(s, wc);
        // `a[2]` (quadratic coefficient) is invariant under `s -> s/wc`, while `a[0]`
        // (constant term) picks up `wc^2`, matching the standard `s^2 + sqrt(2)*wc*s + wc^2`
        // Butterworth form, so the ratio is `a[0] / a[2]`, not `a[2] / a[0]`.
        approx::assert_abs_diff_eq!(t.a[0] / t.a[2], wc * wc, epsilon = 1e-9);
    }

    #[test]
    fn lp_to_lp_scales_numerator_coefficients() {
        // A section with all-nonzero numerator coefficients, so every `b` term is exercised.
        let s = AnalogBiquad::new([3.0_f64, 5.0, 7.0], [1.0, 2.0f64.sqrt(), 1.0]);
        let wc = 5.0;
        let t = lp_to_lp(s, wc);
        approx::assert_abs_diff_eq!(t.b[0], 3.0 * wc * wc, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.b[1], 5.0 * wc, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.b[2], 7.0, epsilon = 1e-9);
    }

    #[test]
    fn lp_to_hp_swaps_and_scales_denominator_at_unit_cutoff() {
        // Canonical order-2 Butterworth: a = [1, sqrt(2), 1], symmetric so `a` is
        // unchanged by the order-0/order-2 swap at wc = 1, while `b = [1,0,0]`
        // becomes `[0,0,1]`, i.e. a pure s^2 highpass numerator.
        let s = butterworth_lowpass::<f64>(2).next().unwrap();
        let t = lp_to_hp(s, 1.0);
        approx::assert_abs_diff_eq!(t.a[0], 1.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.a[1], 2.0f64.sqrt(), epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.a[2], 1.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.b[0], 0.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.b[1], 0.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.b[2], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn lp_to_hp_scales_cutoff() {
        // A section with distinct, nonzero denominator coefficients so the swap
        // and per-term scaling by `wc` are all independently exercised.
        let s = AnalogBiquad::new([1.0_f64, 0.0, 0.0], [2.0, 3.0, 4.0]);
        let wc = 5.0;
        let t = lp_to_hp(s, wc);
        approx::assert_abs_diff_eq!(t.a[0], 4.0 * wc * wc, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.a[1], 3.0 * wc, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(t.a[2], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn bilinear_butterworth_matches_rbj_lowpass() {
        use crate::filters::iir::biquad::coefficients::Butterworth;
        let fs = 48000.0_f64;
        let fc = 1000.0_f64;
        let proto = butterworth_lowpass::<f64>(2).next().unwrap();
        let warped = lp_to_lp(proto, 2.0 * fs * (core::f64::consts::PI * fc / fs).tan());
        let got = bilinear(warped, fs);
        let want = Butterworth::lowpass(fs, fc);
        for i in 0..5 {
            approx::assert_abs_diff_eq!(got[i], want[i], epsilon = 1e-6);
        }
    }
}
