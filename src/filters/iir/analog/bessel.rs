// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::AnalogBiquad;

/// Yields the normalized (`ωc = 1`) lowpass Bessel prototype as second-order sections, for
/// `order` in `0..=4`.
///
/// The Bessel prototype's poles are the roots of the reverse Bessel polynomial `θ_n(s)`, defined
/// by the recurrence `θ_0 = 1`, `θ_1 = s + 1`, `θ_n = (2n − 1)·θ_{n−1} + s²·θ_{n−2}`; for example
/// `θ_2 = s² + 3s + 3`. Unlike [`butterworth_lowpass`](super::butterworth_lowpass)'s poles (which
/// lie exactly on the unit circle) or [`chebyshev1_lowpass`](super::chebyshev1_lowpass)'s (which
/// lie on a fixed ellipse), `θ_n`'s roots have no closed form for general `n` and are instead
/// tabulated here after two normalization steps applied uniformly to every order: first, the
/// unscaled transfer function `H(s) = θ_n(0) / θ_n(s)` has its own "natural", delay-normalized
/// `−3 dB` point at some frequency `w1` (found numerically, since `θ_n`'s roots have no closed
/// form); substituting `s → s/w0` with `w0 = 1/w1` rescales that critical frequency to exactly
/// `ω = 1`, since `H(s/w0)` evaluated at `s = j·w0·w1 = j·1` reproduces `H(s)`'s own response at
/// its critical frequency `j·w1`. Second, each resulting pole pair (poles of the unscaled `θ_n`
/// scaled by `w0`) is combined into a monic quadratic factor `s² − 2·Re(s_k)·s + |s_k|²`, and each
/// unpaired real pole (present for odd `order`) into a monic linear factor `s − s_k`, matching this
/// module's `H(s) = (…)/(a[2]s² + a[1]s + a[0])` ascending-power convention. Because `θ_n`'s roots
/// have no closed form, the resulting coefficients are hardcoded per order rather than derived at
/// call time from the recurrence above.
///
/// Every section's numerator is `b = [1, 0, 0]`, mirroring
/// [`butterworth_lowpass`](super::butterworth_lowpass) and
/// [`chebyshev1_lowpass`](super::chebyshev1_lowpass)'s convention of leaving overall gain
/// normalization to the caller rather than folding a DC-gain correction into each section.
///
/// An `order` of zero yields an empty iterator.
///
/// # Panics
///
/// Panics if `order` is greater than `4`: the hardcoded pole tables this function relies on (see
/// above) only cover `order` in `0..=4`. Returning coefficients silently truncated or approximated
/// to an unsupported order would produce a filter with the wrong number of poles, which is worse
/// than failing loudly.
///
/// # Examples
///
/// ```
/// # use signalo::filters::iir::analog::bessel_lowpass;
/// assert_eq!(bessel_lowpass::<f64>(2).count(), 1);
/// ```
pub fn bessel_lowpass<T: num_traits::Float>(order: usize) -> impl Iterator<Item = AnalogBiquad<T>> {
    // Hardcoded quadratic (and, for odd order, one trailing linear) factors of the reverse
    // Bessel polynomial `θ_n`, after normalizing poles so that `H(jw)` reaches `-3 dB` at
    // `w = 1` with unit DC gain (see the doc comment above for the derivation). Each row is
    // one `AnalogBiquad`'s denominator `[a0, a1, a2]` in this module's ascending-power
    // convention; `a2 == 0` marks a first-order (real-pole) factor. Poles: order 2 at
    // `-1.1016 ± j0.6360`; order 3 at `-1.0474 ± j0.9993` and `-1.3227`; order 4 at
    // `-0.9952 ± j1.2571` and `-1.3701 ± j0.4102` — matching standard published Bessel
    // filter pole tables.
    const SECTIONS: [&[[f64; 3]]; 5] = [
        // order 0: no poles.
        &[],
        // order 1: theta_1 = s + 1, already at its natural -3 dB point (w0 = 1).
        &[[1.0, 1.0, 0.0]],
        // order 2: theta_2 = s^2 + 3s + 3, poles scaled by w0 = 1 / 1.3616541287161308.
        // a0 comes out to 1.6180339887498947, matching the golden ratio
        // phi = (1 + sqrt(5)) / 2 to within floating-point rounding (within ~1 ULP of phi).
        &[[1.6180339887498947, 2.203202661184323, 1.0]],
        // order 3: theta_3 = s^3 + 6s^2 + 15s + 15, poles scaled by
        // w0 = 1 / 1.755672368681211.
        &[
            [2.0955953641807, 2.0948183220178698, 1.0],
            [1.3226757999104441, 1.0, 0.0],
        ],
        // order 4: theta_4 = s^4 + 10s^3 + 45s^2 + 105s + 105, poles scaled by
        // w0 = 1 / 2.113917674904216.
        &[
            [2.5707553248094617, 1.9904175287005477, 1.0],
            [2.0453906910156436, 2.740135661102889, 1.0],
        ],
    ];

    let sections = SECTIONS.get(order).unwrap_or_else(|| {
        panic!("`bessel_lowpass` only supports `order` in `0..=4`, got {order}")
    });

    sections.iter().map(|&[a0, a1, a2]| {
        AnalogBiquad::new(
            [T::one(), T::zero(), T::zero()],
            [
                T::from(a0).expect("Bessel coefficient fits into T"),
                T::from(a1).expect("Bessel coefficient fits into T"),
                T::from(a2).expect("Bessel coefficient fits into T"),
            ],
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_order2_damping_ratio() {
        let s = bessel_lowpass::<f64>(2).next().unwrap();
        let ratio = s.a[1] / (s.a[0] * s.a[2]).sqrt();
        approx::assert_abs_diff_eq!(ratio, 3.0f64.sqrt(), epsilon = 1e-3);
    }

    #[test]
    fn bessel_order0_yields_empty_iterator() {
        let sections: alloc::vec::Vec<_> = bessel_lowpass::<f64>(0).collect();
        assert_eq!(sections.len(), 0);
    }

    #[test]
    fn bessel_order1_is_real_pole_at_unit_cutoff() {
        // theta_1 = s + 1 is already at its natural -3 dB point, so order 1 needs no
        // frequency rescaling: the single real pole sits exactly at s = -1.
        let s = bessel_lowpass::<f64>(1).next().unwrap();
        assert_eq!(s.a, [1.0, 1.0, 0.0]);
        assert_eq!(s.b, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn bessel_odd_order_yields_stable_real_trailing_pole() {
        // Order 3 decomposes into one conjugate pair plus one real pole (theta_3's
        // middle root). Every coefficient must be finite, and the trailing
        // first-order section's `a[0] = -s_pole` must be strictly positive so the
        // encoded pole `s = -a[0]` sits in the stable left half-plane.
        let sections: alloc::vec::Vec<_> = bessel_lowpass::<f64>(3).collect();
        assert_eq!(sections.len(), 2);
        let trailing = sections[1];
        assert_eq!(trailing.a[2], 0.0);
        assert_eq!(trailing.a[1], 1.0);
        assert!(trailing.a[0] > 0.0);
        for section in &sections {
            assert!(section.a[0].is_finite());
            assert!(section.a[1].is_finite());
            assert!(section.a[2].is_finite());
        }
    }

    #[test]
    fn bessel_order4_yields_two_stable_conjugate_pairs() {
        // Order 4 has no single-coefficient oracle like order 2's, so this checks the
        // structural contract instead: exactly two quadratic sections, both encoding
        // stable (positive `a[0]`/`a[1]`, monic `a[2]`) conjugate pole pairs, and the two
        // sections are numerically distinct (theta_4 has two different pole pairs, not
        // a repeated root).
        let sections: alloc::vec::Vec<_> = bessel_lowpass::<f64>(4).collect();
        assert_eq!(sections.len(), 2);
        for section in &sections {
            assert_eq!(section.a[2], 1.0);
            assert!(section.a[0] > 0.0);
            assert!(section.a[1] > 0.0);
            assert!(section.a[0].is_finite());
            assert!(section.a[1].is_finite());
        }
        assert!(
            (sections[0].a[0] - sections[1].a[0]).abs() > 1e-6
                || (sections[0].a[1] - sections[1].a[1]).abs() > 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "`bessel_lowpass` only supports `order` in `0..=4`")]
    fn bessel_order_above_4_panics() {
        let _ = bessel_lowpass::<f64>(5).next();
    }

    /// Evaluates one section's `H(s) = a0 / (a2*s^2 + a1*s + a0)` at `s = j`, returning
    /// `(re, im)` of the complex result. `a2*s^2 = -a2` at `s = j`, so the denominator is
    /// `(a0 - a2) + j*a1`; this is multiplied out by hand (rather than pulled in via
    /// `num_complex`, an optional dependency this module does not otherwise require) to
    /// keep the test self-contained.
    fn section_response_at_j1(section: AnalogBiquad<f64>) -> (f64, f64) {
        let [a0, a1, a2] = section.a;
        let denom_re = a0 - a2;
        let denom_im = a1;
        let denom_mag2 = denom_re * denom_re + denom_im * denom_im;
        // a0 / (denom_re + j*denom_im) = a0*(denom_re - j*denom_im) / |denom|^2
        let re = a0 * denom_re / denom_mag2;
        let im = -a0 * denom_im / denom_mag2;
        (re, im)
    }

    /// Multiplies two complex numbers given as `(re, im)` pairs.
    fn complex_mul(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
        (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
    }

    #[test]
    fn bessel_cascade_reaches_minus_3db_at_unit_frequency() {
        // This closes a gap the mandated `bessel_order2_damping_ratio` oracle cannot see:
        // the ratio `a1/sqrt(a0*a2)` only encodes a quadratic's damping ratio (i.e. its
        // "shape"), and is invariant to whichever frequency the section is scaled to, so
        // it passes identically whether poles were scaled by `w0` or `1/w0`. This test
        // instead directly evaluates the full cascade's magnitude response at `s = j*1`
        // and asserts `|H(j1)|^2 = 0.5` (the defining property of a `-3 dB` point),
        // which only holds for the correctly-direction-scaled poles.
        for order in 1..=4 {
            let mut response = (1.0, 0.0);
            for section in bessel_lowpass::<f64>(order) {
                response = complex_mul(response, section_response_at_j1(section));
            }
            let mag2 = response.0 * response.0 + response.1 * response.1;
            approx::assert_abs_diff_eq!(mag2, 0.5, epsilon = 1e-9);
        }
    }
}
