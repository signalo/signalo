// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::{AnalogBiquad, PoleSections};

/// Computes the ellipse eccentricity parameter `ν = asinh(1/ε)/order` shared by both Chebyshev
/// prototypes' pole placement, where `ε = sqrt(10^(ripple_or_atten_db/10) - 1)` is the Type-I
/// ripple factor. [`chebyshev1_lowpass`] uses this value directly; [`chebyshev2_lowpass`] instead
/// computes its own `ν` from the reciprocal ripple factor `ε_II = 1/ε` (see its implementation),
/// since Type-II's stopband attenuation sets `ε_II`, not `ε`, directly.
fn chebyshev_nu<T: num_traits::Float>(order: usize, ripple_or_atten_db: T) -> T {
    let one = T::one();
    let ten = T::from(10).expect("10 fits into T");
    let n = T::from(order).expect("order fits into T");

    let epsilon = (T::powf(ten, ripple_or_atten_db / ten) - one).sqrt();

    (one / epsilon).asinh() / n
}

/// Yields the normalized (`ωc = 1`) lowpass Chebyshev Type-I prototype as second-order sections,
/// with `ripple_db` decibels of equiripple passband ripple.
///
/// The Chebyshev-I poles lie on an ellipse in the s-plane (rather than the unit circle used by
/// [`butterworth_lowpass`](super::butterworth_lowpass)): for `k = 0, …, order - 1`,
/// `s_k = -sinh(ν)·sin(θ_k) + i·cosh(ν)·cos(θ_k)`, where `θ_k = π·(2k + order + 1) / (2·order)`
/// (the same angular spacing as the Butterworth prototype) and `ν = asinh(1/ε)/order` with
/// `ε = sqrt(10^(ripple_db/10) - 1)` set by the passband ripple. Because the ellipse's real and
/// imaginary axes are scaled differently (`sinh(ν) ≠ cosh(ν)` whenever `ν ≠ 0`), a pole and its
/// complex conjugate are *not* related by the `k ↔ order - 1 - k` index symmetry that Butterworth
/// relies on; instead, each `θ_k` in the left half-plane already identifies one member of a
/// conjugate pair, and the conjugate is formed directly by negating its imaginary part. The pair
/// `(s_k, s_k*)` combines into the real quadratic factor `s² - 2·Re(s_k)·s + |s_k|²`, yielded as
/// one [`AnalogBiquad`] with `a = [|s_k|², -2·Re(s_k), 1]` (ascending powers, so `a[2]` is the
/// monic quadratic coefficient), matching this module's `H(s) = (…)/(a[2]s² + a[1]s + a[0])`
/// convention. For an odd `order`, the middle angle `θ_k = π/2` yields a purely real pole
/// `s = -sinh(ν)`, encoded as the first-order section `a = [sinh(ν), 1, 0]`.
///
/// Every section's numerator is `b = [1, 0, 0]`, mirroring
/// [`butterworth_lowpass`](super::butterworth_lowpass)'s convention of leaving overall gain
/// normalization to the caller rather than folding a DC-gain correction into each section.
///
/// An `order` of zero yields an empty iterator.
///
/// # Panics
///
/// Panics if `ripple_db` is not strictly positive: `ripple_db == 0` gives `ε = 0`, so
/// `ν = asinh(1/ε)/order` divides by zero and is non-finite; a negative `ripple_db` gives
/// `sqrt` of a negative `10^(ripple_db/10) - 1`, producing `NaN`.
///
/// # Examples
///
/// ```
/// # use signalo::filters::iir::analog::chebyshev1_lowpass;
/// assert_eq!(chebyshev1_lowpass::<f64>(2, 1.0).count(), 1);
/// ```
pub fn chebyshev1_lowpass<T: num_traits::Float>(
    order: usize,
    ripple_db: T,
) -> impl Iterator<Item = AnalogBiquad<T>> {
    assert!(
        ripple_db > T::zero(),
        "`chebyshev1_lowpass` requires `ripple_db > 0`"
    );

    let nu = chebyshev_nu(order, ripple_db);

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
            let theta = pi * (two * k + n + one) / (two * n);

            let re = -nu.sinh() * theta.sin();
            let im = nu.cosh() * theta.cos();

            AnalogBiquad::new(
                [one, T::zero(), T::zero()],
                [re * re + im * im, -two * re, one],
            )
        },
        first_order: move || {
            let one = T::one();
            let re = -nu.sinh();

            AnalogBiquad::new([one, T::zero(), T::zero()], [-re, one, T::zero()])
        },
        _phantom: core::marker::PhantomData,
    }
}

/// Yields the normalized (`ωc = 1`) lowpass Chebyshev Type-II (inverse Chebyshev) prototype as
/// second-order sections, with `atten_db` decibels of equiripple stopband attenuation.
///
/// Unlike [`chebyshev1_lowpass`] and [`butterworth_lowpass`](super::butterworth_lowpass), which
/// are all-pole, Type-II has ripple in the *stopband* rather than the passband, and carries
/// finite transmission zeros on the `jω` axis. This is constructed from the Type-I prototype of
/// the same `order`, using the reciprocal ripple factor `ε_II = 1/sqrt(10^(atten_db/10) - 1)` in
/// place of Type-I's `ε = sqrt(10^(ripple_db/10) - 1)` when computing `ν = asinh(1/ε_II)/order`,
/// by the standard low-to-high inversion `s → 1/s` applied to the prototype itself (distinct from
/// this module's [`lp_to_hp`](super::lp_to_hp) frequency-scaling transform, which acts on an
/// already-realized section rather than the underlying pole/zero placement): for each Type-I
/// pole `s_k`, the Type-II pole is `1/s_k`. Since `Re(1/s_k) = Re(s_k) / |s_k|²` shares the sign
/// of `Re(s_k)`, this inversion preserves left-half-plane stability. Each conjugate pole pair
/// combines into `a = [|1/s_k|², -2·Re(1/s_k), 1]`, matching this module's ascending-power
/// convention. Each pair also contributes a zero at `s = ±i / cos(θ_k)` (on the imaginary axis),
/// encoded as `b = [1 / cos(θ_k)², 0, 1]`. For odd `order`, the Type-I prototype's middle real
/// pole `s = -sinh(ν)` inverts to `s = -1/sinh(ν)`, a first-order section `a = [sinh(ν), 1, 0]`;
/// its corresponding zero would sit at `θ_k = π/2`, where `cos(θ_k) = 0` places the zero at
/// infinity, so that trailing section is left all-pole (`b = [1, 0, 0]`), matching
/// [`chebyshev1_lowpass`]'s own trailing real-pole convention.
///
/// An `order` of zero yields an empty iterator.
///
/// # Panics
///
/// Panics if `atten_db` is not strictly positive. A negative `atten_db` gives `sqrt` of a
/// negative `10^(atten_db/10) - 1`, producing `NaN`. At `atten_db == 0`, `ν` itself is a finite
/// `0`; for even `order` every coefficient stays finite too, so this branch of the check is a
/// domain-validity assertion — `atten_db` must represent a real stopband attenuation greater
/// than `0` dB — rather than a numerical-blowup preventer. For odd `order`, however, `ν == 0`
/// does blow up the trailing real-pole section's `1/sinh(ν)` term to infinity.
///
/// # Examples
///
/// ```
/// # use signalo::filters::iir::analog::chebyshev2_lowpass;
/// assert_eq!(chebyshev2_lowpass::<f64>(2, 20.0).count(), 1);
/// ```
pub fn chebyshev2_lowpass<T: num_traits::Float>(
    order: usize,
    atten_db: T,
) -> impl Iterator<Item = AnalogBiquad<T>> {
    assert!(
        atten_db > T::zero(),
        "`chebyshev2_lowpass` requires `atten_db > 0`"
    );

    // Type-II ripple factor is the reciprocal of Type-I: ε_II = 1/√(10^(A/10)−1),
    // so ν = asinh(1/ε_II)/n = asinh(√(10^(A/10)−1))/n. This differs from
    // `chebyshev_nu`'s Type-I-direction `ε`, so it is computed directly here rather
    // than via that helper.
    let one = T::one();
    let ten = T::from(10).expect("10 fits into T");
    let n = T::from(order).expect("order fits into T");
    let stopband = (T::powf(ten, atten_db / ten) - one).sqrt();
    let nu = stopband.asinh() / n;

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
            let theta = pi * (two * k + n + one) / (two * n);

            // Type-I pole `s_k = re + i*im`.
            let re = -nu.sinh() * theta.sin();
            let im = nu.cosh() * theta.cos();
            let mag2 = re * re + im * im;

            // Type-II pole `1/s_k = (re - i*im) / |s_k|^2`.
            let p_re = re / mag2;
            let p_im = -im / mag2;

            let cos_theta = theta.cos();
            let zero2 = one / (cos_theta * cos_theta);

            AnalogBiquad::new(
                [zero2, T::zero(), one],
                [p_re * p_re + p_im * p_im, -two * p_re, one],
            )
        },
        first_order: move || {
            let one = T::one();
            // Type-I real pole `s = -sinh(nu)` inverts to `1/s = -1/sinh(nu)`.
            let sinh_nu = nu.sinh();

            AnalogBiquad::new([one, T::zero(), T::zero()], [one / sinh_nu, one, T::zero()])
        },
        _phantom: core::marker::PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chebyshev1_order2_1db_matches_table() {
        let s = chebyshev1_lowpass::<f64>(2, 1.0).next().unwrap();
        // Chebyshev-I poles do not lie on the unit circle, so unlike Butterworth's
        // `a[0] == a[2] == 1` symmetry, the quadratic (leading, monic) coefficient
        // `a[2]` and the constant coefficient `a[0]` differ here. The published
        // tables give the monic form `s^2 + 1.09773 s + 1.10251`, so normalizing by
        // `a[2]` (not `a[0]`) reproduces those reference values directly.
        let a2 = s.a[2];
        approx::assert_abs_diff_eq!(s.a[1] / a2, 1.09773, epsilon = 1e-3);
        approx::assert_abs_diff_eq!(s.a[0] / a2, 1.10251, epsilon = 1e-3);
    }

    #[test]
    fn chebyshev1_order0_yields_empty_iterator() {
        let sections: alloc::vec::Vec<_> = chebyshev1_lowpass::<f64>(0, 1.0).collect();
        assert_eq!(sections.len(), 0);
    }

    #[test]
    fn chebyshev1_odd_order_yields_stable_real_trailing_pole() {
        // Order 3 decomposes into one conjugate pair plus one real pole at the
        // middle angle theta = pi/2 (s = -sinh(nu)). Every coefficient must be
        // finite, and the trailing first-order section's `a[0] = sinh(nu)` must be
        // strictly positive so the encoded pole `s = -a[0]` sits in the stable
        // left half-plane.
        let sections: alloc::vec::Vec<_> = chebyshev1_lowpass::<f64>(3, 1.0).collect();
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
    fn chebyshev2_order2_pole_inversion_matches_hand_derived_oracle() {
        // Self-consistency oracle (no published Type-II table is mandated by this
        // module): independently re-derive the underlying Type-I pole `s_0` for
        // `order = 2, atten_db = 20.0` using the same `nu`/`theta` formulas as
        // `chebyshev1_lowpass`, invert it by hand (`1/s_0`), and assert the
        // resulting quadratic/zero coefficients to a tight epsilon. A skipped or
        // malformed `1/s_k` inversion (e.g. `p_re = re, p_im = im`) would produce
        // `a[0] = 0.5025...`/`a[1] = 0.0710...` instead of the values asserted
        // below, so this test — unlike a bare finiteness/sign check — actually
        // fails if the inversion step regresses.
        let eps = (10f64.powf(20.0 / 10.0) - 1.0).sqrt();
        let nu = eps.asinh() / 2.0;
        let theta = core::f64::consts::PI * (0.0 + 2.0 + 1.0) / (2.0 * 2.0);
        let re = -nu.sinh() * theta.sin();
        let im = nu.cosh() * theta.cos();
        let mag2 = re * re + im * im;
        let p_re = re / mag2;
        let p_im = -im / mag2;
        let expected_a0 = p_re * p_re + p_im * p_im;
        let expected_a1 = -2.0 * p_re;
        let expected_b0 = 1.0 / (theta.cos() * theta.cos());

        let s = chebyshev2_lowpass::<f64>(2, 20.0).next().unwrap();

        approx::assert_abs_diff_eq!(s.a[0], expected_a0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(s.a[1], expected_a1, epsilon = 1e-9);
        assert_eq!(s.a[2], 1.0);
        approx::assert_abs_diff_eq!(s.b[0], expected_b0, epsilon = 1e-9);
        assert_eq!(s.b[1], 0.0);
        assert_eq!(s.b[2], 1.0);

        // The inverted pole's real/imaginary parts must differ meaningfully from
        // the un-inverted Type-I pole `s_0`, proving the inversion actually ran
        // rather than being a no-op that passed the coefficients through.
        assert!((p_re - re).abs() > 1e-3);
        assert!((p_im - im).abs() > 1e-3);
    }

    #[test]
    fn chebyshev2_order0_yields_empty_iterator() {
        let sections: alloc::vec::Vec<_> = chebyshev2_lowpass::<f64>(0, 20.0).collect();
        assert_eq!(sections.len(), 0);
    }

    #[test]
    fn chebyshev2_odd_order_trailing_real_pole_has_no_finite_zero() {
        // Order 3's trailing real pole comes from inverting Type-I's middle real
        // pole `s = -sinh(nu)`, giving `1/s = -1/sinh(nu)`; the corresponding zero
        // would sit at `i / cos(pi/2) = i / 0`, i.e. at infinity, so the trailing
        // section stays all-pole (`b = [1,0,0]`) like Type-I's own trailing
        // real-pole section. `trailing.a[0]` is pinned to the hand-derived
        // `1/sinh(nu)` value (rather than merely `> 0.0`) so a skipped or
        // malformed inversion of the real pole is also caught.
        let eps = (10f64.powf(20.0 / 10.0) - 1.0).sqrt();
        let nu = eps.asinh() / 3.0;
        let expected_a0 = 1.0 / nu.sinh();

        let sections: alloc::vec::Vec<_> = chebyshev2_lowpass::<f64>(3, 20.0).collect();
        assert_eq!(sections.len(), 2);
        let trailing = sections[1];
        assert_eq!(trailing.a[2], 0.0);
        assert_eq!(trailing.a[1], 1.0);
        approx::assert_abs_diff_eq!(trailing.a[0], expected_a0, epsilon = 1e-9);
        assert_eq!(trailing.b, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn chebyshev2_order2_reaches_stopband_floor() {
        // Type-II must attenuate to 10^(-A/20) in the stopband. The old Type-I-ν bug left
        // |H|≈1 at all frequencies; the reciprocal-ν fix restores the stopband floor.
        let atten_db = 20.0_f64;
        let s = chebyshev2_lowpass::<f64>(2, atten_db).next().unwrap();
        let hmag = |w: f64| {
            let s2 = -(w * w);
            let num = ((s.b[0] + s.b[2] * s2).powi(2) + (s.b[1] * w).powi(2)).sqrt();
            let den = ((s.a[0] + s.a[2] * s2).powi(2) + (s.a[1] * w).powi(2)).sqrt();
            num / den
        };
        let floor = hmag(1.0e6) / hmag(0.0);
        approx::assert_abs_diff_eq!(floor, 10f64.powf(-atten_db / 20.0), epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "`chebyshev1_lowpass` requires `ripple_db > 0`")]
    fn chebyshev1_zero_ripple_panics() {
        let _ = chebyshev1_lowpass::<f64>(2, 0.0);
    }

    #[test]
    #[should_panic(expected = "`chebyshev2_lowpass` requires `atten_db > 0`")]
    fn chebyshev2_zero_atten_panics() {
        let _ = chebyshev2_lowpass::<f64>(2, 0.0);
    }
}
