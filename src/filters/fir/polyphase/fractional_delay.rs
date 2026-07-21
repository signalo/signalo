// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Fractional-delay polyphase FIR design helpers.
//!
//! A dense prototype, sampled on a `num_phases`-times finer grid than the
//! input stream, can be split into `num_phases` branches and executed by
//! [`PolyphaseFir`](super::fir::PolyphaseFir). Branch `q` implements a
//! fractional sample offset of `q / num_phases`. A caller chooses an integer
//! phase index for the desired fractional offset, pushes the input history into
//! [`PolyphaseFir`](super::fir::PolyphaseFir), and executes that phase over the
//! current delay line.
//!
//! These helpers cover the fractional-delay-specific parts: dense prototype
//! geometry and optional Kaiser low-pass prototype generation. Dense prototypes
//! are packed with [`filter_bank::pack_prototype_taps`](super::filter_bank::pack_prototype_taps),
//! and the executor remains a regular [`PolyphaseFir`](super::fir::PolyphaseFir).
//!
//! # Length model
//!
//! Start from the input-rate filter length you want each phase branch to
//! execute. This may be the minimum length needed to meet design requirements,
//! such as a target attenuation for a given transition width. Call that length
//! `H`. For fractional-delay interpolation, branch 0 is the ordinary input-rate
//! interpolation filter with `H` taps; the other branches are fractional shifts
//! of the same filter.
//!
//! If a design helper gives a required filter length `H`, use that value
//! directly as `taps_per_phase`. For example, [`kaiser_order`](crate::filters::fir::design::kaiser_order)
//! returns the input-rate Kaiser filter length in `order.num_taps`; pass
//! `order.num_taps` to [`prototype_len`] and use the returned dense length when
//! generating the prototype.
//!
//! Use [`prototype_len`]`(H, num_phases)` as the dense prototype length to
//! design. The packing step then forms phase branches by taking every
//! `num_phases`-th prototype tap for each branch. After packing, the coefficient
//! table has `num_phases * H` entries; the final `num_phases - 1` slots are
//! padding.
//!
//! Even and odd `H` are both valid. An odd branch length centers phase 0 on an
//! integer input sample, which is the usual linear-phase interpolation layout.
//! An even branch length is still supported for callers that intentionally want
//! that geometry.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

#[cfg(any(feature = "libm", feature = "std"))]
use crate::filters::fir::design::{windowed_sinc::kaiser, KaiserOrder};

/// Returns the meaningful dense-prototype length for a per-phase tap count.
///
/// With `P = num_phases` and `H = taps_per_phase`, one phase branch contains
/// `H` taps separated by `P` dense samples. Its `H - 1` gaps span
/// `(H - 1) * P` samples, and counting both endpoints gives
/// `L = (H - 1) * P + 1`.
///
/// Choose `taps_per_phase` as the full input-rate branch length. For example,
/// if [`kaiser_order`](crate::filters::fir::design::kaiser_order) returns
/// `order.num_taps`, use `prototype_len(order.num_taps, num_phases)` to get the
/// dense prototype length to design.
///
/// This formula does not require odd `H`. If `H` is odd, `L` is always odd. If
/// `H` is even, `L` is even when `P` is odd and odd when `P` is even. A caller
/// that needs phase 0 (the first branch) to be centered on an integer input
/// sample should choose an odd input-rate branch length, pass that length as
/// `taps_per_phase`, use the returned dense length to design the prototype taps,
/// and then pack those taps with
/// [`filter_bank::pack_prototype_taps`](super::filter_bank::pack_prototype_taps).
///
/// The returned length is congruent to one modulo `num_phases`. It is not
/// rounded to odd.
///
/// # Panics
///
/// Panics when `taps_per_phase` or `num_phases` is zero, or if the length
/// overflows `usize`.
#[must_use]
pub const fn prototype_len(taps_per_phase: usize, num_phases: usize) -> usize {
    assert!(
        taps_per_phase > 0,
        "fractional-delay bank taps per phase must be nonzero"
    );
    assert!(num_phases > 0, "fractional-delay bank needs phase branches");
    let Some(span) = num_phases.checked_mul(taps_per_phase - 1) else {
        panic!("fractional-delay prototype length overflowed");
    };
    let Some(len) = span.checked_add(1) else {
        panic!("fractional-delay prototype length overflowed");
    };
    len
}

/// Fills a dense Kaiser low-pass prototype for fractional-delay interpolation.
///
/// `order` is the Kaiser order the caller would use for a regular input-rate
/// low-pass filter. Its `num_taps` becomes the per-phase tap count, and the
/// dense prototype length must be `prototype_len(order.num_taps, num_phases)`.
/// `cutoff` is also in input-rate cycles per sample; this helper converts it to
/// the dense prototype rate internally.
///
/// [`kaiser::lowpass_with_beta`] normalizes the whole dense low-pass prototype
/// to unit DC/passband gain (`sum(h) == 1`). Because each packed phase branch
/// keeps every `num_phases`-th tap, each branch would otherwise have roughly
/// `1 / num_phases` DC gain. The dense taps are multiplied by `num_phases`
/// before packing so every branch has approximately unit DC gain after
/// polyphase decomposition and preserves constant inputs. Phase 0 matches the
/// ordinary sample-rate Kaiser low-pass defined by `order` and `cutoff`; later
/// phases are fractional shifts of the same filter.
///
/// # Panics
///
/// Panics when `num_phases` or `order.num_taps` is zero, when `prototype.len()`
/// does not equal `prototype_len(order.num_taps, num_phases)`, when the dense-rate
/// cutoff is invalid, or when `num_phases` cannot be represented as `T`.
#[cfg(any(feature = "libm", feature = "std"))]
pub fn kaiser_lowpass_prototype<T>(
    prototype: &mut [T],
    num_phases: usize,
    order: KaiserOrder<T>,
    cutoff: T,
) where
    T: Float + core::fmt::Debug,
{
    assert!(num_phases > 0, "fractional-delay bank needs phase branches");
    assert!(
        order.num_taps > 0,
        "fractional-delay bank taps per phase must be nonzero"
    );
    assert_eq!(
        prototype.len(),
        prototype_len(order.num_taps, num_phases),
        "fractional-delay prototype length must match order and phase count"
    );
    let Some(phases) = T::from(num_phases) else {
        panic!("fractional-delay phase count must be representable");
    };

    // The prototype is sampled `num_phases` times faster than the input stream,
    // so an input-rate cutoff in cycles/sample becomes `cutoff / num_phases` at
    // the dense prototype rate.
    kaiser::lowpass_with_beta(prototype, order.beta, cutoff / phases);
    // `lowpass_with_beta` normalizes the whole dense low-pass prototype to
    // `sum(h) == 1`. Phase packing gives each branch roughly one `num_phases`-th
    // of that sum; compensate so each branch has approximately unity DC gain.
    for tap in prototype {
        *tap = *tap * phases;
    }
}

/// Creates a dense Kaiser low-pass prototype for fractional-delay interpolation.
///
/// This is a heap-backed convenience wrapper around [`kaiser_lowpass_prototype`].
///
/// # Panics
///
/// Panics if [`kaiser_lowpass_prototype`] panics.
#[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
#[must_use]
pub fn kaiser_lowpass_prototype_vec<T>(
    num_phases: usize,
    order: KaiserOrder<T>,
    cutoff: T,
) -> alloc::vec::Vec<T>
where
    T: Float + core::fmt::Debug,
{
    let mut prototype = alloc::vec![T::zero(); prototype_len(order.num_taps, num_phases)];
    kaiser_lowpass_prototype(&mut prototype, num_phases, order, cutoff);
    prototype
}

#[cfg(test)]
mod tests {
    use core::ops::{Add, Mul};

    use circular_buffer::FixedCircularBuffer;
    use num_traits::Zero;

    use super::prototype_len;
    use crate::filters::fir::polyphase::filter_bank::{pack_prototype_taps, packed_len, Config};
    use crate::filters::fir::polyphase::fir::{PolyphaseFir, PolyphaseFirArray};
    use crate::storage::RingBuffer;
    use crate::traits::WithConfig;

    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    use super::kaiser_lowpass_prototype_vec;

    #[cfg(feature = "alloc")]
    use crate::filters::fir::polyphase::fir::PolyphaseFirVec;

    /// Test-only bundle proving one PFB pass can run two parallel real filters.
    ///
    /// With `T = f32`, this stores two coefficient weights and, as the
    /// dot-product output, the two corresponding output accumulators.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    struct TapPair<T = f32> {
        /// First parallel filter value.
        first: T,
        /// Second parallel filter value.
        second: T,
    }

    impl<T> Add for TapPair<T>
    where
        T: Add<Output = T>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            Self {
                first: self.first + rhs.first,
                second: self.second + rhs.second,
            }
        }
    }

    impl<T> Zero for TapPair<T>
    where
        T: Zero + Add<Output = T>,
    {
        fn zero() -> Self {
            Self {
                first: T::zero(),
                second: T::zero(),
            }
        }

        fn is_zero(&self) -> bool {
            self.first.is_zero() && self.second.is_zero()
        }
    }

    impl Mul<TapPair> for f32 {
        type Output = TapPair;

        fn mul(self, rhs: TapPair) -> TapPair {
            // This is the key generic operation: one sample times one bundled
            // tap contributes to both output filters.
            TapPair {
                first: self * rhs.first,
                second: self * rhs.second,
            }
        }
    }

    fn push_window<C, R, K>(fir: &mut PolyphaseFir<f32, C, R, K>, window: &[f32])
    where
        R: RingBuffer<f32>,
    {
        for &sample in window {
            fir.push(sample);
        }
    }

    fn zero_filled_taps<const N: usize>() -> FixedCircularBuffer<f32, N> {
        let mut taps = FixedCircularBuffer::new();
        crate::storage::zero_fill_ring(&mut taps);
        taps
    }

    /// Test input where sample `k` equals `k`; interpolating it should expose
    /// the selected branch's fractional sample position directly.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    fn ramp(count: usize) -> alloc::vec::Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..count).map(|k| k as f32).collect()
    }

    /// Evaluates one branch on a complex sinusoid and returns its gain magnitude.
    ///
    /// The real-tap PFB is evaluated on the real and imaginary sinusoid
    /// components separately so the test does not require the optional `complex`
    /// feature. A sinusoid is an FIR eigenfunction, so the output magnitude is
    /// the branch's frequency-response magnitude at `cycles_per_sample`; any
    /// phase rotation from delay does not affect the magnitude.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[allow(clippy::cast_precision_loss)]
    fn response_magnitude(
        fir: &mut PolyphaseFirVec<f32>,
        phase: usize,
        cycles_per_sample: f32,
    ) -> f32 {
        let taps_per_phase = fir.taps_per_phase();
        let omega = core::f32::consts::TAU * cycles_per_sample;

        fir.reset_delay_line();
        for index in 0..taps_per_phase {
            fir.push((omega * index as f32).cos());
        }
        let real: f32 = fir.execute(phase);

        fir.reset_delay_line();
        for index in 0..taps_per_phase {
            fir.push((omega * index as f32).sin());
        }
        let imag: f32 = fir.execute(phase);

        (real * real + imag * imag).sqrt()
    }

    fn quadratic_lagrange_branch_coefficients(mu: f32) -> [f32; 3] {
        // Lagrange weights for samples f(0), f(1), f(2), evaluated at x = 1 + mu.
        let x = 1.0 + mu;
        let w0 = (x - 1.0) * (x - 2.0) / 2.0;
        let w1 = -x * (x - 2.0);
        let w2 = x * (x - 1.0) / 2.0;
        // `execute` pairs the oldest input sample with reversed branch coefficients.
        [w2, w1, w0]
    }

    /// Fractional-delay branch length is the full input-rate FIR length, while
    /// the dense prototype length is derived from that branch length.
    #[test]
    fn length_helpers_use_full_branch_tap_count() {
        let taps_per_phase = 5;
        let num_phases = 16;

        assert_eq!(prototype_len(taps_per_phase, num_phases), 65);
        assert_eq!(packed_len(num_phases, taps_per_phase), 80);
    }

    /// Dense prototype packing permits even per-phase lengths and preserves the
    /// phase-major mapping used by signalo's filter-bank storage.
    #[test]
    fn pack_prototype_taps_accepts_even_phase_lengths() {
        let mut coefficients = [0.0; 4];
        pack_prototype_taps(&mut coefficients, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(coefficients, [1.0, 3.0, 2.0, 4.0]);

        let mut fir: PolyphaseFirArray<f32, 4, 2> = PolyphaseFir::from_parts(
            Config {
                num_phases: 2,
                taps_per_phase: 2,
                coefficients,
            },
            zero_filled_taps(),
        );
        let window = [7.0, 11.0];
        push_window(&mut fir, &window);

        let phase0: f32 = fir.execute(0);
        let phase1: f32 = fir.execute(1);
        assert_eq!(phase0, 32.0);
        assert_eq!(phase1, 50.0);
    }

    /// `pack_prototype_taps` can target larger caller-provided storage and
    /// zero-fills unused branch entries.
    #[test]
    fn pack_prototype_taps_zero_fills_extra_capacity() {
        let mut coefficients = [99; 9];
        pack_prototype_taps(&mut coefficients, 3, &[1, 2, 3, 4, 5]);
        assert_eq!(coefficients, [1, 4, 0, 2, 5, 0, 3, 0, 0]);
    }

    /// Borrowed mutable slices can provide coefficient storage, matching
    /// signalo's generic storage model without forcing heap allocation.
    #[test]
    fn packed_coefficients_can_use_caller_owned_storage() {
        let mut coefficients = [0.0; 6];
        pack_prototype_taps(&mut coefficients, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut fir: PolyphaseFir<f32, &mut [f32], _, f32> = PolyphaseFir::from_parts(
            Config {
                num_phases: 2,
                taps_per_phase: 3,
                coefficients: &mut coefficients[..],
            },
            zero_filled_taps::<3>(),
        );
        let window = [7.0, 11.0, 13.0];
        push_window(&mut fir, &window);
        let out: f32 = fir.execute(0);
        assert_eq!(out, 81.0);
    }

    /// Exact Lagrange branches prove that phase `i` maps to the fractional
    /// delay `i / M`.
    ///
    /// This avoids relying on an approximate windowed-sinc prototype. Phase `i`
    /// gets 3-point Lagrange coefficients for `x = 1 + i/M`. Feeding `f(k)=k²`
    /// must therefore return `f(1 + i/M)` exactly for every phase.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn phase_branches_land_at_i_over_m_fractional_delays() {
        const NUM_PHASES: usize = 8;
        const TAPS_PER_PHASE: usize = 3;
        const TOTAL_TAPS: usize = packed_len(NUM_PHASES, TAPS_PER_PHASE);
        let mut coefficients = [0.0; TOTAL_TAPS];
        for phase in 0..NUM_PHASES {
            let mu = phase as f32 / NUM_PHASES as f32;
            let branch = quadratic_lagrange_branch_coefficients(mu);
            let start = phase * TAPS_PER_PHASE;
            coefficients[start..start + TAPS_PER_PHASE].copy_from_slice(&branch);
        }

        let mut fir: PolyphaseFirArray<f32, TOTAL_TAPS, TAPS_PER_PHASE> = PolyphaseFir::from_parts(
            Config {
                num_phases: NUM_PHASES,
                taps_per_phase: TAPS_PER_PHASE,
                coefficients,
            },
            zero_filled_taps(),
        );
        // Samples of f(k)=k^2, so exact Lagrange interpolation should return
        // f(center + phase/P).
        let window = [0.0, 1.0, 4.0];
        push_window(&mut fir, &window);
        let branch_center = (TAPS_PER_PHASE - 1) / 2;
        for phase in 0..NUM_PHASES {
            let out: f32 = fir.execute(phase);
            let mu = phase as f32 / NUM_PHASES as f32;
            let expected = (branch_center as f32 + mu).powi(2);
            assert!(
                (out - expected).abs() < 1e-6,
                "phase {phase} got {out}, want {expected}"
            );
        }
    }

    /// Dense prototype packing works with bundled coefficient types, producing
    /// multiple related outputs from one delay-line traversal.
    #[test]
    fn bundled_taps_produce_two_outputs_in_one_pass() {
        let prototype = [
            TapPair {
                first: 1.0,
                second: 2.0,
            },
            TapPair {
                first: 3.0,
                second: 5.0,
            },
        ];
        let mut coefficients = [TapPair::zero(); 2];
        pack_prototype_taps(&mut coefficients, 1, &prototype);
        let mut fir: PolyphaseFir<f32, [TapPair; 2], _, TapPair> = PolyphaseFir::from_parts(
            Config {
                num_phases: 1,
                taps_per_phase: 2,
                coefficients,
            },
            zero_filled_taps::<2>(),
        );
        let window = [7.0, 11.0];
        push_window(&mut fir, &window);
        let out: TapPair = fir.execute(0);
        assert_eq!(out.first, 32.0);
        assert_eq!(out.second, 57.0);
    }

    /// Bundled coefficients still obey phase selection: selecting phase 1 uses
    /// that phase's bundle and accumulates both parallel outputs independently.
    #[test]
    fn bundled_taps_select_phase_and_run_parallel_filters() {
        let coefficients = [
            TapPair {
                first: 1.0,
                second: 10.0,
            },
            TapPair {
                first: 2.0,
                second: 20.0,
            },
            TapPair {
                first: 3.0,
                second: 30.0,
            },
            TapPair {
                first: 4.0,
                second: 40.0,
            },
            TapPair {
                first: 5.0,
                second: 50.0,
            },
            TapPair {
                first: 6.0,
                second: 60.0,
            },
        ];
        let mut fir: PolyphaseFir<f32, [TapPair; 6], _, TapPair> = PolyphaseFir::from_parts(
            Config {
                num_phases: 2,
                taps_per_phase: 3,
                coefficients,
            },
            zero_filled_taps::<3>(),
        );
        let window = [1.0, 2.0, 3.0];
        push_window(&mut fir, &window);

        let out: TapPair = fir.execute(1);
        // Phase 1 coefficients are [(4, 40), (5, 50), (6, 60)] and are applied
        // in reverse order.
        assert_eq!(out.first, 28.0);
        assert_eq!(out.second, 280.0);
    }

    /// `PolyphaseFirVec::from_prototype_taps` remains the heap-backed
    /// convenience path for packed fractional-delay prototypes.
    #[cfg(feature = "alloc")]
    #[test]
    fn polyphase_fir_vec_from_prototype_taps_uses_same_packing() {
        let mut fir = PolyphaseFirVec::<f32>::from_prototype_taps(2, &[1.0, 2.0, 3.0, 4.0]);
        let window = [7.0, 11.0];
        push_window(&mut fir, &window);
        assert_eq!(fir.execute::<f32>(0), 32.0);
        assert_eq!(fir.execute::<f32>(1), 50.0);
    }

    /// A Kaiser low-pass prototype should preserve DC gain for the
    /// zero-fractional-delay branch. This catches normalization regressions in
    /// `kaiser_lowpass_prototype_vec`.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    fn kaiser_phase_zero_passes_a_constant() {
        let cutoff = 0.4;
        let order = crate::filters::fir::design::kaiser_order(60.0, 0.5 - cutoff);
        let prototype = kaiser_lowpass_prototype_vec(32, order, cutoff);
        assert_eq!(prototype.len(), prototype_len(order.num_taps, 32));
        let mut fir = PolyphaseFirVec::<f64>::from_prototype_taps(32, &prototype);
        // A unity-gain fractional-delay branch must leave a constant input unchanged.
        for _ in 0..fir.taps_per_phase() {
            fir.push(2.5);
        }
        let out: f64 = fir.execute(0);
        assert!((out - 2.5).abs() < 1e-3, "got {out}");
    }

    /// The Kaiser helper scales the dense prototype by `num_phases`, so every
    /// packed branch has approximately unit DC gain after polyphase
    /// decomposition.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    fn kaiser_all_phases_preserve_dc() {
        let cutoff = 0.4;
        let order = crate::filters::fir::design::kaiser_order(60.0, 0.5 - cutoff);
        let prototype = kaiser_lowpass_prototype_vec(16, order, cutoff);
        let mut fir = PolyphaseFirVec::<f64>::from_prototype_taps(16, &prototype);
        // Check every phase branch, not just phase zero, against the same DC input.
        for _ in 0..fir.taps_per_phase() {
            fir.push(1.0);
        }
        for phase in 0..fir.num_phases() {
            let out: f64 = fir.execute(phase);
            assert!((out - 1.0).abs() < 5e-3, "phase {phase} got {out}");
        }
    }

    /// Designs a fractional-delay PFB from user-facing Hz values converted to
    /// input-rate cycles/sample. This proves `cutoff` and `kaiser_order` use the
    /// same units as signalo's regular input-rate low-pass design API.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    fn kaiser_lowpass_accepts_input_rate_hz_converted_design_parameters() {
        let sample_rate_hz = 48_000.0_f32;
        let cutoff_hz = 4_000.0_f32;
        let transition_hz = 2_000.0_f32;
        let attenuation_db = 60.0_f32;
        let num_phases = 32;

        let cutoff = cutoff_hz / sample_rate_hz;
        let width = transition_hz / sample_rate_hz;
        let order = crate::filters::fir::design::kaiser_order(attenuation_db, width);
        let prototype = kaiser_lowpass_prototype_vec(num_phases, order, cutoff);
        let mut fir = PolyphaseFirVec::<f32>::from_prototype_taps(num_phases, &prototype);

        assert_eq!(fir.num_phases(), num_phases);
        assert_eq!(fir.taps_per_phase(), order.num_taps);

        let passband = response_magnitude(&mut fir, 0, 1_000.0 / sample_rate_hz);
        let stopband = response_magnitude(&mut fir, 0, 12_000.0 / sample_rate_hz);
        assert!(
            (passband - 1.0).abs() < 0.02,
            "passband response was {passband}"
        );
        assert!(stopband < 0.01, "stopband response was {stopband}");
    }

    /// Every Kaiser-designed branch should reject a frequency well beyond the
    /// transition band by at least the requested stopband attenuation.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    fn kaiser_all_phases_meet_stopband_attenuation() {
        let cutoff = 4_000.0_f32 / 48_000.0;
        let transition_width = 2_000.0_f32 / 48_000.0;
        let attenuation_db = 60.0_f32;
        let num_phases = 32;
        let stopband_probe = 12_000.0_f32 / 48_000.0;

        let order = crate::filters::fir::design::kaiser_order(attenuation_db, transition_width);
        let prototype = kaiser_lowpass_prototype_vec(num_phases, order, cutoff);
        let mut fir = PolyphaseFirVec::<f32>::from_prototype_taps(num_phases, &prototype);
        let stopband_limit = 10.0_f32.powf(-attenuation_db / 20.0);

        for phase in 0..fir.num_phases() {
            let stopband = response_magnitude(&mut fir, phase, stopband_probe);
            assert!(
                stopband < stopband_limit,
                "phase {phase} stopband response was {stopband}, limit {stopband_limit}"
            );
        }
    }

    /// A linear ramp exposes the fractional position of one approximate Kaiser
    /// branch. The exact Lagrange test above proves the phase mapping. This test
    /// checks the built-in Kaiser design follows the same convention closely
    /// enough for a ramp.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn kaiser_interpolates_a_linear_ramp() {
        let num_phases = 32;
        let cutoff = 0.4;
        let order = crate::filters::fir::design::kaiser_order(60.0, 0.5 - cutoff);
        let prototype = kaiser_lowpass_prototype_vec(num_phases, order, cutoff);
        let mut fir = PolyphaseFirVec::<f32>::from_prototype_taps(num_phases, &prototype);
        let window = ramp(fir.taps_per_phase());
        push_window(&mut fir, &window);
        let phase = num_phases / 4;
        let out: f32 = fir.execute(phase);
        let branch_center = (fir.taps_per_phase() - 1) as f32 / 2.0;
        let expected = branch_center + phase as f32 / num_phases as f32;
        assert!((out - expected).abs() < 0.05, "got {out}, want {expected}");
    }

    #[test]
    fn reset_delay_line_clears_fractional_delay_state() {
        let mut fir = PolyphaseFirArray::<f32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1.0, 3.0, 2.0, 4.0],
        });

        fir.push(10.0);
        fir.push(20.0);
        assert_eq!(fir.execute::<f32>(0), 50.0);

        fir.reset_delay_line();
        assert_eq!(fir.execute::<f32>(0), 0.0);
        assert_eq!(fir.execute::<f32>(1), 0.0);
    }
}
