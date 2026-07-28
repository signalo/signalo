// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Fractional-delay polyphase FIR design helpers.
//!
//! A fractional-delay bank stores `num_phases` differently-timed versions of one
//! filter. A dense prototype kernel is sampled on a grid `num_phases` times
//! finer than the input stream, then split into `num_phases` branches. Selecting
//! a branch selects a fractional sample offset. Every branch is a complete
//! input-rate FIR, exactly one runs per output, and the input and output rates
//! are equal.
//!
//! It is therefore not a rate-changing bank. For rate conversion use
//! [`interpolator`](super::interpolator), [`decimator`](super::decimator), or
//! [`rational_resampler`](super::rational_resampler) instead.
//!
//! The kernel can be any finite bandlimited kernel: a windowed-sinc low-pass for
//! interpolation, or a pulse such as a root-raised cosine for matched filtering.
//! The geometry below is the same in every case. This module supplies only the
//! fractional-delay-specific pieces, the length model and an optional Kaiser
//! low-pass prototype. The rest is existing machinery:
//!
//! 1. Pick `taps_per_phase`, the input-rate length each branch must execute.
//! 2. Compute the dense prototype length with [`prototype_len`].
//! 3. Generate the dense kernel over that many samples, for instance with
//!    [`kaiser_lowpass_prototype`].
//! 4. Split it into raw branches with
//!    [`pack_prototype_taps`](super::filter_bank::pack_prototype_taps), which
//!    zero-pads the tail.
//! 5. Feed samples to [`PolyphaseFir::push`](super::fir::PolyphaseFir::push) and
//!    read a branch with
//!    [`PolyphaseFir::execute`](super::fir::PolyphaseFir::execute).
//!
//! Steps 3 and 4 can share one buffer instead of two. Allocate the packed
//! length, generate the dense kernel into `taps[..dense_len]`, then call
//! [`pack_prototype_taps_in_place`](super::filter_bank::pack_prototype_taps_in_place)
//! over the whole slice. That reorder costs more than the linear two-buffer
//! route, so prefer two buffers unless the second allocation is unacceptable.
//! The subslice is required because the design helpers assert an exact
//! prototype length.
//!
//! Throughout, `d0` is the bulk group delay every branch shares, in input
//! samples:
//!
//! ```text
//! d0 = (taps_per_phase - 1) / 2
//! ```
//!
//! This is real division, not integer division. `d0` is a half-integer when
//! `taps_per_phase` is even, so do not compute it with integer arithmetic.
//!
//! # Phase convention
//!
//! ```text
//! execute(q) evaluates the input q / num_phases samples later than execute(0)
//! ```
//!
//! This is a property of the whole pipeline above, not of `execute` on its own.
//! It holds when the kernel is sampled on the `num_phases`-times finer grid,
//! split with the raw stride packing, and stored unreversed or at most
//! conjugated, as described under [Tap order](#tap-order). Storing reversed
//! branches instead inverts the sign to `-q / num_phases`, which is the failure
//! mode when porting a packed table from a library that pre-reverses its
//! subfilters.
//!
//! The offset is given as an evaluation time rather than as a delay because
//! three related quantities carry different signs. As `q` increases:
//!
//! - the branch samples farther forward in the input,
//! - a feature in the input appears at an earlier output index,
//! - the effective fractional delay decreases.
//!
//! Branch 0 is not delay-free. Every branch carries the same bulk group delay
//! `d0`, so zero fractional offset does not mean zero causal delay. Combining
//! the two gives the input position an output refers to: after pushing up to and
//! including `x[n]`, `execute(q)` approximates the input at
//!
//! ```text
//! n - d0 + q / num_phases
//! ```
//!
//! # Choosing a phase
//!
//! A fractional-delay bank is usually driven by timing recovery, for instance
//! alongside [`LoopFilter`](crate::filters::iir::loop_filter::LoopFilter). An
//! upstream estimator reports where a symbol instant or correlation peak really
//! falls, which in general lies between two input samples. Split that estimate
//! into an integer and a fractional part:
//!
//! ```text
//! position = floor_index + tau,   tau in [0, 1)
//! ```
//!
//! The integer part chooses which input window to evaluate. The fractional part
//! chooses the branch:
//!
//! ```text
//! q = round(tau * num_phases)
//! ```
//!
//! Rounding can produce `q == num_phases`. That is not the last branch of the
//! current sample, it is branch 0 of the next one, so carry it:
//!
//! ```text
//! if q == num_phases { floor_index += 1; q = 0; }
//! ```
//!
//! Do not clamp `q` to `num_phases - 1`. Clamping folds the top of the last
//! phase interval onto the wrong input sample and biases timing by up to
//! `1 / num_phases`.
//!
//! # Length model
//!
//! Start from the input-rate filter length each branch must execute, and use it
//! as `taps_per_phase`. It is an ordinary FIR length, chosen the same way as for
//! a non-polyphase filter: from a transition width and attenuation for a
//! low-pass, or from a pulse span and roll-off for a shaped pulse. For example
//! [`kaiser_order`](crate::filters::fir::design::kaiser_order) returns that
//! length in `order.num_taps`.
//!
//! `taps_per_phase` must be at least 2. A single-coefficient branch is a scalar
//! gain rather than a filter, so it cannot evaluate the input at a fractional
//! offset.
//!
//! Branch 0 is then the ordinary input-rate filter of `taps_per_phase` taps, and
//! the other branches are fractional shifts of it. The dense prototype length
//! follows from [`prototype_len`]:
//!
//! ```text
//! dense_len = num_phases * (taps_per_phase - 1) + 1
//! packed    = num_phases * taps_per_phase
//! padding   = num_phases - 1
//! ```
//!
//! The dense length multiplies the filter *order*, `taps_per_phase - 1`, not
//! the tap count. Sizing the dense prototype as `num_phases * taps_per_phase` is the
//! common mistake: that is the packed rectangle, not the meaningful kernel
//! length, and it treats each branch as a decimated subfilter rather than a full
//! input-rate FIR.
//!
//! Odd and even `taps_per_phase` are both accepted, and the parity only moves
//! `d0`. Odd puts `d0` on an integer sample, so branch 0 is a whole-sample
//! delay. Even puts `d0` half a sample off, so branch 0 already carries half a
//! sample of fractional delay. Neither is wrong, but the offset has to be
//! accounted for in either case.
//! [`kaiser_order`](crate::filters::fir::design::kaiser_order) commonly returns
//! an even length.
//!
//! # Tap order
//!
//! [`PolyphaseFir`](super::fir::PolyphaseFir) is a convolution-style executor:
//! it applies each branch's stored taps in reverse chronological order, so the
//! first stored coefficient meets the newest sample. Coefficients are supplied
//! as an impulse response, so
//! [`pack_prototype_taps`](super::filter_bank::pack_prototype_taps) output feeds
//! the executor directly.
//!
//! A real kernel requires no branch reversal. Reversing the raw branches to
//! obtain a chronological correlation template, then reversing again to prepare
//! convolution storage, cancels out and leaves only the reversal the executor
//! already performs.
//!
//! Storage order and chronological order therefore run opposite ways, which
//! matters when inspecting a packed table directly. For a symmetric kernel,
//! stored branch `q` holds the kernel sampled at `k - d0 + q / num_phases` for
//! stored index `k`, so its peak sits at stored index `d0 - q / num_phases` and
//! moves toward index 0 as `q` grows. The effective chronological coefficients
//! peak at `d0 + q / num_phases` and move in the opposite direction, which is
//! the direction `execute` follows. Each branch's padding zero shows the same
//! inversion: stored last, applied against the oldest sample. A plot of the raw
//! branches does not show the order the executor applies them in.
//!
//! For a complex kernel used as a matched filter, store the conjugated raw
//! branches, `conj(prototype[k * num_phases + q])`. The executor supplies the
//! reversal, so conjugation is the only preparation needed. Do not reverse the
//! dense prototype before packing: reversal is positional and does not commute
//! with the stride split, so it moves taps between branches. Conjugation is
//! elementwise and commutes, so it may be applied before or after packing.

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
/// Panics when `num_phases` is zero, when `order.num_taps` is less than two,
/// when `prototype.len()` does not equal
/// `prototype_len(order.num_taps, num_phases)`, when `cutoff` is outside
/// `(0, 0.5)`, or when `num_phases` cannot be represented as `T`.
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
    // A single-coefficient branch is a scalar gain rather than a filter, so it
    // cannot evaluate the input at a fractional offset for any kernel. The
    // mechanical symptom is that `prototype_len(1, num_phases)` is 1 for every
    // `num_phases`, which the windowed-sinc designer rejects as degenerate.
    assert!(
        order.num_taps >= 2,
        "fractional-delay bank needs at least 2 taps per phase"
    );
    assert_eq!(
        prototype.len(),
        prototype_len(order.num_taps, num_phases),
        "fractional-delay prototype length must match order and phase count"
    );
    let Some(phases) = T::from(num_phases) else {
        panic!("fractional-delay phase count must be representable");
    };
    // Bound the input-rate cutoff here. `lowpass_with_beta` only ever sees
    // `cutoff / num_phases`, so on its own it would accept any input-rate value
    // below `0.5 * num_phases`, well above the input Nyquist limit.
    let half = T::from(0.5).expect("0.5 is representable");
    assert!(
        cutoff > T::zero() && cutoff < half,
        "fractional-delay cutoff must be in (0, 0.5) input-rate cycles per sample"
    );

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
    // Checked here as well as in `kaiser_lowpass_prototype`, because the length
    // computation below would otherwise report `prototype_len`'s message first.
    assert!(
        order.num_taps >= 2,
        "fractional-delay bank needs at least 2 taps per phase"
    );
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
        crate::storage::zero_filled_fixed_ring::<f32, N>()
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

        for index in 0..taps_per_phase {
            fir.push((omega * index as f32).cos());
        }
        let real: f32 = fir.execute(phase);

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
        // The rectangle is wider than the dense prototype by exactly one slot
        // per branch beyond the first.
        assert_eq!(
            packed_len(num_phases, taps_per_phase) - prototype_len(taps_per_phase, num_phases),
            num_phases - 1,
            "padding must be num_phases - 1"
        );
    }

    /// Raw polyphase decomposition takes every `num_phases`-th dense tap, so
    /// branch `q` holds `dense[k * num_phases + q]`.
    ///
    /// With the dense length from [`prototype_len`], branch 0 has
    /// `taps_per_phase` meaningful taps and every later branch has one fewer,
    /// with its single padding zero in the last stored position.
    #[test]
    fn raw_branch_taps_follow_the_dense_stride() {
        const NUM_PHASES: usize = 4;
        const TAPS_PER_PHASE: usize = 5;
        let dense_len = prototype_len(TAPS_PER_PHASE, NUM_PHASES);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let dense: [i32; 17] = core::array::from_fn(|j| j as i32 + 1);
        assert_eq!(dense.len(), dense_len);

        let mut packed = [0_i32; packed_len(NUM_PHASES, TAPS_PER_PHASE)];
        pack_prototype_taps(&mut packed, NUM_PHASES, &dense);

        for phase in 0..NUM_PHASES {
            let branch = &packed[phase * TAPS_PER_PHASE..(phase + 1) * TAPS_PER_PHASE];
            let mut meaningful = 0;
            for (k, tap) in branch.iter().enumerate() {
                let index = k * NUM_PHASES + phase;
                let expected = if index < dense_len { dense[index] } else { 0 };
                assert_eq!(*tap, expected, "branch {phase} tap {k}");
                if index < dense_len {
                    meaningful += 1;
                }
            }
            if phase == 0 {
                assert_eq!(
                    meaningful, TAPS_PER_PHASE,
                    "branch 0 is the full-length branch"
                );
            } else {
                assert_eq!(
                    meaningful,
                    TAPS_PER_PHASE - 1,
                    "branch {phase} is one tap short"
                );
                assert_eq!(
                    branch[TAPS_PER_PHASE - 1],
                    0,
                    "branch {phase} pads on the right"
                );
            }
        }
    }

    /// `prototype_len` is a `const fn`, so it must be usable in a const context.
    #[test]
    fn prototype_len_is_const_evaluable() {
        const LEN: usize = prototype_len(5, 16);
        assert_eq!(LEN, 65);
    }

    #[test]
    #[should_panic(expected = "taps per phase must be nonzero")]
    fn prototype_len_zero_taps_per_phase_panics() {
        let _ = prototype_len(0, 4);
    }

    #[test]
    #[should_panic(expected = "needs phase branches")]
    fn prototype_len_zero_phases_panics() {
        let _ = prototype_len(4, 0);
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

    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    #[should_panic(expected = "prototype length must match order and phase count")]
    fn kaiser_lowpass_prototype_length_mismatch_panics() {
        let order = crate::filters::fir::design::kaiser_order(60.0_f32, 0.1);
        let mut prototype = alloc::vec![0.0_f32; prototype_len(order.num_taps, 4) + 1];
        super::kaiser_lowpass_prototype(&mut prototype, 4, order, 0.2);
    }

    /// The input-rate cutoff is bounded here rather than only after division by
    /// `num_phases`, which would have admitted anything below `0.5 * num_phases`.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    #[should_panic(expected = "cutoff must be in (0, 0.5) input-rate cycles per sample")]
    fn kaiser_lowpass_prototype_above_input_nyquist_panics() {
        let order = crate::filters::fir::design::kaiser_order(60.0_f32, 0.1);
        let _ = kaiser_lowpass_prototype_vec(4, order, 0.9_f32);
    }

    /// Every branch evaluates the input `phase / num_phases` samples later than
    /// branch zero, checked on the shipped Kaiser design rather than a synthetic
    /// bank. A ramp makes the offset read directly as an output difference.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn every_branch_evaluates_its_phase_later_than_branch_zero() {
        let num_phases = 8;
        let order = crate::filters::fir::design::kaiser_order(80.0_f64, 0.10);
        let prototype = kaiser_lowpass_prototype_vec(num_phases, order, 0.20);
        let mut fir = PolyphaseFirVec::<f64>::from_prototype_taps(num_phases, &prototype);
        for index in 0..fir.taps_per_phase() {
            fir.push(index as f64);
        }

        let branch_zero: f64 = fir.execute(0);
        for phase in 0..num_phases {
            let out: f64 = fir.execute(phase);
            let offset = out - branch_zero;
            let expected = phase as f64 / num_phases as f64;
            assert!(
                (offset - expected).abs() < 2e-3,
                "phase {phase} offset {offset}, want {expected}"
            );
        }
    }

    /// A waveform arriving `phase / num_phases` samples late is realigned by
    /// branch `phase`, so every branch reports the same value for its own shift.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn each_branch_realigns_its_own_fractional_shift() {
        let num_phases = 8;
        let order = crate::filters::fir::design::kaiser_order(80.0_f64, 0.10);
        let prototype = kaiser_lowpass_prototype_vec(num_phases, order, 0.20);
        let mut fir = PolyphaseFirVec::<f64>::from_prototype_taps(num_phases, &prototype);
        let taps_per_phase = fir.taps_per_phase();
        let branch_center = (taps_per_phase - 1) as f64 / 2.0;
        // Smooth and well inside the passband, so the interpolation error stays small.
        let wave = |t: f64| (0.10 * core::f64::consts::TAU * t).cos();

        for phase in 0..num_phases {
            let shift = phase as f64 / num_phases as f64;
            for index in 0..taps_per_phase {
                fir.push(wave(index as f64 - shift));
            }
            let out: f64 = fir.execute(phase);
            assert!(
                (out - wave(branch_center)).abs() < 5e-3,
                "phase {phase} realigned to {out}, want {}",
                wave(branch_center)
            );
        }
    }

    /// Packing in place reuses one buffer for both the dense prototype and the
    /// packed table, and must agree with the two-buffer route tap for tap.
    ///
    /// The design helpers assert an exact prototype length, so the kernel is
    /// generated into `taps[..dense_len]` rather than the whole packed buffer.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    fn in_place_packing_matches_the_two_buffer_route() {
        use super::kaiser_lowpass_prototype;
        use crate::filters::fir::polyphase::filter_bank::pack_prototype_taps_in_place;

        const NUM_PHASES: usize = 8;
        let order = crate::filters::fir::design::kaiser_order(60.0_f32, 0.10);
        let dense_len = prototype_len(order.num_taps, NUM_PHASES);
        let packed = packed_len(NUM_PHASES, order.num_taps);
        assert!(
            dense_len < packed,
            "the dense prototype must fit with room to pad"
        );

        // two buffers: design into one, pack into the other
        let mut dense = alloc::vec![0.0_f32; dense_len];
        kaiser_lowpass_prototype(&mut dense, NUM_PHASES, order, 0.20);
        let mut two_buffer = alloc::vec![0.0_f32; packed];
        pack_prototype_taps(&mut two_buffer, NUM_PHASES, &dense);

        // one buffer: design into its front, then reorder the whole slice
        let mut one_buffer = alloc::vec![0.0_f32; packed];
        kaiser_lowpass_prototype(&mut one_buffer[..dense_len], NUM_PHASES, order, 0.20);
        pack_prototype_taps_in_place(&mut one_buffer, NUM_PHASES, dense_len);

        assert_eq!(one_buffer, two_buffer);
    }

    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    #[should_panic(expected = "needs phase branches")]
    fn kaiser_lowpass_prototype_zero_phases_panics() {
        let order = crate::filters::fir::design::kaiser_order(60.0_f32, 0.1);
        let _ = kaiser_lowpass_prototype_vec(0, order, 0.2);
    }

    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    #[should_panic(expected = "needs at least 2 taps per phase")]
    fn kaiser_lowpass_prototype_zero_taps_panics() {
        let order = crate::filters::fir::design::KaiserOrder {
            num_taps: 0,
            beta: 4.0_f32,
        };
        let _ = kaiser_lowpass_prototype_vec(4, order, 0.2);
    }

    /// One tap per phase cannot represent a fractional delay for any kernel,
    /// since a single-coefficient branch is a scalar gain. It is rejected at
    /// this boundary rather than several layers down in the windowed-sinc
    /// designer, which would report its own degenerate case instead.
    #[cfg(all(feature = "alloc", any(feature = "libm", feature = "std")))]
    #[test]
    #[should_panic(expected = "needs at least 2 taps per phase")]
    fn kaiser_lowpass_prototype_one_tap_panics() {
        let order = crate::filters::fir::design::KaiserOrder {
            num_taps: 1,
            beta: 4.0_f32,
        };
        let _ = kaiser_lowpass_prototype_vec(4, order, 0.2);
    }
}
