// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Polyphase FIR filter bank.
//!
//! See [`PolyphaseFilterBank`] for coefficient ordering and phase-major storage
//! layout.

use core::ops::{Add, Mul};

use num_traits::Zero;

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Reset, WithConfig,
};

/// The polyphase filter bank's configuration.
///
/// Holds the phase-major coefficient storage `C`, which must implement
/// [`AsSlice<K>`] on relevant impls. Use [`PolyphaseFilterBankArray`] for
/// stack-allocated coefficients, `PolyphaseFilterBankVec` for heap-allocated
/// coefficients, or [`PolyphaseFilterBankRefMut`] for caller-owned coefficient
/// storage.
#[derive(Clone, Debug)]
pub struct Config<C> {
    /// Number of polyphase branches.
    pub num_phases: usize,
    /// Number of coefficients in each phase branch.
    pub taps_per_phase: usize,
    /// Phase-major coefficients in normal FIR order within each phase.
    ///
    /// For phase `p`, the phase coefficient slice is:
    ///
    /// ```text
    /// coefficients[p * taps_per_phase .. (p + 1) * taps_per_phase]
    /// ```
    pub coefficients: C,
}

/// A polyphase FIR filter bank.
///
/// This type stores phase coefficients and can evaluate a selected phase against
/// caller-provided sample history. It does not own input history, advance time,
/// or implement rate conversion by itself. Stateful users such as rational
/// resamplers own delay lines and use [`Self::execute`] to evaluate selected
/// phases.
///
/// # Coefficient ordering
///
/// Before packing, coefficients may be represented as a dense prototype: a
/// single ordinary FIR coefficient sequence:
///
/// ```text
/// h[0], h[1], h[2], h[3], ...
/// ```
///
/// The packed representation is always rectangular: each phase has exactly
/// `taps_per_phase` coefficients, with zero padding in shorter final branches
/// when a dense prototype does not fill the rectangle. It stores one contiguous
/// slice per phase. For `P = num_phases`, phase `p` stores every `P`-th
/// prototype tap:
///
/// ```text
/// phase p: h[p], h[p + P], h[p + 2P], ...
/// ```
///
/// [`Self::execute`] accepts sample history in oldest-to-newest order. It pairs
/// coefficients with samples in convolution order: for the selected phase, the
/// first phase coefficient is applied to the newest sample.
///
/// The bank does not reverse, conjugate, or otherwise reinterpret
/// coefficients. For correlation or matched-filter behavior, prepare those FIR
/// coefficients before phase packing. That means time-reversing the template
/// coefficients; for complex templates, conjugate them as well.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`PolyphaseFilterBankArray<K, N>`] — stack-allocated, `no_std`-friendly.
#[cfg_attr(
    feature = "alloc",
    doc = "- [`PolyphaseFilterBankVec<K>`] — heap-allocated, requires the `alloc` feature."
)]
#[cfg_attr(
    not(feature = "alloc"),
    doc = "- `PolyphaseFilterBankVec<K>` — heap-allocated, requires the `alloc` feature."
)]
/// - [`PolyphaseFilterBankRefMut<'_, K>`] — caller-owned coefficient storage.
///
/// # Complexity
///
/// - **Time per [`Self::execute`] call:** O(H/P), where H is the total tap count
///   and P is the number of phases; each call convolves one phase branch of H/P coefficients.
/// - **Space:** O(H); stores all H coefficients, but owns no delay-line state.
#[derive(Clone, Debug)]
pub struct PolyphaseFilterBank<C> {
    config: Config<C>,
}

/// A polyphase filter bank backed by a const-generic coefficient array.
pub type PolyphaseFilterBankArray<K, const N: usize> = PolyphaseFilterBank<[K; N]>;

/// A polyphase filter bank backed by heap-allocated coefficients.
#[cfg(feature = "alloc")]
pub type PolyphaseFilterBankVec<K> = PolyphaseFilterBank<alloc::vec::Vec<K>>;

/// A polyphase filter bank that borrows caller-owned coefficient storage.
pub type PolyphaseFilterBankRefMut<'a, K> = PolyphaseFilterBank<&'a mut [K]>;

/// Returns the length of a packed polyphase coefficient slice.
///
/// The packed slice stores `num_phases` contiguous phase branches, with
/// `taps_per_phase` coefficients in each branch:
///
/// ```text
/// [phase 0 taps..., phase 1 taps..., ..., phase P - 1 taps...]
/// ```
///
/// The total length is `num_phases * taps_per_phase`, which is the storage
/// length expected by [`PolyphaseFilterBank`].
///
/// # Panics
///
/// Panics when `num_phases` or `taps_per_phase` is zero, or if the product
/// overflows `usize`.
#[must_use]
pub const fn packed_len(num_phases: usize, taps_per_phase: usize) -> usize {
    assert!(
        num_phases > 0,
        "PolyphaseFilterBank: number of phases must be > 0"
    );
    assert!(
        taps_per_phase > 0,
        "PolyphaseFilterBank: taps per phase must be > 0"
    );
    let Some(len) = num_phases.checked_mul(taps_per_phase) else {
        panic!("PolyphaseFilterBank: num_phases * taps_per_phase overflowed");
    };
    len
}

/// Returns the per-phase tap count needed to pack `prototype_len` dense
/// prototype taps.
///
/// Dense prototype taps are distributed across phases by tap index. This returns
/// `ceil(prototype_len / num_phases)`, the number of taps needed by each phase
/// after the shorter final branches are zero-padded to a rectangular packed
/// layout.
///
/// # Panics
///
/// Panics when `num_phases` or `prototype_len` is zero.
#[must_use]
pub const fn taps_per_phase_for_prototype_len(num_phases: usize, prototype_len: usize) -> usize {
    assert!(
        num_phases > 0,
        "PolyphaseFilterBank: number of phases must be > 0"
    );
    assert!(
        prototype_len > 0,
        "PolyphaseFilterBank: tap count must be > 0"
    );
    prototype_len.div_ceil(num_phases)
}

/// Returns the packed coefficient length needed to store `prototype_len` dense
/// prototype taps.
///
/// This is `num_phases * ceil(prototype_len / num_phases)`, i.e. the dense
/// prototype length rounded up to a rectangular phase-major coefficient layout.
///
/// # Panics
///
/// Panics when `num_phases` or `prototype_len` is zero, or if the packed length
/// overflows `usize`.
#[must_use]
pub const fn packed_len_for_prototype_len(num_phases: usize, prototype_len: usize) -> usize {
    let taps_per_phase = taps_per_phase_for_prototype_len(num_phases, prototype_len);
    packed_len(num_phases, taps_per_phase)
}

/// Packs dense prototype taps into rectangular phase-major coefficient storage.
///
/// Dense prototype taps are ordered as:
///
/// ```text
/// h[0], h[1], h[2], h[3], ...
/// ```
///
/// For `num_phases = P`, phase `p` receives every `P`-th tap:
///
/// ```text
/// phase p: h[p], h[p + P], h[p + 2P], ...
/// ```
///
/// The output `coefficients` slice uses phase-major storage:
///
/// ```text
/// [phase 0 taps..., phase 1 taps..., ..., phase P - 1 taps...]
/// ```
///
/// `coefficients.len()` chooses the rectangular packed geometry. It must be a
/// nonzero multiple of `num_phases`. Any packed entries not covered by
/// `prototype` are zero-filled, so callers may preallocate a larger fixed
/// per-phase length than the minimum required by the prototype.
///
/// # Tap interpretation
///
/// This is only a storage transform; it does not reverse, conjugate, or
/// otherwise reinterpret `prototype`. When the packed taps are used with
/// [`PolyphaseFilterBank::execute`], `prototype` must already use that method's
/// convolution coefficient order. For correlation or matched filtering through
/// that executor, time-reverse the template first; for complex templates,
/// conjugate it as well. For other executors, prepare the prototype in the order
/// they expect before packing.
///
/// # Panics
///
/// Panics when `num_phases` is zero, `prototype` is empty, `coefficients` is
/// empty, `coefficients.len()` is not a multiple of `num_phases`, or
/// `prototype.len() > coefficients.len()`.
pub fn pack_prototype_taps<K>(coefficients: &mut [K], num_phases: usize, prototype: &[K])
where
    K: Clone + Zero,
{
    assert!(
        num_phases > 0,
        "PolyphaseFilterBank: number of phases must be > 0"
    );
    assert!(
        !prototype.is_empty(),
        "PolyphaseFilterBank: tap count must be > 0"
    );
    assert!(
        !coefficients.is_empty(),
        "PolyphaseFilterBank: coefficient storage must be nonempty"
    );
    assert!(
        coefficients.len().is_multiple_of(num_phases),
        "PolyphaseFilterBank: coefficient count must be a multiple of num_phases"
    );
    assert!(
        prototype.len() <= coefficients.len(),
        "PolyphaseFilterBank: prototype length must fit coefficient storage"
    );

    let taps_per_phase = coefficients.len() / num_phases;
    crate::math::matrix_transpose_padded(
        coefficients,
        prototype,
        num_phases,
        taps_per_phase,
        K::zero(),
    );
}

/// Packs dense prototype taps into rectangular phase-major storage in place.
///
/// `coefficients[..prototype_len]` is treated as the dense prototype. The tail
/// `coefficients[prototype_len..]` is zero-filled, then the full slice is
/// reordered into the phase-major layout described by [`pack_prototype_taps`].
/// See [`pack_prototype_taps`] for tap ordering and interpretation.
///
/// `coefficients.len()` chooses the rectangular packed geometry, with the same
/// rules as [`pack_prototype_taps`]. Use [`packed_len_for_prototype_len`] to get
/// the minimum buffer length for `prototype_len`, or a larger multiple of
/// `num_phases` to use a fixed per-phase length.
///
/// # Panics
///
/// Panics when `num_phases` or `prototype_len` is zero, `coefficients` is empty,
/// `coefficients.len()` is not a multiple of `num_phases`, or
/// `prototype_len > coefficients.len()`.
pub fn pack_prototype_taps_in_place<K>(
    coefficients: &mut [K],
    num_phases: usize,
    prototype_len: usize,
) where
    K: Clone + Zero,
{
    assert!(
        num_phases > 0,
        "PolyphaseFilterBank: number of phases must be > 0"
    );
    assert!(
        prototype_len > 0,
        "PolyphaseFilterBank: tap count must be > 0"
    );
    assert!(
        !coefficients.is_empty(),
        "PolyphaseFilterBank: coefficient storage must be nonempty"
    );
    assert!(
        coefficients.len().is_multiple_of(num_phases),
        "PolyphaseFilterBank: coefficient count must be a multiple of num_phases"
    );
    assert!(
        prototype_len <= coefficients.len(),
        "PolyphaseFilterBank: prototype length must fit coefficient storage"
    );

    let taps_per_phase = coefficients.len() / num_phases;
    coefficients[prototype_len..].fill(K::zero());
    crate::math::matrix_transpose_in_place(coefficients, num_phases, taps_per_phase);
}

impl<C> PolyphaseFilterBank<C> {
    /// Creates a [`PolyphaseFilterBank`] from phase-major coefficient storage.
    ///
    /// # Panics
    ///
    /// Panics if `num_phases` or `taps_per_phase` is zero, or if
    /// `coefficients.len() != num_phases * taps_per_phase`.
    pub fn from_parts<K>(config: Config<C>) -> Self
    where
        C: AsSlice<K>,
    {
        assert!(
            config.num_phases > 0,
            "PolyphaseFilterBank: number of phases must be > 0"
        );
        assert!(
            config.taps_per_phase > 0,
            "PolyphaseFilterBank: taps per phase must be > 0"
        );

        let expected_taps = packed_len(config.num_phases, config.taps_per_phase);

        assert_eq!(
            config.coefficients.as_slice().len(),
            expected_taps,
            "PolyphaseFilterBank: coefficient count must equal num_phases * taps_per_phase"
        );

        Self { config }
    }

    /// Returns the number of polyphase branches.
    #[must_use]
    pub fn num_phases(&self) -> usize {
        self.config.num_phases
    }

    /// Returns the number of coefficients in each phase branch.
    #[must_use]
    pub fn taps_per_phase(&self) -> usize {
        self.config.taps_per_phase
    }

    /// Returns the total number of coefficients.
    #[must_use]
    pub fn total_taps(&self) -> usize {
        self.num_phases() * self.taps_per_phase()
    }

    /// Returns the phase-major coefficient slice for `phase`.
    ///
    /// # Panics
    ///
    /// Panics if `phase >= self.num_phases()`.
    #[must_use]
    pub fn phase_coefficients<K>(&self, phase: usize) -> &[K]
    where
        C: AsSlice<K>,
    {
        assert!(
            phase < self.num_phases(),
            "PolyphaseFilterBank: phase index out of range"
        );

        let start = phase * self.taps_per_phase();
        let end = start + self.taps_per_phase();
        &self.config.coefficients.as_slice()[start..end]
    }

    /// Evaluates `phase` against caller-provided sample history.
    ///
    /// The `taps` iterator must yield samples in oldest-to-newest order. Phase
    /// coefficients are paired so the first coefficient in the selected phase,
    /// `h[phase]`, multiplies the newest sample.
    ///
    /// # Panics
    ///
    /// Panics if `phase >= self.num_phases()`, or if the number of samples
    /// yielded by `taps` does not equal `self.taps_per_phase()`.
    #[must_use]
    pub fn execute<'a, T, K, O, I>(&self, phase: usize, taps: I) -> O
    where
        T: 'a + Clone + Mul<K, Output = O>,
        K: Clone,
        C: AsSlice<K>,
        O: Zero + Add<Output = O>,
        I: IntoIterator<Item = &'a T>,
    {
        let coefficients = self.phase_coefficients(phase);
        let mut taps = taps.into_iter();
        let mut count = 0;
        let output =
            taps.by_ref()
                .zip(coefficients.iter().rev())
                .fold(O::zero(), |sum, (state, coeff)| {
                    count += 1;
                    sum + ((*state).clone() * (*coeff).clone())
                });

        assert!(
            count == self.taps_per_phase() && taps.next().is_none(),
            "PolyphaseFilterBank: taps count must equal taps_per_phase"
        );

        output
    }
}

#[cfg(feature = "alloc")]
impl<K> PolyphaseFilterBankVec<K>
where
    K: Clone + Zero,
{
    /// Creates a heap-backed filter bank from dense prototype coefficients.
    ///
    /// This convenience constructor allocates coefficient storage. `prototype`
    /// is provided as a dense prototype, not phase-major storage. It is
    /// zero-padded at the tail as needed, then packed using the phase-major
    /// ordering described on [`PolyphaseFilterBank`].
    ///
    /// See [`pack_prototype_taps`] for tap ordering and interpretation.
    ///
    /// # Panics
    ///
    /// Panics if `num_phases` is zero, `prototype` is empty, or the padded
    /// coefficient count overflows `usize`.
    #[must_use]
    pub fn from_prototype_taps(num_phases: usize, prototype: &[K]) -> Self {
        assert!(
            num_phases > 0,
            "PolyphaseFilterBank: number of phases must be > 0"
        );
        assert!(
            !prototype.is_empty(),
            "PolyphaseFilterBank: tap count must be > 0"
        );

        let taps_per_phase = taps_per_phase_for_prototype_len(num_phases, prototype.len());
        let total_taps = packed_len(num_phases, taps_per_phase);
        let mut coefficients = alloc::vec![K::zero(); total_taps];
        pack_prototype_taps(&mut coefficients, num_phases, prototype);

        Self::from_parts(Config {
            num_phases,
            taps_per_phase,
            coefficients,
        })
    }
}

impl<C> ConfigTrait for PolyphaseFilterBank<C> {
    type Config = Config<C>;
}

impl<K, const N: usize> WithConfig for PolyphaseFilterBankArray<K, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self::from_parts(config)
    }
}

impl<C> ConfigRef for PolyphaseFilterBank<C> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<C> ConfigClone for PolyphaseFilterBank<C>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<C> HasGuts for PolyphaseFilterBank<C> {
    type Guts = Config<C>;
}

impl<C> FromGuts for PolyphaseFilterBank<C> {
    fn from_guts(guts: Self::Guts) -> Self {
        Self { config: guts }
    }
}

impl<C> IntoGuts for PolyphaseFilterBank<C> {
    fn into_guts(self) -> Self::Guts {
        self.config
    }
}

impl<C> Reset for PolyphaseFilterBank<C> {
    fn reset(self) -> Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pack_prototype_taps, pack_prototype_taps_in_place, packed_len,
        packed_len_for_prototype_len, taps_per_phase_for_prototype_len, Config,
        PolyphaseFilterBank, PolyphaseFilterBankArray, PolyphaseFilterBankRefMut,
    };
    use crate::filters::fir::polyphase::test_support::Pair;
    use crate::traits::WithConfig;

    #[test]
    fn packed_lengths_match_phase_geometry() {
        assert_eq!(packed_len(3, 2), 6);
        assert_eq!(taps_per_phase_for_prototype_len(3, 5), 2);
        assert_eq!(packed_len_for_prototype_len(3, 5), 6);
        assert_eq!(taps_per_phase_for_prototype_len(4, 8), 2);
        assert_eq!(packed_len_for_prototype_len(4, 8), 8);
    }

    #[test]
    fn pack_prototype_taps_reorders_and_pads() {
        let mut coefficients = [99; 6];

        pack_prototype_taps(&mut coefficients, 3, &[1, 2, 3, 4, 5]);

        assert_eq!(coefficients, [1, 4, 2, 5, 3, 0]);
    }

    #[test]
    fn pack_prototype_taps_handles_exact_rectangles() {
        let mut coefficients = [99; 6];

        pack_prototype_taps(&mut coefficients, 3, &[1, 2, 3, 4, 5, 6]);

        assert_eq!(coefficients, [1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn pack_prototype_taps_allows_larger_geometry() {
        let mut coefficients = [99; 9];

        pack_prototype_taps(&mut coefficients, 3, &[1, 2, 3, 4, 5]);

        assert_eq!(coefficients, [1, 4, 0, 2, 5, 0, 3, 0, 0]);
    }

    #[test]
    fn pack_prototype_taps_in_place_reorders_and_pads() {
        let mut coefficients = [1, 2, 3, 4, 5, 99];

        pack_prototype_taps_in_place(&mut coefficients, 3, 5);

        assert_eq!(coefficients, [1, 4, 2, 5, 3, 0]);
    }

    #[test]
    fn pack_prototype_taps_in_place_handles_exact_rectangles() {
        let mut coefficients = [1, 2, 3, 4, 5, 6];

        pack_prototype_taps_in_place(&mut coefficients, 3, 6);

        assert_eq!(coefficients, [1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn pack_prototype_taps_in_place_allows_larger_geometry() {
        let mut coefficients = [1, 2, 3, 4, 5, 99, 98, 97, 96];

        pack_prototype_taps_in_place(&mut coefficients, 3, 5);

        assert_eq!(coefficients, [1, 4, 0, 2, 5, 0, 3, 0, 0]);
    }

    #[test]
    fn from_parts_accepts_phase_major_coefficients() {
        let bank = PolyphaseFilterBankArray::<i32, 6>::from_parts(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });

        assert_eq!(bank.num_phases(), 3);
        assert_eq!(bank.taps_per_phase(), 2);
        assert_eq!(bank.total_taps(), 6);
        assert_eq!(bank.phase_coefficients(0), &[1, 4]);
        assert_eq!(bank.phase_coefficients(1), &[2, 5]);
        assert_eq!(bank.phase_coefficients(2), &[3, 6]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn from_prototype_taps_reorders_and_pads() {
        use super::PolyphaseFilterBankVec;

        let bank = PolyphaseFilterBankVec::<i32>::from_prototype_taps(3, &[1, 2, 3, 4, 5]);

        assert_eq!(bank.num_phases(), 3);
        assert_eq!(bank.taps_per_phase(), 2);
        assert_eq!(bank.config.coefficients.as_slice(), &[1, 4, 2, 5, 3, 0]);
        assert_eq!(bank.phase_coefficients(0), &[1, 4]);
        assert_eq!(bank.phase_coefficients(1), &[2, 5]);
        assert_eq!(bank.phase_coefficients(2), &[3, 0]);
    }

    #[test]
    fn with_config_accepts_phase_major_coefficients() {
        let bank = PolyphaseFilterBank::<[i32; 4]>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });

        assert_eq!(bank.phase_coefficients(0), &[1, 3]);
        assert_eq!(bank.phase_coefficients(1), &[2, 4]);
    }

    #[test]
    fn execute_uses_normal_convolution_order() {
        let bank = PolyphaseFilterBankArray::<i32, 6>::from_parts(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });

        assert_eq!(bank.execute(0, [10, 40].iter()), 80);
        assert_eq!(bank.execute(1, [20, 50].iter()), 200);
        assert_eq!(bank.execute(2, [30, 60].iter()), 360);
    }

    /// Bundled coefficients can produce an output type different from the input sample type.
    ///
    /// This keeps ordinary scalar FIRs unchanged while allowing callers such as ML-TED timing
    /// recovery to evaluate interpolation and derivative taps over the same sample history.
    #[test]
    fn execute_can_return_a_different_output_type() {
        let bank = PolyphaseFilterBankArray::<Pair, 2>::from_parts(Config {
            num_phases: 1,
            taps_per_phase: 2,
            coefficients: [
                Pair {
                    first: 1,
                    second: 2,
                },
                Pair {
                    first: 3,
                    second: 5,
                },
            ],
        });

        let out: Pair = bank.execute(0, [7, 11].iter());
        assert_eq!(
            out,
            Pair {
                first: 32,
                second: 57
            }
        );
    }

    #[test]
    #[should_panic(expected = "phase index out of range")]
    fn execute_phase_out_of_range_panics() {
        let bank = PolyphaseFilterBankArray::<i32, 4>::from_parts(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });

        let _ = bank.execute(2, [10, 20].iter());
    }

    #[test]
    #[should_panic(expected = "taps count must equal taps_per_phase")]
    fn execute_taps_count_mismatch_panics() {
        let bank = PolyphaseFilterBankArray::<i32, 4>::from_parts(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });

        let _ = bank.execute(0, [10].iter());
    }

    #[test]
    fn ref_mut_uses_caller_owned_phase_major_coefficients() {
        let mut coefficients = [1, 3, 2, 4];
        let bank: PolyphaseFilterBankRefMut<'_, i32> = PolyphaseFilterBank::from_parts(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: &mut coefficients[..],
        });

        assert_eq!(bank.phase_coefficients(0), &[1, 3]);
        assert_eq!(bank.phase_coefficients(1), &[2, 4]);
    }
}
