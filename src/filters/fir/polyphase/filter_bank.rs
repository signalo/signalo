// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Polyphase FIR filter bank.

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
/// Coefficients are stored phase-major in normal FIR order within each phase:
///
/// ```text
/// phase p: h[p], h[p + P], h[p + 2P], ...
/// ```
///
/// where `P = num_phases`. [`Self::execute`] accepts sample history in
/// oldest-to-newest order. For the selected phase, the first phase coefficient
/// is applied to the newest sample, matching normal FIR convolution coefficient
/// semantics.
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
/// - **Time per sample:** O(H/P) per [`Self::execute`] call, where H is the total tap count and
///   P the number of phases; each call convolves one phase branch of H/P coefficients.
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

        let Some(expected_taps) = config.num_phases.checked_mul(config.taps_per_phase) else {
            panic!("PolyphaseFilterBank: num_phases * taps_per_phase overflowed");
        };

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
    /// is provided in normal tap order, not phase-major order. It is zero-padded
    /// at the tail as needed, then packed using the phase-major coefficient
    /// ordering described on [`PolyphaseFilterBank`].
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

        let taps_per_phase = prototype.len().div_ceil(num_phases);
        let Some(total_taps) = num_phases.checked_mul(taps_per_phase) else {
            panic!("PolyphaseFilterBank: num_phases * taps_per_phase overflowed");
        };

        let mut coefficients = alloc::vec![K::zero(); total_taps];
        for phase in 0..num_phases {
            for tap in 0..taps_per_phase {
                let src = tap * num_phases + phase;
                if src < prototype.len() {
                    let dst = phase * taps_per_phase + tap;
                    coefficients[dst] = prototype[src].clone();
                }
            }
        }

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
    use super::{Config, PolyphaseFilterBank, PolyphaseFilterBankArray, PolyphaseFilterBankRefMut};
    use crate::filters::fir::polyphase::test_support::Pair;
    use crate::traits::WithConfig;

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
