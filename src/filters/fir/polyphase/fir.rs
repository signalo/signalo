// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Stateful polyphase FIR phase executor.

use core::marker::PhantomData;
use core::ops::{Add, Mul};

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::{Num, Zero};

use crate::storage::{zero_filled_fixed_ring, AsSlice, RingBuffer};
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Reset, State as StateTrait, StateMut,
    WithConfig,
};

use super::filter_bank::{Config, PolyphaseFilterBank};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "alloc")]
use super::filter_bank::PolyphaseFilterBankVec;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The polyphase FIR executor's state.
///
/// Holds the delay line used by all phase branches.
#[derive(Clone, Debug)]
pub struct State<R> {
    /// Buffered input samples used as the delay line.
    pub taps: R,
}

/// A stateful polyphase FIR executor.
///
/// This type combines a [`PolyphaseFilterBank`] with one delay line. Call
/// [`push`](Self::push) to append a new input sample, then
/// [`execute`](Self::execute) to evaluate any phase against the current delay
/// state.
///
/// # Coefficient ordering
///
/// Coefficients use the same phase-major ordering and execution semantics as
/// [`PolyphaseFilterBank`]. This type keeps sample history and supplies it to
/// the bank when [`Self::execute`] is called.
///
/// # Type aliases
///
/// Prefer the concrete aliases for common use:
/// - [`PolyphaseFirArray<T, N, H, K>`] for stack-allocated coefficients and
///   delay-line storage.
#[cfg_attr(
    feature = "alloc",
    doc = "- [`PolyphaseFirVec<T, K>`] for heap-allocated coefficients and delay-line storage."
)]
#[cfg_attr(
    not(feature = "alloc"),
    doc = "- `PolyphaseFirVec<T, K>` for heap-allocated coefficients and delay-line storage."
)]
/// - [`PolyphaseFirRefMut<'_, T, C, K>`] for caller-owned delay-line storage.
///
/// # Complexity
///
/// - **Time per sample:** [`push`](Self::push) is O(1); [`execute`](Self::execute) is O(H/P),
///   where H is the total tap count and P the number of phases.
/// - **Space:** O(H) for coefficients plus O(H/P) for the delay line.
#[derive(Clone, Debug)]
pub struct PolyphaseFir<T, C, R, K = T> {
    bank: PolyphaseFilterBank<C>,
    state: State<R>,
    _pd: PhantomData<(T, K)>,
}

/// A polyphase FIR executor backed by fixed coefficient and delay-line storage.
///
/// `N` is the total coefficient count. `H` is the number of coefficients in
/// each phase branch and must match the configuration's `taps_per_phase`.
pub type PolyphaseFirArray<T, const N: usize, const H: usize, K = T> =
    PolyphaseFir<T, [K; N], FixedCircularBuffer<T, H>, K>;

/// A polyphase FIR executor backed by heap-allocated storage.
///
/// Requires the `alloc` feature.
#[cfg(feature = "alloc")]
pub type PolyphaseFirVec<T, K = T> = PolyphaseFir<T, alloc::vec::Vec<K>, HeapCircularBuffer<T>, K>;

/// A polyphase FIR executor that borrows caller-owned delay-line storage.
pub type PolyphaseFirRefMut<'a, T, C, K = T> = PolyphaseFir<T, C, &'a mut CircularBuffer<T>, K>;

impl<T, C, R, K> PolyphaseFir<T, C, R, K>
where
    C: AsSlice<K>,
    R: RingBuffer<T>,
{
    /// Creates a [`PolyphaseFir`] from an already-constructed `config` and
    /// delay-line buffer.
    ///
    /// Use this constructor when the delay-line storage is caller-owned or must
    /// be constructed with a runtime capacity.
    ///
    /// The delay-line buffer is taken as-is with its current contents. It must
    /// contain `taps_per_phase` samples before [`Self::execute`] is called.
    ///
    /// # Expected storage state
    ///
    /// For zero-padded cold-start behavior, prefill the buffer with
    /// `taps_per_phase` zeros before passing it here.
    ///
    /// # Panics
    ///
    /// Panics if the delay-line capacity does not equal the config's
    /// `taps_per_phase`.
    pub fn from_parts(config: Config<C>, taps: R) -> Self {
        let bank = PolyphaseFilterBank::from_parts(config);
        assert_eq!(
            taps.capacity(),
            bank.taps_per_phase(),
            "PolyphaseFir: taps capacity must equal taps_per_phase"
        );

        Self {
            bank,
            state: State { taps },
            _pd: PhantomData,
        }
    }
}

impl<T, C, R, K> PolyphaseFir<T, C, R, K>
where
    R: RingBuffer<T>,
{
    /// Appends a new input sample to the delay line.
    pub fn push(&mut self, input: T) {
        let _ = self.state.taps.push_back(input);
    }
}

impl<T, C, R, K> PolyphaseFir<T, C, R, K>
where
    T: Clone + Zero + Add<Output = T> + Mul<K, Output = T>,
    K: Clone,
    C: AsSlice<K>,
    R: RingBuffer<T>,
{
    /// Evaluates `phase` against the current delay line.
    ///
    /// # Panics
    ///
    /// Panics if `phase >= self.num_phases()`, or if the current delay-line
    /// length does not equal `self.taps_per_phase()`.
    #[must_use]
    pub fn execute(&self, phase: usize) -> T {
        self.bank.execute(phase, self.state.taps.iter())
    }
}

impl<T, C, R, K> PolyphaseFir<T, C, R, K> {
    /// Returns the number of polyphase branches.
    #[must_use]
    pub fn num_phases(&self) -> usize {
        self.bank.num_phases()
    }

    /// Returns the number of coefficients in each phase branch.
    #[must_use]
    pub fn taps_per_phase(&self) -> usize {
        self.bank.taps_per_phase()
    }

    /// Returns the total number of coefficients.
    #[must_use]
    pub fn total_taps(&self) -> usize {
        self.bank.total_taps()
    }
}

impl<T, C, R, K> ConfigTrait for PolyphaseFir<T, C, R, K> {
    type Config = Config<C>;
}

impl<T, const N: usize, const H: usize, K> WithConfig for PolyphaseFirArray<T, N, H, K>
where
    T: Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let taps = zero_filled_fixed_ring::<T, H>();
        Self::from_parts(config, taps)
    }
}

#[cfg(feature = "alloc")]
impl<T, K> PolyphaseFirVec<T, K>
where
    T: Zero,
    K: Clone + Zero,
{
    /// Creates a heap-backed polyphase FIR executor from dense prototype
    /// coefficients.
    ///
    /// This convenience constructor allocates coefficient and delay-line storage.
    /// `prototype` is passed to
    /// [`PolyphaseFilterBankVec::from_prototype_taps`] for ordering and padding
    /// behavior. The delay line is zero-filled using the resulting
    /// `taps_per_phase`.
    ///
    /// # Panics
    ///
    /// Panics if `num_phases` is zero, `prototype` is empty, or the padded
    /// coefficient count overflows `usize`.
    #[must_use]
    pub fn from_prototype_taps(num_phases: usize, prototype: &[K]) -> Self {
        let bank = PolyphaseFilterBankVec::from_prototype_taps(num_phases, prototype);
        let mut taps = HeapCircularBuffer::with_capacity(bank.taps_per_phase());
        for _ in 0..bank.taps_per_phase() {
            let _ = taps.push_back(T::zero());
        }
        Self::from_parts(bank.into_guts(), taps)
    }
}

impl<T, C, R, K> ConfigRef for PolyphaseFir<T, C, R, K> {
    fn config_ref(&self) -> &Self::Config {
        self.bank.config_ref()
    }
}

impl<T, C, R, K> ConfigClone for PolyphaseFir<T, C, R, K>
where
    Config<C>: Clone,
{
    fn config(&self) -> Self::Config {
        self.bank.config()
    }
}

impl<T, C, R, K> StateTrait for PolyphaseFir<T, C, R, K> {
    type State = State<R>;
}

impl<T, C, R, K> StateMut for PolyphaseFir<T, C, R, K> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C, R, K> HasGuts for PolyphaseFir<T, C, R, K> {
    type Guts = (Config<C>, State<R>);
}

impl<T, C, R, K> FromGuts for PolyphaseFir<T, C, R, K> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self {
            bank: PolyphaseFilterBank::from_guts(config),
            state,
            _pd: PhantomData,
        }
    }
}

impl<T, C, R, K> IntoGuts for PolyphaseFir<T, C, R, K> {
    fn into_guts(self) -> Self::Guts {
        (self.bank.into_guts(), self.state)
    }
}

impl<T, const N: usize, const H: usize, K> Reset for PolyphaseFirArray<T, N, H, K>
where
    T: Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.bank.into_guts())
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize, const H: usize, K> ResetMut for PolyphaseFirArray<T, N, H, K> where
    Self: Reset
{
}

#[cfg(test)]
mod tests {
    use super::{PolyphaseFir, PolyphaseFirArray, PolyphaseFirRefMut, State};
    use crate::filters::fir::convolve::{Config as ConvolveConfig, ConvolveArray};
    use crate::filters::fir::polyphase::filter_bank::Config;
    use crate::traits::{
        guts::{FromGuts, IntoGuts},
        Filter, Reset, WithConfig,
    };

    #[test]
    fn execute_uses_normal_convolution_order() {
        let mut fir = PolyphaseFirArray::<i32, 6, 2>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });

        assert_eq!(fir.num_phases(), 3);
        assert_eq!(fir.taps_per_phase(), 2);
        assert_eq!(fir.total_taps(), 6);

        fir.push(10);
        assert_eq!(fir.execute(0), 10);
        assert_eq!(fir.execute(1), 20);
        assert_eq!(fir.execute(2), 30);

        fir.push(20);
        assert_eq!(fir.execute(0), 60);
        assert_eq!(fir.execute(1), 90);
        assert_eq!(fir.execute(2), 120);
    }

    #[test]
    fn each_phase_matches_convolve_with_same_phase_coefficients() {
        let mut fir = PolyphaseFirArray::<i32, 6, 2>::with_config(Config {
            num_phases: 3,
            taps_per_phase: 2,
            coefficients: [1, 4, 2, 5, 3, 6],
        });
        let mut phase0 = ConvolveArray::<i32, 2>::with_config(ConvolveConfig {
            coefficients: [1, 4],
        });
        let mut phase1 = ConvolveArray::<i32, 2>::with_config(ConvolveConfig {
            coefficients: [2, 5],
        });
        let mut phase2 = ConvolveArray::<i32, 2>::with_config(ConvolveConfig {
            coefficients: [3, 6],
        });

        for input in [10, 20, 30, 40] {
            fir.push(input);
            assert_eq!(fir.execute(0), phase0.filter(input));
            assert_eq!(fir.execute(1), phase1.filter(input));
            assert_eq!(fir.execute(2), phase2.filter(input));
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn real_taps_complex_samples_match_independent_real_filters() {
        use approx::assert_abs_diff_eq;

        use crate::complex::Complex32;

        let coefficients = [0.5_f32, 0.125, -0.25, 0.0625];
        let real_input = [1.0_f32, -2.0, 3.0, 5.0, -8.0];
        let imag_input = [13.0_f32, -21.0, 34.0, -55.0, 89.0];

        let mut real_filter = PolyphaseFirArray::<f32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients,
        });
        let mut imag_filter = PolyphaseFirArray::<f32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients,
        });
        let mut complex_filter = PolyphaseFirArray::<Complex32, 4, 2, f32>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients,
        });

        for (&real, &imag) in real_input.iter().zip(&imag_input) {
            real_filter.push(real);
            imag_filter.push(imag);
            complex_filter.push(Complex32::new(real, imag));

            for phase in 0..2 {
                let real_output = real_filter.execute(phase);
                let imag_output = imag_filter.execute(phase);
                let complex_output = complex_filter.execute(phase);

                assert_abs_diff_eq!(complex_output.re, real_output, epsilon = 1e-6);
                assert_abs_diff_eq!(complex_output.im, imag_output, epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "taps capacity must equal taps_per_phase")]
    fn from_parts_capacity_mismatch_panics() {
        let config = Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        };
        let taps = circular_buffer::FixedCircularBuffer::<i32, 3>::new();

        let _ = PolyphaseFir::from_parts(config, taps);
    }

    #[test]
    #[should_panic(expected = "phase index out of range")]
    fn execute_phase_out_of_range_panics() {
        let fir = PolyphaseFirArray::<i32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });

        let _ = fir.execute(2);
    }

    #[test]
    fn reset_clears_delay_line() {
        let mut fir = PolyphaseFirArray::<i32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });

        fir.push(10);
        fir.push(20);
        assert_eq!(fir.execute(0), 50);

        let fir = fir.reset();
        assert_eq!(fir.execute(0), 0);
        assert_eq!(fir.execute(1), 0);
    }

    #[test]
    fn ref_mut_uses_caller_owned_delay_line() {
        let config = Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        };
        let mut taps = circular_buffer::FixedCircularBuffer::<i32, 2>::new();
        let _ = taps.push_back(0);
        let _ = taps.push_back(0);
        let mut fir: PolyphaseFirRefMut<'_, i32, [i32; 4]> =
            PolyphaseFir::from_parts(config, &mut taps);

        fir.push(10);

        assert_eq!(fir.execute(0), 10);
        assert_eq!(fir.execute(1), 20);
    }

    #[test]
    fn guts_round_trip_preserves_state() {
        let mut fir = PolyphaseFirArray::<i32, 4, 2>::with_config(Config {
            num_phases: 2,
            taps_per_phase: 2,
            coefficients: [1, 3, 2, 4],
        });
        fir.push(10);
        fir.push(20);

        let guts = fir.into_guts();
        let fir = PolyphaseFirArray::<i32, 4, 2>::from_guts(guts);

        assert_eq!(fir.execute(0), 50);
        assert_eq!(fir.execute(1), 80);
    }

    #[test]
    fn from_guts_accepts_existing_state() {
        let mut taps = circular_buffer::FixedCircularBuffer::<i32, 2>::new();
        let _ = taps.push_back(10);
        let _ = taps.push_back(20);

        let fir = PolyphaseFirArray::<i32, 4, 2>::from_guts((
            Config {
                num_phases: 2,
                taps_per_phase: 2,
                coefficients: [1, 3, 2, 4],
            },
            State { taps },
        ));

        assert_eq!(fir.execute(0), 50);
        assert_eq!(fir.execute(1), 80);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn vec_from_prototype_taps_reorders_and_zero_fills_delay_line() {
        let mut fir = super::PolyphaseFirVec::<i32>::from_prototype_taps(3, &[1, 2, 3, 4, 5]);

        assert_eq!(fir.num_phases(), 3);
        assert_eq!(fir.taps_per_phase(), 2);

        fir.push(10);
        assert_eq!(fir.execute(0), 10);
        assert_eq!(fir.execute(1), 20);
        assert_eq!(fir.execute(2), 30);

        fir.push(20);
        assert_eq!(fir.execute(0), 60);
        assert_eq!(fir.execute(1), 90);
        assert_eq!(fir.execute(2), 60);
    }
}
