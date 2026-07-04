// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! FIR window functions for spectral analysis and filter design.
//!
//! # Important: streaming vs. frame mode
//!
//! Each window is a stateful streaming filter that multiplies input samples by
//! a window coefficient `w[k mod N]`. This is a **periodic mask**: the same
//! sequence `w[0], w[1], …, w[N−1]` repeats every N calls, indefinitely.
//!
//! This differs from the typical use of window functions for frame-based
//! FFT pre-processing or FIR filter design:
//!
//! - Windows that are zero at both endpoints (Hann, Blackman, Blackman-Harris,
//!   Flat-top) will **zero the output** every N samples — the output train has
//!   hard zeros at periodic intervals.
//! - The result acts as an AM-modulated comb, not a conventional apodisation
//!   window.
//!
//! For frame-based windowing, construct the weights via `Config::new()` on
//! each window type and apply them manually to your frame, or use the
//! [`windowed_sinc`](super::convolve::windowed_sinc) module for window-based
//! FIR filter design.
//!

pub mod blackman;
pub mod blackman_harris;
pub mod flat_top;
pub mod hamming;
pub mod hann;
pub mod kaiser;
pub mod rectangular;
pub mod triangular;

/// Generate shared behavior tests for window types.
///
/// `$with_config` — expression creating the window for `N=8` (used in
/// `periodicity` and `reset_restarts_counter`).
/// `$zero_window_expr` — expression creating the window for `N=0`
/// (used in `zero_window_panics`).
/// `$panic_prefix` — substring matched in the panic message.
///
/// The `periodicity` and `reset_restarts_counter` tests call `Config::new()`
/// (which requires `Float`) and collect into `Vec`; they are therefore gated
/// on `any(feature = "libm", feature = "std")`.  The `zero_window_panics`
/// test only exercises the `with_config` constructor and is always compiled.
#[cfg(test)]
#[macro_export]
macro_rules! window_behavior_tests {
    ($with_config:expr, $zero_window_expr:expr, $panic_prefix:literal) => {
        #[cfg(any(feature = "libm", feature = "std"))]
        #[test]
        fn periodicity() {
            const N: usize = 8;
            let mut window = $with_config;
            let input: alloc::vec::Vec<f32> = core::iter::repeat(1.0_f32).take(3 * N).collect();
            let output: alloc::vec::Vec<_> = input.iter().map(|&x| window.filter(x)).collect();
            for block in 0..3 {
                let start = block * N;
                let end = start + N;
                assert_eq!(&output[start..end], &output[0..N]);
            }
        }

        #[cfg(any(feature = "libm", feature = "std"))]
        #[test]
        fn reset_restarts_counter() {
            const N: usize = 8;
            let mut window = $with_config;
            let first_half: alloc::vec::Vec<f32> = (0..N as u32).map(|i| i as f32).collect();
            let output_a: alloc::vec::Vec<_> =
                first_half.iter().map(|&x| window.filter(x)).collect();
            window = window.reset();
            let output_b: alloc::vec::Vec<_> =
                first_half.iter().map(|&x| window.filter(x)).collect();
            assert_eq!(output_a, output_b);
        }

        #[test]
        #[should_panic(expected = $panic_prefix)]
        fn zero_window_panics() {
            let _ = $zero_window_expr;
        }
    };
}
