// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Unit impulse signal source.
//!
//! Generates a single-sample pulse at time zero with zero amplitude thereafter, representing
//! the discrete-time unit impulse (Dirac delta function). Fundamental for impulse response
//! measurement and filter characterization.
//! Generates a single-sample pulse at time zero, with zero amplitude thereafter.
//! This is the fundamental DSP primitive for impulse response measurement and filter testing.

use crate::traits::{Reset, Source};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// A source that generates a unit impulse signal.
///
/// Returns the specified amplitude on the first call, then zero on all subsequent calls.
/// Mathematically: `δ[n] = amplitude` for n=0, `δ[n] = 0` for n>0.
///
/// ### Example:
///
/// ```
/// # fn main() {
/// use signalo::sources::impulse::Impulse;
/// let impulse = Impulse::new(1.0f32);
/// // ╭─────╮  ╭─────╮  ╭─────╮  ╭─────╮  ╭─────╮
/// // │ 1.0 │─▶│ 0.0 │─▶│ 0.0 │─▶│ 0.0 │─▶│ 0.0 │─▶ ...
/// // ╰─────╯  ╰─────╯  ╰─────╯  ╰─────╯  ╰─────╯
/// # }
///```
#[derive(Clone, Debug)]
pub struct Impulse<T> {
    amplitude: T,
    fired: bool,
}

impl<T> Impulse<T> {
    /// Creates a new `Impulse` source with the given amplitude.
    #[inline]
    #[must_use]
    pub fn new(amplitude: T) -> Self {
        Self {
            amplitude,
            fired: false,
        }
    }
}

impl<T> Reset for Impulse<T> {
    fn reset(self) -> Self {
        Self {
            amplitude: self.amplitude,
            fired: false,
        }
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for Impulse<T> where Self: Reset {}

impl<T> Source for Impulse<T>
where
    T: Clone + Default,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        if self.fired {
            Some(T::default())
        } else {
            self.fired = true;
            Some(self.amplitude.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_impulse_fires_once_then_zeros() {
        const AMPLITUDE: f32 = 1.0;
        const COUNT: usize = 5;
        let source = Impulse::new(AMPLITUDE);
        let subject: Vec<_> = (0..COUNT)
            .scan(source, |source, _| source.source())
            .collect();
        let expected = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(subject, expected);
    }

    #[test]
    fn test_impulse_different_amplitudes() {
        let test_cases = vec![
            (2.5f64, vec![2.5, 0.0, 0.0]),
            (-1.5f64, vec![-1.5, 0.0, 0.0]),
            (0.0f64, vec![0.0, 0.0, 0.0]),
        ];

        for (amplitude, expected) in test_cases {
            let source = Impulse::new(amplitude);
            let subject: Vec<_> = (0..expected.len())
                .scan(source, |source, _| source.source())
                .collect();
            assert_eq!(subject, expected, "Failed for amplitude={}", amplitude);
        }
    }

    #[test]
    fn test_impulse_integer_type() {
        const AMPLITUDE: i32 = 42;
        const COUNT: usize = 4;
        let source = Impulse::new(AMPLITUDE);
        let subject: Vec<_> = (0..COUNT)
            .scan(source, |source, _| source.source())
            .collect();
        let expected = vec![42, 0, 0, 0];
        assert_eq!(subject, expected);
    }
}
