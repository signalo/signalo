// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Unit step signal source.
//!
//! Generates an infinite stream of a constant value, representing the discrete-time
//! unit step function. This is the DSP primitive for step responses and system testing.

use crate::traits::Source;

/// A source that generates a unit step signal.
///
/// Returns a constant amplitude value on each call, implementing the discrete unit step function.
/// Mathematically: `u[n] = amplitude` for all n >= 0.
///
/// ### Example:
///
/// ```
/// # fn main() {
/// use signalo::sources::step::Step;
/// let step = Step::new(1.0f32);
/// // ╭─────╮  ╭─────╮  ╭─────╮  ╭─────╮  ╭─────╮
/// // │ 1.0 │─▶│ 1.0 │─▶│ 1.0 │─▶│ 1.0 │─▶│ 1.0 │─▶ ...
/// // ╰─────╯  ╰─────╯  ╰─────╯  ╰─────╯  ╰─────╯
/// # }
///```
#[derive(Default, Clone, Debug)]
pub struct Step<T> {
    amplitude: T,
}

impl<T> Step<T> {
    /// Creates a new `Step` source with the given amplitude.
    #[inline]
    #[must_use]
    pub fn new(amplitude: T) -> Self {
        Self { amplitude }
    }
}

impl<T> From<T> for Step<T> {
    #[inline]
    fn from(amplitude: T) -> Self {
        Self { amplitude }
    }
}

impl<T> Source for Step<T>
where
    T: Clone,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        Some(self.amplitude.clone())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_step_constant_output() {
        const AMPLITUDE: f32 = 1.0;
        const COUNT: usize = 5;
        let source = Step::new(AMPLITUDE);
        let subject: Vec<_> = (0..COUNT)
            .scan(source, |source, _| source.source())
            .collect();
        let expected = vec![AMPLITUDE; COUNT];
        assert_eq!(subject, expected);
    }

    #[test]
    fn test_step_different_amplitudes() {
        let test_cases = vec![(0.0f64, 3), (1.5f64, 4), (-2.5f64, 3)];

        for (amplitude, count) in test_cases {
            let source = Step::new(amplitude);
            let subject: Vec<_> = (0..count)
                .scan(source, |source, _| source.source())
                .collect();
            let expected = vec![amplitude; count];
            assert_eq!(subject, expected, "Failed for amplitude={}", amplitude);
        }
    }

    #[test]
    fn test_step_from_value() {
        const VALUE: i32 = 42;
        let mut step: Step<i32> = Step::from(VALUE);
        let output = step.source();
        assert_eq!(output, Some(VALUE));
    }
}
