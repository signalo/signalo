// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Monotonically incrementing sources.

use core::ops::AddAssign;

use signalo_traits::Source;

/// A source that returns an auto-incremented value on each call.
///
/// ### Example:
///
/// ```
/// # fn main() {
/// use signalo_sources::increment::Increment;
/// let increment = Increment::new(0, 2);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 2 │─▶│ 4 │─▶│ 6 │─▶│ 8 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
/// # }
///```
#[derive(Default, Clone, Debug)]
pub struct Increment<T> {
    state: T,
    interval: T,
}

impl<T> Increment<T> {
    /// Creates a new `Increment` source for a given `initial` value and an `interval`.
    #[inline]
    pub fn new(initial: T, interval: T) -> Self {
        Self {
            state: initial,
            interval,
        }
    }
}

impl<T> Source for Increment<T>
where
    T: Clone + AddAssign<T>,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        let output = self.state.clone();
        self.state += self.interval.clone();
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let source = Increment::new(42, 2);
        let subject: Vec<_> = (0..5).scan(source, |source, _| source.source()).collect();
        let expected = vec![42, 44, 46, 48, 50];
        assert_eq!(subject, expected);
    }
}
