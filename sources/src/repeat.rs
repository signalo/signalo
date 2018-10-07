// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Value repeating sources.

use signalo_traits::Source;

use constant::Constant;
use take::Take;

/// A source that returns a specified number of constant values.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_sources;
/// #
/// # fn main() {
/// use signalo_sources::repeat::Repeat;
/// let repeat = Repeat::new(42, 3);
/// // ╭────╮  ╭────╮  ╭────╮
/// // │ 42 │─▶│ 42 │─▶│ 42 │
/// // ╰────╯  ╰────╯  ╰────╯
/// # }
///```
#[derive(Clone, Debug)]
pub struct Repeat<T> {
    inner: Take<Constant<T>>,
}

impl<T> Repeat<T> {
    /// Creates a new `Repeat` source for a given `initial` value and an `interval`.
    #[inline]
    pub fn new(value: T, count: usize) -> Self {
        let constant = Constant::new(value);
        let inner = Take::new(constant, count);
        Self { inner }
    }
}

impl<T> Source for Repeat<T>
where
    T: Clone,
    Constant<T>: Source<Output = T>,
    Take<Constant<T>>: Source<Output = T>,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        self.inner.source()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut source = Repeat::new(42, 5);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        // let subject: Vec<_> = (0..EXCESS_COUNT)
        //     .scan(source, |source, _| source.source())
        //     .collect();
        let expected = vec![42; 5];
        assert_eq!(subject, expected);
    }
}
