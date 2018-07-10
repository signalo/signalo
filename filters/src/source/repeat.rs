// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo_traits::source::Source;

use source::take::Take;
use source::constant::Constant;

/// A source that returns a specified number of constant values.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_filters;
/// #
/// # fn main() {
/// use signalo_filters::source::Repeat;
/// let repeat = Repeat::new(42, 3);
/// // ╭────╮  ╭────╮  ╭────╮
/// // │ 42 │─▶│ 42 │─▶│ 42 │
/// // ╰────╯  ╰────╯  ╰────╯
/// # }
///```
#[derive(Clone, Debug)]
pub struct Repeat<T> {
    inner: Take<Constant<T>>
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
        const VALUE: f32 = 4.2;
        const COUNT: usize = 3;

        const EXCESS_COUNT: usize = COUNT + 10;

        let constant = Repeat::new(VALUE, COUNT);
        let source = Take::new(constant, COUNT);
        let subject: Vec<_> = (0..EXCESS_COUNT).scan(source, |source, _| {
            source.source()
        }).collect();
        let expected = vec![VALUE; COUNT];
        assert_eq!(subject, expected);
    }
}
