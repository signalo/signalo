// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::Div as StdDiv;

use signalo_traits::filter::Filter;

#[derive(Clone, Debug)]
pub struct Div<T> {
    value: T
}

impl<T> Div<T> {
    #[inline]
    pub fn new(value: T) -> Self {
        Div { value }
    }

    #[inline]
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T, U> Filter<U> for Div<T>
where
    T: Copy,
    U: StdDiv<T>,
{
    type Output = <U as StdDiv<T>>::Output;

    #[inline]
    fn filter(&mut self, input: U) -> Self::Output {
        input / self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Div::new(42);
        let input = vec![0, 42, 294, 84, 210, 336, 672, 126, 798, 252, 588, 378, 378, 714, 714, 168, 504, 840, 840, 294];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7]);
    }

    #[test]
    fn floating_point() {
        let filter = Div::new(4.2);
        let input = vec![0.0, 4.2, 29.4, 8.4, 21.0, 33.6, 67.2, 12.6, 79.8, 25.2, 58.8, 37.8, 37.8, 71.4, 71.4, 16.8, 50.4, 84.0, 84.0, 29.4];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0]);
    }
}
