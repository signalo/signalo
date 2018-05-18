// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::{
    PartialOrd,
    Ordering,
};

use signalo_traits::filter::Filter;

use filter::classify::{
    Slope,
    Classification,
};

/// A slope's kind.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Peak {
    /// A local maximum.
    Max,
    /// A local constant.
    None,
    /// A local minimum.
    Min
}

impl Default for Peak {
    fn default() -> Self {
        Peak::None
    }
}

/// A trait describing a classification value.
impl Classification<[Peak; 3]> for Peak {
    fn classes() -> [Peak; 3] {
        [Peak::Max, Peak::None, Peak::Min]
    }
}

/// A peak detection filter.
#[derive(Clone, Debug)]
pub struct Peaks<T, U> {
    state: (Option<T>, Slope),
    /// [rising, flat, falling] outputs.
    outputs: [U; 3],
}

impl<T, U> Peaks<T, U>
where
    U: Clone,
{
    /// Creates a new `Peaks` filter with given `threshold` and `outputs` (`[max, none, min]`).
    #[inline]
    pub fn new(outputs: [U; 3]) -> Self {
        Peaks {
            state: (None, Slope::None),
            outputs,
        }
    }
}

impl<T, U> Filter<T> for Peaks<T, U>
where
    T: Clone + PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let (slope, index) = match self.state {
            (None, _) => {
                (Slope::None, 1) // None
            },
            (Some(ref state), Slope::Rising) => {
                match state.partial_cmp(&input).unwrap() {
                    Ordering::Less => (Slope::Rising, 1), // None
                    Ordering::Equal => (Slope::None, 1), // None
                    Ordering::Greater => (Slope::Falling, 0), // Max
                }
            },
            (Some(ref state), Slope::None) => {
                match state.partial_cmp(&input).unwrap() {
                    Ordering::Less => (Slope::Rising, 1), // None
                    Ordering::Equal => (Slope::None, 1), // None
                    Ordering::Greater => (Slope::Falling, 1), // None
                }
            },
            (Some(ref state), Slope::Falling) => {
                match state.partial_cmp(&input).unwrap() {
                    Ordering::Less => (Slope::Rising, 2), // Min
                    Ordering::Equal => (Slope::None, 1), // None
                    Ordering::Greater => (Slope::Falling, 1), // None
                }
            }
        };
        self.state = (Some(input), slope);
        self.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn peak() {
        use self::Peak::*;

        let filter = Peaks::new(Peak::classes());
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_eq!(output, vec![
            None, None, None, Max, Min, None, None, Max, Min, Max,
            Min, Max, None, None, None, None, Min, None, None, None,
        ]);
    }
}
