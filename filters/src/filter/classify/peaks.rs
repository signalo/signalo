// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialOrd;

use signalo_traits::filter::Filter;

use filter::classify::{
    slopes::{
        Slope,
        State as SlopesState,
        Slopes,
    },
    Classification,
};

use traits::{
    InitialState,
    Resettable,
    Stateful,
    StatefulUnsafe,
};

/// A slope's kind.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
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

/// A peak detection filter's internal state.
#[derive(Clone, Debug)]
pub struct State<T> {
    pub slopes: Slopes<T, Slope>,
    pub slope: Option<Slope>,
}

/// A peak detection filter.
#[derive(Clone, Debug)]
pub struct Peaks<T, U> {
    state: State<T>,
    /// rising, flat, falling outputs.
    outputs: [U; 3],
}

impl<T, U> Peaks<T, U>
where
    U: Clone,
{
    /// Creates a new `Peaks` filter with given `threshold` and `outputs` (`[max, none, min]`).
    #[inline]
    pub fn new(outputs: [U; 3]) -> Self {
        let state = Self::initial_state(());
        Peaks { state, outputs }
    }

    fn filter_internal(&mut self, slope: Slope) -> (Slope, usize) {
        let index = match self.state.slope {
            None => {
                1
            },
            Some(Slope::Rising) => {
                match &slope {
                    Slope::Rising => 1, // None
                    Slope::None => 1, // None
                    Slope::Falling => 0, // Max
                }
            },
            Some(Slope::None) => {
                match &slope {
                    Slope::Rising => 1, // None
                    Slope::None => 1, // None
                    Slope::Falling => 1, // None
                }
            },
            Some(Slope::Falling) => {
                match &slope {
                    Slope::Rising => 2, // Min
                    Slope::None => 1, // None
                    Slope::Falling => 1, // None
                }
            }
        };
        (slope, index)
    }
}

impl<T, U> Stateful for Peaks<T, U> {
    type State = State<T>;
}

unsafe impl<T, U> StatefulUnsafe for Peaks<T, U> {
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> InitialState<()> for Peaks<T, U> {
    fn initial_state(_: ()) -> Self::State {
        let slopes = Slopes::new(Slope::classes());
        let slope = None;
        State { slopes, slope }
    }
}

impl<T, U> Resettable for Peaks<T, U> {
    fn reset(&mut self) {
        self.state = Self::initial_state(());
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
        let slope = self.state.slopes.filter(input);
        let (state, index) = self.filter_internal(slope);
        self.state.slope = Some(state);
        self.outputs[index].clone()
    }
}

impl<U> Filter<Slope> for Peaks<Slope, U>
where
    U: Clone,
{
    type Output = U;

    #[inline]
    fn filter(&mut self, slope: Slope) -> Self::Output {
        let (state, index) = self.filter_internal(slope);
        unsafe {
            let inner_state = self.state.slopes.state_mut();
            *inner_state = SlopesState { input: Some(slope) };
        }
        self.state.slope = Some(state);
        self.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn values() {
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

    #[test]
    fn slopes() {
        use self::Peak::*;

        let filter = Peaks::new(Peak::classes());
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = {
            use self::Slope::*;
            vec![
                None, Rising, Rising, Falling, Rising, Rising, Rising, Falling, Rising, Falling,
                Rising, Falling, None, Rising, None, Falling, Rising, Rising, None, Falling,
            ]
        };
        let output: Vec<_> = input.iter().scan(filter, |filter, input| {
            Some(filter.filter(input.clone()))
        }).collect();
        assert_eq!(output, vec![
            None, None, None, Max, Min, None, None, Max, Min, Max,
            Min, Max, None, None, None, None, Min, None, None, None,
        ]);
    }
}
