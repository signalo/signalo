use std::ops::BitOr;
use std::ops::Sub as StdSub;
use std::mem;

use num_traits::Zero;

use filter::pipe::Pipe;
use filter::Filter;

#[derive(Default, Clone)]
pub struct Differentiate<T> {
    prev: Option<T>,
}

impl<T, Rhs> BitOr<Rhs> for Differentiate<T> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, filter: Rhs) -> Self::Output {
        Pipe::new(self, filter)
    }
}

impl<T> Filter<T> for Differentiate<T>
where
    T: Copy + StdSub<T>,
    <T as StdSub<T>>::Output: Zero
{
    type Output = <T as StdSub<T>>::Output;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        let mut prev = Some(input);
        mem::swap(&mut self.prev, &mut prev);
        if let Some(prev) = prev {
            input - prev
        } else {
            <T as StdSub<T>>::Output::zero()
        }
    }

    fn reset(&mut self) {
        self.prev = None;
    }

    fn phase_shift(&self) -> isize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Differentiate::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_eq!(output, vec![0, 1, 6, -5, 3, 3, 8, -13, 16, -13, 8, -5, 0, 8, 0, -13, 8, 8, 0, -13]);
    }

    #[test]
    fn floating_point() {
        let filter = Differentiate::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -13.0, 16.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0, -13.0, 8.0, 8.0, 0.0, -13.0]);
    }
}
