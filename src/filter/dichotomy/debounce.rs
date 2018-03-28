use std::ops::BitOr;
use std::cmp::PartialEq;

use num_traits::Zero;

use filter::pipe::Pipe;
use filter::Filter;

// static itg_t m_itg = { .max = 20, .acc = 0, .output = 0 };

#[derive(Clone, Debug)]
pub struct Debounce<T, U> {
    /// Threshold of how long input must remain same to be accepted
    threshold: usize,
    /// [off, on] output
    output: [U; 2],
    /// Value to debounce
    predicate: T,
    /// Counter of how long input was the same
    counter: usize,
}

impl<T, U> Debounce<T, U>
where
    T: Copy + Zero
{
    #[inline]
    pub fn new(threshold: usize, predicate: T, output: [U; 2]) -> Self {
        Debounce { threshold, output, predicate, counter: 0 }
    }
}

impl_pipe!(Debounce<T, U>);

impl<T, U> Filter<T> for Debounce<T, U>
where
    T: Copy + PartialEq<T>,
    U: Copy,
{
    type Output = U;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        if input == self.predicate {
            self.counter = (self.counter + 1).min(self.threshold);
        } else {
            self.counter = 0;
        }
        self.output[(self.counter >= self.threshold) as usize]
    }

    fn reset(&mut self) {
        self.counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        let filter = Debounce::new(3, 1, [0, 1]);
        let input = vec![0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_eq!(output, vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
}
