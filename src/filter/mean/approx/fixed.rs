use std::ops::BitOr;
use std::ops::{Sub, Add, Shl, Shr};

use num_integer::Integer;

use filter::pipe::Pipe;
use filter::Filter;

#[derive(Clone, Debug)]
pub struct Mean<T> {
    beta: usize,
    shift: usize,
    state: Option<T>,
}

impl<T> Mean<T> {
    #[inline]
    pub fn new(beta: usize, shift: usize) -> Self {
        Mean { beta, shift, state: None }
    }

    pub fn beta(&self) -> usize {
        self.beta
    }

    pub fn shift(&self) -> usize {
        self.shift
    }
}

impl_pipe!(Mean<T>);

impl<T> Filter<T> for Mean<T>
where
    T: Clone + Integer + Add<T, Output=T> + Sub<T, Output=T> + Shl<usize, Output=T> + Shr<usize, Output=T>
{
    type Output = T;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        let input = input << self.shift;
        let state = match self.state.clone() {
            None => {
                input
            },
            Some(mut state) => {
                state.clone() + ((input - state) >> self.beta)
            },
        };
        self.state = Some(state.clone());
        state >> self.shift
    }

    fn reset(&mut self) {
        self.state = None;
    }

    fn phase_shift(&self) -> isize {
        0 // FIXME!!!
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed() {
        let filter = Mean::new(2, 0);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7, 7, 15, 15, 10, 23, 10, 111, 18, 18, 18, 106, 5, 26, 13, 13, 21, 21, 21, 34, 8, 109, 8, 29, 16, 16, 16, 104, 11, 24, 24];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_eq!(output, vec![0, 0, 1, 1, 2, 3, 6, 5, 8, 7, 8, 8, 8, 10, 11, 9, 9, 11, 13, 11, 10, 11, 12, 11, 14, 13, 37, 32, 28, 25, 45, 35, 32, 27, 23, 22, 21, 21, 24, 20, 42, 33, 32, 28, 25, 22, 42, 34, 31, 29]);
    }
}
