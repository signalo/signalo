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
}

impl_pipe!(Mean<T>);

impl<T> Filter<T> for Mean<T>
where
    T: Clone + Integer + Add<T, Output=T> + Sub<T, Output=T> + Shl<usize, Output=T> + Shr<usize, Output=T>
{
    type Output = T;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        let state = match self.state.clone() {
            None => {
                input << self.shift
            },
            Some(mut state) => {
                state = (state.clone() << self.beta) - state;
                state = state + (input << self.shift);
                state = state >> self.beta;
                state
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
        let filter = Mean::new(1, 0);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_eq!(output, vec![0, 0, 3, 2, 3, 5, 10, 6, 12, 9, 11, 10, 9, 13, 15, 9, 10, 15, 17, 12]);
    }
}
