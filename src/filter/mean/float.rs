use std::ops::BitOr;
use std::ops::{Sub, Add, Mul};

use num_traits::Float;

use filter::pipe::Pipe;
use filter::Filter;

#[derive(Clone)]
pub struct Mean<T> {
    beta: T,
    state: Option<T>,
}

impl<T> Mean<T> {
    #[inline]
    pub fn new(beta: T) -> Self {
        Mean { beta, state: None }
    }
}

impl_pipe!(Mean<T>);

impl<T> Filter<T> for Mean<T>
where
    T: Clone + Float + Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T>
{
    type Output = T;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        let state = match self.state.clone() {
            None => {
                input
            },
            Some(mut state) => {
                state.clone() + ((input - state) * self.beta)
            },
        };
        self.state = Some(state.clone());
        state
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
    fn floating_point() {
        let filter = Mean::new(0.25);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0, 20.0, 20.0, 7.0];
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();
        assert_nearly_eq!(output, vec![0.000, 0.250, 1.938, 1.953, 2.715, 4.036, 7.027, 6.020, 9.265, 8.449, 9.837, 9.628, 9.471, 11.353, 12.765, 10.574, 10.930, 13.198, 14.898, 12.924], 0.001);
    }
}
