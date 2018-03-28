use std::ops::BitOr;
use std::ops::{Sub, Add, Mul, Div};

use num_traits::{Zero, One};

use filter::pipe::Pipe;
use filter::Filter;

#[derive(Clone, Debug)]
pub struct Kalman<T> {
    r: T, // Process noise
    q: T, // Measurement noise
    a: T, // State
    c: T, // Measurement
    cov: T, // Uncertainty
    state: Option<T>,
}

impl<T> Kalman<T>
where
    T: Zero + One
{
    #[inline]
    pub fn new(r: T, q: T, a: T, c: T) -> Self {
        Kalman {
            r: r,
            q: q,
            a: a,
            c: c,
            cov: Zero::zero(),
            state: None,
        }
    }

    pub fn r(&self) -> &T {
        &self.r
    }

    pub fn q(&self) -> &T {
        &self.q
    }

    pub fn a(&self) -> &T {
        &self.a
    }

    pub fn c(&self) -> &T {
        &self.c
    }
}

impl<T> Default for Kalman<T>
where
    T: Zero + One
{
    fn default() -> Self {
        Kalman::new(One::one(), One::one(), One::one(), One::one())
    }
}

impl_pipe!(Kalman<T>);

impl<T> Filter<T> for Kalman<T>
where
    T: Copy + Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T>
{
    type Output = T;

    #[inline]
    fn apply(&mut self, input: T) -> Self::Output {
        let (state, cov) = match self.state {
            None => {
                let state = input / self.c;
                let cov = self.q / (self.c * self.c);
                (state, cov)
            },
            Some(mut state) => {
                // Compute prediction
                let pred_state = self.a * state;
                let pred_cov = (self.a * self.cov * self.a) + self.r;

                // Kalman gain
                let k = pred_cov * self.c / ((self.c * pred_cov * self.c) + self.q);

                // Correction
                let state = pred_state + k * (input - (self.c * pred_state));
                let cov = pred_cov - (k * self.c * pred_cov);
                (state, cov)
            },
        };
        self.state = Some(state);
        self.cov = cov;
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

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.000, 0.502, 2.704, 2.522, 3.047, 3.946, 5.883, 5.463, 7.288, 7.125, 7.951, 8.071,
            8.175, 9.129, 9.960, 9.342, 9.614, 10.660, 11.591, 11.138, 10.731, 11.148, 11.522,
            11.375, 12.497, 12.256, 21.739, 21.381, 21.057, 20.765, 28.908, 26.625, 26.566,
            25.272, 24.103, 23.807, 23.540, 23.298, 24.317, 22.763, 30.971, 28.785, 28.806,
            27.587, 26.485, 25.487, 32.957, 30.868, 30.215, 29.623
        ]
    }

    #[test]
    fn floating_point() {
        let r = 0.01; // Process noise
        let q = 1.0; // Measurement noise
        let a = 1.0; // State
        let c = 1.0; // Measurement
        let filter = Kalman::new(r, q, a, c);

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();

        assert_nearly_eq!(output, get_output(), 0.001);
    }

    #[test]
    fn fixed_point() {
        use std::convert::TryFrom;
        use fpa::I16F16;
        type Fixed = I16F16;

        let r = Fixed::try_from(0.01).unwrap(); // Process noise
        let q = Fixed::try_from(1.00).unwrap(); // Measurement noise
        let a = Fixed::try_from(1.00).unwrap(); // State
        let c = Fixed::try_from(1.00).unwrap(); // Measurement
        let filter = Kalman::new(r, q, a, c);

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let float_input = get_input();
        println!("input: {:?}", float_input);

        let input: Vec<_> = float_input.into_iter().map(|float| {
            Fixed::try_from(float).unwrap()
        }).collect();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.apply(input))
        }).collect();

        let float_output: Vec<_> = output.into_iter().map(|fixed| {
            f32::try_from(fixed).unwrap()
        }).collect();
        println!("output: {:?}", float_output);

        assert_nearly_eq!(float_output, get_output(), 0.01);
    }
}
