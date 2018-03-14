mod pipe;
mod source;
mod identity;
mod differentiate;

pub mod ops;
pub mod dichotomy;

pub use self::pipe::Pipe;
pub use self::source::Source;
pub use self::identity::Identity;
pub use self::differentiate::Differentiate;

pub trait LinearPhase {
    fn phase_shift() -> isize {
        0 // specialize for linearly phase-shifting filter types
    }
}

pub trait Filter<Input>: Sized {
    type Output;

    fn apply(&mut self, input: Input) -> Self::Output;

    fn reset(&mut self) {
        // specialize for stateful filter types
    }

    fn phase_shift(&self) -> isize {
        0 // specialize for phase-shifting filter types
    }
}

impl<F, T, U> Filter<T> for F
where
    F: FnMut(T) -> U,
{
    type Output = U;

    fn apply(&mut self, input: T) -> Self::Output {
        self(input)
    }
}
