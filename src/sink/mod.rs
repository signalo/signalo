mod sum;
mod last;

pub use self::sum::Sum;
pub use self::last::Last;

pub trait Sink<T>: Sized {
    type Output;

    fn sink(&mut self, input: T);
    fn finalize(self) -> Self::Output;
}

impl<F, T> Sink<T> for F
where
    F: FnMut(T) -> (),
{
    type Output = ();

    fn sink(&mut self, input: T) {
        self(input)
    }

    fn finalize(self) -> Self::Output {
        ()
    }
}
