use source::Source;

pub trait Sink<T>: Sized {
    type Output;

    fn sink(&mut self, input: T);
    fn finalize(self) -> Self::Output;

    fn drain<S>(mut self, mut source: S) -> Self::Output
    where
        S: Source<Output=T>,
    {
        while let Some(input) = source.next() {
            self.consume(input);
        }
        self.finalize()
    }
}
