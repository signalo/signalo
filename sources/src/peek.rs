// // This Source Code Form is subject to the terms of the Mozilla Public
// // License, v. 2.0. If a copy of the MPL was not distributed with this
// // file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Peakable wrapper sinks.

use signalo_traits::Source;

/// The peek source's state.
#[derive(Clone, Debug)]
pub struct State<T, U> {
    /// Inner source.
    pub inner: T,
    /// Peekable value.
    pub peeked: Option<Option<U>>,
}

/// A source wrapper that caches the wrapped inner source.
#[derive(Clone, Debug)]
pub struct Peek<T, U> {
    state: State<T, U>,
}

impl<T, U> Peek<T, U>
where
    T: Source<Output = U>,
{
    /// Returns the most recent value returned from `self.source()`, otherwise `None`.
    pub fn peek(&mut self) -> Option<&U> {
        if self.state.peeked.is_none() {
            self.state.peeked = Some(self.state.inner.source());
        }
        match self.state.peeked {
            Some(Some(ref value)) => Some(value),
            Some(None) => None,
            _ => unreachable!(),
        }
    }
}

impl<T, U> From<T> for Peek<T, U> {
    fn from(inner: T) -> Self {
        let peeked = None;
        let state = State { inner, peeked };
        Self { state }
    }
}

impl<T, U> Default for Peek<T, U>
where
    T: Default,
{
    fn default() -> Self {
        Self::from(T::default())
    }
}

impl<T, U> Source for Peek<T, U>
where
    T: Source<Output = U>,
    U: Clone,
{
    type Output = U;

    fn source(&mut self) -> Option<Self::Output> {
        match self.state.peeked.take() {
            Some(output) => output,
            None => self.state.inner.source(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Dummy {
        value: f32,
    }

    impl Source for Dummy {
        type Output = f32;

        fn source(&mut self) -> Option<Self::Output> {
            let value = self.value;
            self.value += 1.0;
            Some(value)
        }
    }

    #[test]
    fn test() {
        let mut peek = Peek::from(Dummy::default());

        assert_nearly_eq!(peek.peek(), Some(0.0));
        assert_nearly_eq!(peek.peek(), Some(0.0));

        assert_nearly_eq!(peek.source(), Some(0.0));
        assert_nearly_eq!(peek.source(), Some(1.0));
        assert_nearly_eq!(peek.source(), Some(2.0));
    }
}
