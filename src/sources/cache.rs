// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Cache source that stores and repeats the most recent generated value.
//!
//! Wraps another source and caches its last output, allowing repeated access without
//! requesting new values from the underlying source.

use crate::traits::Source;

/// The cache source's state.
#[derive(Clone, Debug)]
pub struct State<T, U> {
    /// Inner source.
    pub inner: T,
    /// Cached value.
    pub cached: Option<U>,
}

/// A source wrapper that caches the wrapped inner source.
///
/// # Complexity
///
/// - **Time per sample:** same as the wrapped source `T` on the first call; O(1) on subsequent
///   calls when a cached value is already present.
/// - **Space:** same as `T` plus O(1) for the cached value.
#[derive(Clone, Debug)]
pub struct Cache<T, U> {
    state: State<T, U>,
}

impl<T, U> Cache<T, U> {
    /// Returns the most recent value returned from `self.source()`, otherwise `None`.
    pub fn cached(&self) -> Option<&U> {
        self.state.cached.as_ref()
    }
}

impl<T, U> From<T> for Cache<T, U> {
    fn from(inner: T) -> Self {
        let cached = None;
        let state = State { inner, cached };
        Self { state }
    }
}

impl<T, U> Default for Cache<T, U>
where
    T: Default,
{
    fn default() -> Self {
        Self::from(T::default())
    }
}

impl<T, U> Source for Cache<T, U>
where
    T: Source<Output = U>,
    U: Clone,
{
    type Output = U;

    fn source(&mut self) -> Option<Self::Output> {
        let cached = self.state.inner.source();
        self.state.cached.clone_from(&cached);
        cached
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
        let mut cache = Cache::from(Dummy::default());
        assert_eq!(cache.cached(), None);

        let expected = cache.source();
        assert_eq!(cache.cached(), expected.as_ref());

        let expected = cache.source();
        assert_eq!(cache.cached(), expected.as_ref());
    }
}
