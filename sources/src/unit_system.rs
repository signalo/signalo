// // This Source Code Form is subject to the terms of the Mozilla Public
// // License, v. 2.0. If a copy of the MPL was not distributed with this
// // file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Dimensional analysis wrapper sinks.

#![cfg(feature = "dimensioned")]

use std::marker::PhantomData;

use dimensioned::{
    traits::Dimensioned,
    unit_systems::{cgs::CGS, fps::FPS, mks::MKS, si::SI, ucum::UCUM},
};

use signalo_traits::Source;

/// A sink that computes the integrate of all received values of a signal.
#[derive(Clone, Debug)]
pub struct UnitSystem<S, T> {
    /// Inner sink.
    pub inner: S,
    _phantom: PhantomData<T>,
}

impl<S, T> From<S> for UnitSystem<S, T> {
    fn from(inner: S) -> Self {
        let _phantom = PhantomData;
        Self { inner, _phantom }
    }
}

impl<S, T> Default for UnitSystem<S, T>
where
    S: Default,
{
    fn default() -> Self {
        Self::from(S::default())
    }
}

macro_rules! impl_dimensioned {
    ($t:ident) => {
        impl<S, U, V> Source for UnitSystem<S, $t<V, U>>
        where
            S: Source<Output = V>,
            $t<V, U>: Dimensioned<Value = V, Units = U>,
        {
            type Output = $t<V, U>;

            fn source(&mut self) -> Option<Self::Output> {
                self.inner.source().map(|value| $t::new(value))
            }
        }
    };
}

impl_dimensioned!(CGS);
impl_dimensioned!(FPS);
impl_dimensioned!(MKS);
impl_dimensioned!(SI);
impl_dimensioned!(UCUM);

#[cfg(test)]
mod tests {
    use super::*;

    use dimensioned::unit_systems::si::Meter;

    struct Values {
        values: Vec<f32>,
    }

    impl Values {
        fn new(mut values: Vec<f32>) -> Self {
            values.reverse();
            Self { values }
        }
    }

    impl Source for Values {
        type Output = f32;

        fn source(&mut self) -> Option<Self::Output> {
            self.values.pop()
        }
    }

    #[test]
    fn test() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input: Vec<_> = vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        let expected: Vec<_> = input
            .iter()
            .map(|unitless| Meter::new(unitless.clone()))
            .collect();
        let mut source: UnitSystem<_, Meter<f32>> = UnitSystem::from(Values::new(input));
        let mut subject = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        assert_eq!(subject.len(), expected.len());
        for (subject, expected) in subject.into_iter().zip(expected) {
            assert_eq!(subject, expected);
        }
    }
}
