// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Dimensional analysis wrapper sinks.

#![cfg(feature = "dimensioned")]

use std::marker::PhantomData;

use dimensioned::{
    traits::Dimensioned,
    unit_systems::{cgs::CGS, fps::FPS, mks::MKS, si::SI, ucum::UCUM},
};

use signalo_traits::{Finalize, Sink};

/// A sink wrapper that preserves the signal's dimensional unit.
#[derive(Clone, Debug)]
pub struct UnitSystem<S, T> {
    /// Inner sink.
    pub inner: S,
    _phantom: PhantomData<T>,
}

impl<S, T> From<S> for UnitSystem<S, T> {
    fn from(inner: S) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
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
        impl<S, U, V> Sink<$t<V, U>> for UnitSystem<S, $t<V, U>>
        where
            S: Sink<V>,
            V: Clone,
            $t<V, U>: Dimensioned<Value = V, Units = U>,
        {
            #[inline]
            fn sink(&mut self, input: $t<V, U>) {
                self.inner.sink(input.value_unsafe().clone())
            }
        }

        impl<S, U, V> Finalize for UnitSystem<S, $t<V, U>>
        where
            S: Finalize<Output = V>,
            $t<V, U>: Dimensioned<Value = V, Units = U>,
        {
            type Output = $t<V, U>;

            #[inline]
            fn finalize(self) -> Self::Output {
                $t::new(self.inner.finalize())
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

    struct Sum {
        sum: f32,
    }

    impl Sink<f32> for Sum {
        fn sink(&mut self, input: f32) {
            self.sum += input;
        }
    }

    impl Finalize for Sum {
        type Output = f32;

        fn finalize(self) -> Self::Output {
            self.sum
        }
    }

    #[test]
    fn test() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input: Vec<_> = vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ].into_iter()
        .map(|unitless| Meter::new(unitless))
        .collect();
        let mut sink = UnitSystem::from(Sum { sum: 0.0 });
        for input in input {
            sink.sink(input);
        }
        let subject = sink.finalize();
        assert_eq!(subject, Meter::new(196.0));
    }
}
