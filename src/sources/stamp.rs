// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Constant time-delta tagging of a discrete source.

use crate::time::Timed;
use crate::traits::Source;

/// Wraps a [`Source`] so each produced value carries a fixed `dt`, producing
/// [`Timed`] values.
#[derive(Clone, Debug)]
pub struct Stamp<S, Tm = f32> {
    source: S,
    dt: Tm,
}

impl<S, Tm> Stamp<S, Tm> {
    /// Creates a stamper that attaches `dt` to every value from `source`.
    pub const fn new(source: S, dt: Tm) -> Self {
        Self { source, dt }
    }
}

impl<S, Tm> Source for Stamp<S, Tm>
where
    S: Source,
    Tm: Copy,
{
    type Output = Timed<S::Output, Tm>;

    fn source(&mut self) -> Option<Self::Output> {
        let dt = self.dt;

        self.source.source().map(|value| Timed { value, dt })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Once(Option<f32>);

    impl Source for Once {
        type Output = f32;

        fn source(&mut self) -> Option<f32> {
            self.0.take()
        }
    }

    #[test]
    fn stamp_attaches_constant_dt() {
        let mut s = Stamp::new(Once(Some(7.0_f32)), 0.1_f32);
        assert_eq!(
            s.source(),
            Some(Timed {
                value: 7.0,
                dt: 0.1
            })
        );
        assert_eq!(s.source(), None);
    }
}
