// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Lift a time-agnostic filter over time-tagged samples.
//!
//! [`Lift`] adapts an existing [`Filter`]`<T>` so it can be dropped into a
//! continuous-time pipeline that carries [`Timed`] samples. The inner filter
//! only ever sees the signal value; the time delta `dt` rides through untouched.
//! Prefer `Lift` over reimplementing a filter for `Timed` whenever the
//! transformation is independent of the spacing between samples.
//!
//! # Examples
//!
//! ```
//! use signalo::filters::ops::square::Square;
//! use signalo::filters::util::lift::Lift;
//! use signalo::traits::Filter;
//! use signalo::Timed;
//!
//! let mut filter = Lift(Square::default());
//! let out = filter.filter(Timed::new(3.0_f32, 0.25_f32));
//!
//! assert_eq!(out.value, 9.0);
//! assert_eq!(out.dt, 0.25);
//! ```

use crate::time::Timed;
use crate::traits::Filter;

/// Lifts a time-agnostic [`Filter<T>`] into a [`Filter<Timed<T, Tm>>`], applying the inner
/// filter to the value and passing `dt` through unchanged.
#[derive(Clone, Debug)]
pub struct Lift<F>(pub F);

impl<F, T, Tm> Filter<Timed<T, Tm>> for Lift<F>
where
    F: Filter<T>,
{
    type Output = Timed<F::Output, Tm>;

    fn filter(&mut self, input: Timed<T, Tm>) -> Self::Output {
        Timed {
            value: self.0.filter(input.value),
            dt: input.dt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::ops::square::Square; // pointwise x -> x*x, Filter<T, Output = T>

    #[test]
    fn lift_transforms_value_preserves_dt() {
        let mut f = Lift(Square::default());
        let out = f.filter(Timed::new(3.0_f32, 0.25_f32));
        assert_eq!(out.value, 9.0);
        assert_eq!(out.dt, 0.25);
    }
}
