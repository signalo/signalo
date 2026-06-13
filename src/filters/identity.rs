// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Identify function filters.

use crate::traits::Filter;

/// A filter that simply returns the values it receives.
#[derive(Default, Clone, Debug)]
pub struct Identity;

impl<T> Filter<T> for Identity {
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        input
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test() {
        let filter = Identity;
        let input = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_abs_diff_eq!(
            output.as_slice(),
            [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0].as_slice(),
            epsilon = 1e-6
        );
    }
}
