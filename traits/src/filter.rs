// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub trait Filter<Input>: Sized {
    type Output;

    fn filter(&mut self, input: Input) -> Self::Output;
}

impl<F, T, U> Filter<T> for F
where
    F: FnMut(T) -> U,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        self(input)
    }
}
