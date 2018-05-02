// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub trait Source: Sized {
    type Output;

    fn source(&mut self) -> Option<Self::Output>;
}

impl<F, T> Source for F
where
    F: FnMut() -> Option<T>,
{
    type Output = T;

    fn source(&mut self) -> Option<Self::Output> {
        self()
    }
}
