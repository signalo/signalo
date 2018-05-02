// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub trait Sink<T>: Sized {
    type Output;

    fn sink(&mut self, input: T);
    fn finalize(self) -> Self::Output;
}

impl<F, T> Sink<T> for F
where
    F: FnMut(T) -> (),
{
    type Output = ();

    fn sink(&mut self, input: T) {
        self(input)
    }

    fn finalize(self) -> Self::Output {
        ()
    }
}
