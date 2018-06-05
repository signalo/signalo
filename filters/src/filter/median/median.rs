// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving median filters.

use median::stack::Filter1 as InnerMedian1;
use median::stack::Filter2 as InnerMedian2;
use median::stack::Filter3 as InnerMedian3;
use median::stack::Filter4 as InnerMedian4;
use median::stack::Filter5 as InnerMedian5;
use median::stack::Filter6 as InnerMedian6;
use median::stack::Filter7 as InnerMedian7;
use median::stack::Filter8 as InnerMedian8;
use median::stack::Filter9 as InnerMedian9;
use median::stack::Filter10 as InnerMedian10;
use median::stack::Filter11 as InnerMedian11;
use median::stack::Filter12 as InnerMedian12;
use median::stack::Filter13 as InnerMedian13;
use median::stack::Filter14 as InnerMedian14;
use median::stack::Filter15 as InnerMedian15;
use median::stack::Filter16 as InnerMedian16;
use median::stack::Filter17 as InnerMedian17;
use median::stack::Filter18 as InnerMedian18;
use median::stack::Filter19 as InnerMedian19;
use median::stack::Filter20 as InnerMedian20;
use median::stack::Filter21 as InnerMedian21;
use median::stack::Filter22 as InnerMedian22;
use median::stack::Filter23 as InnerMedian23;
use median::stack::Filter24 as InnerMedian24;
use median::stack::Filter25 as InnerMedian25;
use median::stack::Filter26 as InnerMedian26;
use median::stack::Filter27 as InnerMedian27;
use median::stack::Filter28 as InnerMedian28;
use median::stack::Filter29 as InnerMedian29;
use median::stack::Filter30 as InnerMedian30;
use median::stack::Filter31 as InnerMedian31;
use median::stack::Filter32 as InnerMedian32;

use signalo_traits::filter::Filter;

macro_rules! median_def {
    ($n:ident<$t:ident>: $i:ident<$u:ident>) => {
        /// A filter producing the approximated moving median over a given signal.
        #[derive(Clone, Debug)]
        pub struct $n<$t> {
            inner: $i<$u>,
        }

        impl<T> $n<$t>
        where
            T: Clone + PartialOrd
        {
            /// Creates a new `Median` filter.
            #[inline]
            pub fn new() -> Self {
                Self { inner: $i::new() }
            }
        }

        impl<T> Filter<T> for $n<$t>
        where
            T: Copy + PartialOrd
        {
            type Output = T;

            fn filter(&mut self, input: T) -> Self::Output {
                self.inner.consume(input)
            }
        }
    }
}

median_def!(Median1<T>: InnerMedian1<T>);
median_def!(Median2<T>: InnerMedian2<T>);
median_def!(Median3<T>: InnerMedian3<T>);
median_def!(Median4<T>: InnerMedian4<T>);
median_def!(Median5<T>: InnerMedian5<T>);
median_def!(Median6<T>: InnerMedian6<T>);
median_def!(Median7<T>: InnerMedian7<T>);
median_def!(Median8<T>: InnerMedian8<T>);
median_def!(Median9<T>: InnerMedian9<T>);
median_def!(Median10<T>: InnerMedian10<T>);
median_def!(Median11<T>: InnerMedian11<T>);
median_def!(Median12<T>: InnerMedian12<T>);
median_def!(Median13<T>: InnerMedian13<T>);
median_def!(Median14<T>: InnerMedian14<T>);
median_def!(Median15<T>: InnerMedian15<T>);
median_def!(Median16<T>: InnerMedian16<T>);
median_def!(Median17<T>: InnerMedian17<T>);
median_def!(Median18<T>: InnerMedian18<T>);
median_def!(Median19<T>: InnerMedian19<T>);
median_def!(Median20<T>: InnerMedian20<T>);
median_def!(Median21<T>: InnerMedian21<T>);
median_def!(Median22<T>: InnerMedian22<T>);
median_def!(Median23<T>: InnerMedian23<T>);
median_def!(Median24<T>: InnerMedian24<T>);
median_def!(Median25<T>: InnerMedian25<T>);
median_def!(Median26<T>: InnerMedian26<T>);
median_def!(Median27<T>: InnerMedian27<T>);
median_def!(Median28<T>: InnerMedian28<T>);
median_def!(Median29<T>: InnerMedian29<T>);
median_def!(Median30<T>: InnerMedian30<T>);
median_def!(Median31<T>: InnerMedian31<T>);
median_def!(Median32<T>: InnerMedian32<T>);


#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 0.0, 1.0, 2.0, 5.0, 5.0, 8.0, 8.0, 16.0, 6.0, 14.0, 9.0, 9.0, 9.0, 17.0, 17.0,
            12.0, 12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 15.0, 10.0, 23.0, 18.0, 18.0, 18.0,
            18.0, 18.0, 26.0, 13.0, 13.0, 13.0, 21.0, 21.0, 21.0, 21.0, 34.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 16.0, 16.0, 24.0, 24.0
        ]
    }

    #[test]
    fn median() {
        let filter = Median3::new();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
