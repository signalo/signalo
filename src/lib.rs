#![cfg_attr(feature = "nightly", feature(try_from))]

#![cfg_attr(feature = "missing_mpl", feature(plugin))]
#![cfg_attr(feature = "missing_mpl", plugin(missing_mpl))]
#![cfg_attr(feature = "missing_mpl", deny(missing_mpl))]

extern crate num_traits;
extern crate num_integer;

#[cfg(feature = "fpa")]
extern crate fpa;

#[cfg(feature = "fpa")]
extern crate typenum;

#[cfg(test)]
#[macro_use]
extern crate nearly_eq;

pub mod filter;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
