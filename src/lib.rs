#![feature(try_from)]

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
