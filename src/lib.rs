extern crate num_traits;
extern crate num_integer;

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
