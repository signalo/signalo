// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![cfg_attr(feature = "missing_mpl", feature(plugin))]
#![cfg_attr(feature = "missing_mpl", plugin(missing_mpl))]
#![cfg_attr(feature = "missing_mpl", deny(missing_mpl))]

extern crate signalo_traits;
extern crate signalo_filters;
extern crate signalo_pipes;

pub mod prelude {
    pub use signalo_traits::prelude as traits;
    pub use signalo_filters::prelude as filters;
    pub use signalo_pipes::prelude as pipes;
}
