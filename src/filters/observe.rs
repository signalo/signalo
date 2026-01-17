// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! [State observing](https://en.wikipedia.org/wiki/State_observer) filters.

/// Alpha-Beta filter for tracking signals with constant or linearly changing values.
///
/// A simplified observer providing first-order adaptive estimation with separate gains
/// for position and velocity tracking. Computationally lighter than Kalman filters.
pub mod alpha_beta;

/// Kalman filter for optimal recursive estimation of noisy signals.
///
/// Implements the discrete Kalman filter for state-space estimation, providing optimal
/// linear filtering in the presence of process and measurement noise.
pub mod kalman;
