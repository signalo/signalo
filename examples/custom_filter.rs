// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A user-defined one-pole [`Filter`] driven by a [`Step`] source.
//!
//! The filter implements the exponential smoothing recurrence
//! `state += alpha * (input - state)`.
//!
//! Run with `cargo run --example custom_filter`.
//!
//! [`Filter`]: signalo::traits::Filter
//! [`Step`]: signalo::sources::step::Step

use signalo::sources::step::Step;
use signalo::traits::{Filter, Source};

/// A first-order low-pass filter smoothing its input towards the current state.
struct OnePole {
    /// Smoothing factor in `0.0..=1.0`; larger values track the input more closely.
    alpha: f32,
    /// The most recently emitted output.
    state: f32,
}

impl Filter<f32> for OnePole {
    type Output = f32;

    fn filter(&mut self, input: f32) -> f32 {
        self.state += self.alpha * (input - self.state);

        self.state
    }
}

fn main() {
    let mut source = Step::new(1.0_f32);
    let mut one_pole = OnePole {
        alpha: 0.1,
        state: 0.0,
    };

    let mut outputs = Vec::with_capacity(64);

    for _ in 0..64 {
        let sample = source.source().expect("step source is infinite");
        outputs.push(one_pole.filter(sample));
    }

    let last = outputs[63];

    println!("final: {last:.6}, outputs[..8]: {:?}", &outputs[..8]);

    assert_eq!(outputs.len(), 64);
    assert!(outputs[63] > 0.99, "final output {last} did not converge");
    for (index, pair) in outputs.windows(2).enumerate() {
        assert!(
            pair[1] >= pair[0] - 1.0e-6,
            "outputs[{index}] and outputs[{next}] are not monotonic",
            next = index + 1
        );
    }
}
