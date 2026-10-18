// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A sine oscillator and a square oscillator mixed, envelope-tracked, then peak-held.
//!
//! Run with `cargo run --example audio_synth --features alloc`.

use signalo::filters::iir::envelope::{self, Envelope};
use signalo::filters::ops::add::Add;
use signalo::sinks::peak_hold::PeakHold;
use signalo::sources::oscillator::sine::{self, SineOscillator};
use signalo::sources::oscillator::square::{self, SquareOscillator};
use signalo::traits::{Filter, Finalize, Sink, Source, WithConfig};

fn main() {
    let sample_rate = 48_000.0;
    let samples = 4_800;

    let mut sine = SineOscillator::<f32>::with_config(sine::Config::new(220.0, sample_rate));
    let mut square =
        SquareOscillator::<f32>::with_config(square::Config::new(110.0, sample_rate, 0.3));
    let mut add = Add;
    let mut envelope = Envelope::with_config(envelope::Config {
        attack: 0.05,
        release: 0.005,
    });
    let mut peak_hold = PeakHold::<f32>::default();

    for _ in 0..samples {
        let sine_sample = sine.source().expect("sine oscillator is infinite");
        let square_sample = square.source().expect("square oscillator is infinite");
        let mixed = add.filter((sine_sample, square_sample));
        let tracked = envelope.filter(mixed);
        peak_hold.sink(tracked);
    }

    // The mix peaks at roughly 1.3 (unit sine plus 0.3 square); the envelope follower
    // tracks its magnitude and the peak hold reports the largest tracked value.
    let peak = peak_hold.finalize();
    println!("samples: {samples}, peak: {:?}", peak);
    assert!(matches!(peak, Some(value) if 0.5 < value && value < 1.5));
}
