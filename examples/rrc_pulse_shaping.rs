// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A real symbol stream through a root-raised-cosine polyphase interpolator.
//!
//! Run with `cargo run --example rrc_pulse_shaping`.

use signalo::filters::fir::design::{self, root_raised_cosine};
use signalo::filters::fir::polyphase::filter_bank;
use signalo::filters::fir::polyphase::interpolator::PolyphaseInterpolatorArray;
use signalo::traits::{MultirateFilter, WithConfig};

/// Bipolar symbol stream, one symbol per input sample of the interpolator.
const SYMBOLS: [f32; 16] = [
    1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
];

/// Interpolation factor, i.e. output samples per input symbol.
const SAMPLES_PER_SYMBOL: usize = 4;

/// Full pulse length in symbols, i.e. `SPAN / 2` symbols on either side of the
/// centre tap.
const SPAN: usize = 4;

/// RRC roll-off factor in `0.0..=1.0`.
const ROLLOFF: f32 = 0.35;

/// Number of prototype taps: `span * sps + 1`, so the pulse is centred on one
/// tap.
const PROTOTYPE_TAPS: usize = design::len_from_span(SPAN, SAMPLES_PER_SYMBOL);

/// Number of prototype taps per phase, rounded up so every prototype tap fits.
const TAPS_PER_PHASE: usize =
    filter_bank::taps_per_phase_for_prototype_len(SAMPLES_PER_SYMBOL, PROTOTYPE_TAPS);

/// Number of phase-major coefficients, with the shorter phases zero-padded.
const COEFFICIENTS: usize =
    filter_bank::packed_len_for_prototype_len(SAMPLES_PER_SYMBOL, PROTOTYPE_TAPS);

/// Number of zero symbols fed after the stream to flush the FIR tail.
const FLUSH_SYMBOLS: usize = TAPS_PER_PHASE - 1;

/// Number of output samples of the complete finite waveform.
const OUTPUT_SAMPLES: usize = (SYMBOLS.len() + FLUSH_SYMBOLS) * SAMPLES_PER_SYMBOL;

fn main() {
    let mut prototype = [0.0_f32; PROTOTYPE_TAPS];
    root_raised_cosine::taps(&mut prototype, SPAN, SAMPLES_PER_SYMBOL, ROLLOFF);

    // The polyphase bank stores its coefficients phase-major: phase `p` holds the
    // prototype taps whose index is congruent to `p` modulo the phase count, in
    // prototype order, with the slots past the prototype's end zero-padded.
    let mut coefficients = [0.0_f32; COEFFICIENTS];
    filter_bank::pack_prototype_taps(&mut coefficients, SAMPLES_PER_SYMBOL, &prototype);

    let mut interpolator =
        PolyphaseInterpolatorArray::<f32, COEFFICIENTS, TAPS_PER_PHASE>::with_config(
            filter_bank::Config {
                num_phases: SAMPLES_PER_SYMBOL,
                taps_per_phase: TAPS_PER_PHASE,
                coefficients,
            },
        );
    let mut shaped = Vec::with_capacity(OUTPUT_SAMPLES);

    for symbol in SYMBOLS
        .into_iter()
        .chain(core::iter::repeat_n(0.0, FLUSH_SYMBOLS))
    {
        let mut output = [0.0_f32; SAMPLES_PER_SYMBOL];
        let (consumed, produced) = interpolator.process(&[symbol], &mut output);

        assert!(consumed == 1, "each symbol is consumed exactly once");
        assert!(
            produced == SAMPLES_PER_SYMBOL,
            "each symbol yields exactly one sample per interpolation phase"
        );
        shaped.extend_from_slice(&output);
    }

    println!("samples: {}, first_16: {:?}", shaped.len(), &shaped[..16]);

    assert!(
        PROTOTYPE_TAPS == SPAN * SAMPLES_PER_SYMBOL + 1
            && interpolator.taps_per_phase() == TAPS_PER_PHASE,
        "the prototype spans the pulse symmetrically around its center tap"
    );
    assert!(
        shaped.len() == OUTPUT_SAMPLES,
        "the shaped stream is the flushed symbol stream interpolated by the interpolation factor"
    );
    assert!(
        shaped.iter().all(|value| value.is_finite()),
        "every shaped sample is finite"
    );
    assert!(
        shaped.iter().any(|value| value.abs() > 0.1),
        "the shaped stream carries non-zero energy"
    );
}
