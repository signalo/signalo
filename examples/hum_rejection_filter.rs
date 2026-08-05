// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo::filters::iir::analog::{butterworth_lowpass, lowpass_biquad_configs};
use signalo::filters::iir::biquad::cascade::{self, BiquadCascadeArray};
use signalo::sources::noise::Noise;
use signalo::traits::{ConfigRef, Filter, Source, StateMut, WithConfig};

fn baseline(t: f64) -> f64 {
    50.0 + 10.0 * (2.0 * core::f64::consts::PI * 0.5 * t).sin()
}

fn hum(t: f64) -> f64 {
    3.0 * (2.0 * core::f64::consts::PI * 50.0 * t).sin()
}

/// Scales a `Noise` source's raw `u32` output to `[-half_width, +half_width]`.
fn noise_to_range(raw: u32, half_width: f64) -> f64 {
    (f64::from(raw) / f64::from(u32::MAX) - 0.5) * 2.0 * half_width
}

#[allow(clippy::needless_range_loop)]
fn main() {
    let fs = 1000.0_f64;
    let duration_secs = 2.0_f64;
    let n_samples = (fs * duration_secs) as usize;
    let dt = 1.0 / fs;

    let mut noise = Noise::new(99);
    let mut raw = [0.0_f64; 2000];
    let mut filtered_arr = [0.0_f64; 2000];

    let cfgs = lowpass_biquad_configs::<f64, 2>(butterworth_lowpass::<f64>(4), fs, 5.0);
    let cascade_config = cascade::Config::new(cfgs);
    let mut filter: BiquadCascadeArray<f64, 2> = BiquadCascadeArray::with_config(cascade_config);

    // Initialize state to avoid startup transient
    let initial_val = 50.0;
    for i in 0..2 {
        let (b0, b2, a2) = {
            let cfg = &filter.config_ref().sections[i];
            (cfg.b0, cfg.b2, cfg.a2)
        };
        let st = &mut filter.state_mut().sections[i];
        st.s1 = (1.0 - b0) * initial_val;
        st.s2 = (b2 - a2) * initial_val;
    }

    for i in 0..n_samples {
        let t = i as f64 * dt;
        let raw_noise = noise.source().expect("failed to generate noise");

        raw[i] = baseline(t) + hum(t) + noise_to_range(raw_noise, 0.5);
        filtered_arr[i] = filter.filter(raw[i]);
    }

    let warm_up = 200usize;
    let delay = 83usize;

    let mut sum_sq_before = 0.0_f64;
    let mut sum_sq_after = 0.0_f64;

    for i in warm_up..n_samples {
        if i >= delay {
            let j = i - delay;
            let t = j as f64 * dt;
            sum_sq_before += (raw[j] - baseline(t)).powi(2);
            sum_sq_after += (filtered_arr[i] - baseline(t)).powi(2);
        }
    }

    for i in 0..n_samples {
        if i % 200 == 0 {
            println!(
                "Sample {:4}: raw = {:8.4}, filtered = {:8.4}",
                i, raw[i], filtered_arr[i]
            );
        }
    }

    let count = n_samples - warm_up;
    let rms_before = (sum_sq_before / count as f64).sqrt();
    let rms_after = (sum_sq_after / count as f64).sqrt();
    let attenuation_db = 20.0 * (rms_before / rms_after).log10();

    println!("Hum+noise attenuation: {} dB", attenuation_db);

    assert!(
        attenuation_db > 20.0,
        "filter must attenuate hum+noise by more than 20 dB"
    );
}
