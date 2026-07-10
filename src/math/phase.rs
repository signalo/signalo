// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Fixed-point phase trigonometry.
//!
//! A `u32` phase word represents one full turn:
//!
//! - `0x0000_0000`: 0 turns
//! - `0x4000_0000`: 1/4 turn
//! - `0x8000_0000`: 1/2 turn
//! - `0xC000_0000`: 3/4 turn
//! - `0xFFFF_FFFF`: one phase-word ULP below 1 turn
//!
//! The represented interval is `[0, 1)` turns. Wrapping arithmetic on the phase
//! word naturally wraps the oscillator phase.

const QUARTER_TURN: u32 = 0x4000_0000;
const QUARTER_BITS: u32 = 8;
const QUARTER_SIZE: usize = 1_usize << QUARTER_BITS;
const FRAC_BITS: u32 = 30 - QUARTER_BITS;
const FRAC_MASK: u32 = (1_u32 << FRAC_BITS) - 1;
#[allow(clippy::cast_precision_loss)]
const FRAC_SCALE_RECIP: f32 = 1.0 / ((1_u32 << FRAC_BITS) as f32);

/// Computes sine for a 32-bit wrapping phase word.
///
/// The full `u32` range maps to one turn, so `0x4000_0000` is a quarter turn
/// and `0xFFFF_FFFF` is one phase-word ULP below a full turn.
#[must_use]
#[inline]
pub fn sin(phase: u32) -> f32 {
    sin_from_quarter(phase)
}

/// Computes cosine for a 32-bit wrapping phase word.
///
/// This is equivalent to `sin(phase.wrapping_add(0x4000_0000))`.
#[must_use]
#[inline]
pub fn cos(phase: u32) -> f32 {
    sin_from_quarter(phase.wrapping_add(QUARTER_TURN))
}

/// Computes sine and cosine for a 32-bit wrapping phase word.
///
/// The return order matches Rust's `sin_cos` convention: `(sin, cos)`.
#[must_use]
#[inline]
pub fn sin_cos(phase: u32) -> (f32, f32) {
    (sin(phase), cos(phase))
}

/// Computes the complex phasor `cos(phase) + j sin(phase)`.
#[cfg(feature = "complex")]
#[must_use]
#[inline]
pub fn phasor(phase: u32) -> crate::complex::Complex32 {
    let (imag, real) = sin_cos(phase);
    crate::complex::Complex32::new(real, imag)
}

#[inline]
fn sin_from_quarter(phase: u32) -> f32 {
    let quadrant = phase >> 30;
    let offset = phase & (QUARTER_TURN - 1);
    let quarter_phase = if (quadrant & 1) == 0 {
        offset
    } else {
        QUARTER_TURN - offset
    };
    let value = sin_quarter(quarter_phase);

    if quadrant >= 2 {
        -value
    } else {
        value
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#[inline]
fn sin_quarter(quarter_phase: u32) -> f32 {
    debug_assert!(quarter_phase <= QUARTER_TURN);

    if quarter_phase == QUARTER_TURN {
        return SINE_QUARTER_TABLE[QUARTER_SIZE];
    }

    let index = (quarter_phase >> FRAC_BITS) as usize;
    let frac = (quarter_phase & FRAC_MASK) as f32 * FRAC_SCALE_RECIP;
    let y0 = SINE_QUARTER_TABLE[index];
    let y1 = SINE_QUARTER_TABLE[index + 1];

    y0 + (y1 - y0) * frac
}

#[allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::unreadable_literal
)]
#[rustfmt::skip]
const SINE_QUARTER_TABLE: [f32; QUARTER_SIZE + 1] = [
    0.0000000000e+00_f32, 6.1358846492e-03_f32, 1.2271538286e-02_f32, 1.8406729906e-02_f32,
    2.4541228523e-02_f32, 3.0674803177e-02_f32, 3.6807222941e-02_f32, 4.2938256935e-02_f32,
    4.9067674327e-02_f32, 5.5195244350e-02_f32, 6.1320736302e-02_f32, 6.7443919564e-02_f32,
    7.3564563600e-02_f32, 7.9682437971e-02_f32, 8.5797312344e-02_f32, 9.1908956497e-02_f32,
    9.8017140330e-02_f32, 1.0412163387e-01_f32, 1.1022220729e-01_f32, 1.1631863091e-01_f32,
    1.2241067520e-01_f32, 1.2849811079e-01_f32, 1.3458070851e-01_f32, 1.4065823933e-01_f32,
    1.4673047446e-01_f32, 1.5279718526e-01_f32, 1.5885814333e-01_f32, 1.6491312049e-01_f32,
    1.7096188876e-01_f32, 1.7700422041e-01_f32, 1.8303988796e-01_f32, 1.8906866415e-01_f32,
    1.9509032202e-01_f32, 2.0110463484e-01_f32, 2.0711137619e-01_f32, 2.1311031992e-01_f32,
    2.1910124016e-01_f32, 2.2508391136e-01_f32, 2.3105810828e-01_f32, 2.3702360608e-01_f32,
    2.4298017990e-01_f32, 2.4892760575e-01_f32, 2.5486565960e-01_f32, 2.6079411792e-01_f32,
    2.6671275747e-01_f32, 2.7262135545e-01_f32, 2.7851968939e-01_f32, 2.8440753721e-01_f32,
    2.9028467725e-01_f32, 2.9615088824e-01_f32, 3.0200594932e-01_f32, 3.0784964004e-01_f32,
    3.1368174040e-01_f32, 3.1950203082e-01_f32, 3.2531029216e-01_f32, 3.3110630576e-01_f32,
    3.3688985339e-01_f32, 3.4266071731e-01_f32, 3.4841868025e-01_f32, 3.5416352542e-01_f32,
    3.5989503653e-01_f32, 3.6561299780e-01_f32, 3.7131719395e-01_f32, 3.7700741022e-01_f32,
    3.8268343237e-01_f32, 3.8834504670e-01_f32, 3.9399204006e-01_f32, 3.9962419985e-01_f32,
    4.0524131400e-01_f32, 4.1084317106e-01_f32, 4.1642956010e-01_f32, 4.2200027080e-01_f32,
    4.2755509343e-01_f32, 4.3309381885e-01_f32, 4.3861623854e-01_f32, 4.4412214457e-01_f32,
    4.4961132965e-01_f32, 4.5508358713e-01_f32, 4.6053871096e-01_f32, 4.6597649577e-01_f32,
    4.7139673683e-01_f32, 4.7679923006e-01_f32, 4.8218377208e-01_f32, 4.8755016015e-01_f32,
    4.9289819223e-01_f32, 4.9822766697e-01_f32, 5.0353838373e-01_f32, 5.0883014254e-01_f32,
    5.1410274419e-01_f32, 5.1935599017e-01_f32, 5.2458968268e-01_f32, 5.2980362469e-01_f32,
    5.3499761989e-01_f32, 5.4017147273e-01_f32, 5.4532498842e-01_f32, 5.5045797294e-01_f32,
    5.5557023302e-01_f32, 5.6066157620e-01_f32, 5.6573181078e-01_f32, 5.7078074589e-01_f32,
    5.7580819142e-01_f32, 5.8081395810e-01_f32, 5.8579785746e-01_f32, 5.9075970186e-01_f32,
    5.9569930449e-01_f32, 6.0061647938e-01_f32, 6.0551104140e-01_f32, 6.1038280630e-01_f32,
    6.1523159058e-01_f32, 6.2005721176e-01_f32, 6.2485948814e-01_f32, 6.2963823891e-01_f32,
    6.3439328416e-01_f32, 6.3912444486e-01_f32, 6.4383154289e-01_f32, 6.4851440102e-01_f32,
    6.5317284295e-01_f32, 6.5780669330e-01_f32, 6.6241577759e-01_f32, 6.6699992230e-01_f32,
    6.7155895485e-01_f32, 6.7609270358e-01_f32, 6.8060099780e-01_f32, 6.8508366777e-01_f32,
    6.8954054474e-01_f32, 6.9397146089e-01_f32, 6.9837624941e-01_f32, 7.0275474446e-01_f32,
    7.0710678119e-01_f32, 7.1143219575e-01_f32, 7.1573082528e-01_f32, 7.2000250796e-01_f32,
    7.2424708295e-01_f32, 7.2846439045e-01_f32, 7.3265427167e-01_f32, 7.3681656888e-01_f32,
    7.4095112535e-01_f32, 7.4505778544e-01_f32, 7.4913639452e-01_f32, 7.5318679904e-01_f32,
    7.5720884651e-01_f32, 7.6120238548e-01_f32, 7.6516726562e-01_f32, 7.6910333765e-01_f32,
    7.7301045336e-01_f32, 7.7688846567e-01_f32, 7.8073722857e-01_f32, 7.8455659716e-01_f32,
    7.8834642763e-01_f32, 7.9210657730e-01_f32, 7.9583690461e-01_f32, 7.9953726911e-01_f32,
    8.0320753148e-01_f32, 8.0684755354e-01_f32, 8.1045719825e-01_f32, 8.1403632971e-01_f32,
    8.1758481315e-01_f32, 8.2110251499e-01_f32, 8.2458930279e-01_f32, 8.2804504526e-01_f32,
    8.3146961230e-01_f32, 8.3486287499e-01_f32, 8.3822470555e-01_f32, 8.4155497744e-01_f32,
    8.4485356525e-01_f32, 8.4812034480e-01_f32, 8.5135519311e-01_f32, 8.5455798837e-01_f32,
    8.5772861000e-01_f32, 8.6086693864e-01_f32, 8.6397285612e-01_f32, 8.6704624552e-01_f32,
    8.7008699111e-01_f32, 8.7309497842e-01_f32, 8.7607009420e-01_f32, 8.7901222643e-01_f32,
    8.8192126435e-01_f32, 8.8479709843e-01_f32, 8.8763962040e-01_f32, 8.9044872324e-01_f32,
    8.9322430120e-01_f32, 8.9596624976e-01_f32, 8.9867446569e-01_f32, 9.0134884705e-01_f32,
    9.0398929312e-01_f32, 9.0659570451e-01_f32, 9.0916798309e-01_f32, 9.1170603201e-01_f32,
    9.1420975570e-01_f32, 9.1667905992e-01_f32, 9.1911385169e-01_f32, 9.2151403934e-01_f32,
    9.2387953251e-01_f32, 9.2621024214e-01_f32, 9.2850608047e-01_f32, 9.3076696108e-01_f32,
    9.3299279883e-01_f32, 9.3518350994e-01_f32, 9.3733901191e-01_f32, 9.3945922360e-01_f32,
    9.4154406518e-01_f32, 9.4359345816e-01_f32, 9.4560732538e-01_f32, 9.4758559102e-01_f32,
    9.4952818059e-01_f32, 9.5143502097e-01_f32, 9.5330604035e-01_f32, 9.5514116831e-01_f32,
    9.5694033573e-01_f32, 9.5870347490e-01_f32, 9.6043051942e-01_f32, 9.6212140427e-01_f32,
    9.6377606580e-01_f32, 9.6539444170e-01_f32, 9.6697647104e-01_f32, 9.6852209427e-01_f32,
    9.7003125319e-01_f32, 9.7150389099e-01_f32, 9.7293995221e-01_f32, 9.7433938279e-01_f32,
    9.7570213004e-01_f32, 9.7702814266e-01_f32, 9.7831737072e-01_f32, 9.7956976569e-01_f32,
    9.8078528040e-01_f32, 9.8196386911e-01_f32, 9.8310548743e-01_f32, 9.8421009239e-01_f32,
    9.8527764239e-01_f32, 9.8630809724e-01_f32, 9.8730141816e-01_f32, 9.8825756773e-01_f32,
    9.8917650996e-01_f32, 9.9005821026e-01_f32, 9.9090263543e-01_f32, 9.9170975367e-01_f32,
    9.9247953460e-01_f32, 9.9321194923e-01_f32, 9.9390697000e-01_f32, 9.9456457073e-01_f32,
    9.9518472667e-01_f32, 9.9576741447e-01_f32, 9.9631261218e-01_f32, 9.9682029929e-01_f32,
    9.9729045668e-01_f32, 9.9772306664e-01_f32, 9.9811811290e-01_f32, 9.9847558057e-01_f32,
    9.9879545621e-01_f32, 9.9907772775e-01_f32, 9.9932238459e-01_f32, 9.9952941750e-01_f32,
    9.9969881870e-01_f32, 9.9983058180e-01_f32, 9.9992470184e-01_f32, 9.9998117528e-01_f32,
    1.0000000000e+00_f32,
];

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    const EPS: f32 = 1.0e-6;
    const FULL_TURN_PHASE_WORDS: f64 = (u32::MAX as f64) + 1.0;
    const QUARTER_SEGMENTS: f64 = (1_u32 << QUARTER_BITS) as f64;
    const SEGMENT_RADIANS: f64 = core::f64::consts::FRAC_PI_2 / QUARTER_SEGMENTS;
    // Linear interpolation error is bounded by max(|f''(x)|) * h^2 / 8.
    // For sin(x), max(|f''(x)|) <= 1, so the table's segment width h gives
    // h^2 / 8. Add one f32 epsilon for table quantization/arithmetic margin.
    const MAX_INTERPOLATION_ERROR: f64 =
        (SEGMENT_RADIANS * SEGMENT_RADIANS / 8.0) + (f32::EPSILON as f64);
    const PRE_WRAP_EPS: f64 = 1.0e-12;

    #[test]
    fn quadrant_boundaries_are_exact() {
        assert_abs_diff_eq!(sin(0x0000_0000), 0.0, epsilon = EPS);
        assert_abs_diff_eq!(sin(0x4000_0000), 1.0, epsilon = EPS);
        assert_abs_diff_eq!(sin(0x8000_0000), 0.0, epsilon = EPS);
        assert_abs_diff_eq!(sin(0xC000_0000), -1.0, epsilon = EPS);

        assert_abs_diff_eq!(cos(0x0000_0000), 1.0, epsilon = EPS);
        assert_abs_diff_eq!(cos(0x4000_0000), 0.0, epsilon = EPS);
        assert_abs_diff_eq!(cos(0x8000_0000), -1.0, epsilon = EPS);
        assert_abs_diff_eq!(cos(0xC000_0000), 0.0, epsilon = EPS);
    }

    #[test]
    fn max_phase_word_is_just_before_wrap() {
        let radians = (f64::from(u32::MAX) / FULL_TURN_PHASE_WORDS) * core::f64::consts::TAU;
        let sin_value = sin(u32::MAX);

        assert!(sin_value.is_sign_negative());
        assert_abs_diff_eq!(f64::from(sin_value), radians.sin(), epsilon = PRE_WRAP_EPS);
        assert_abs_diff_eq!(
            f64::from(cos(u32::MAX)),
            radians.cos(),
            epsilon = PRE_WRAP_EPS
        );
    }

    #[test]
    fn sin_cos_matches_separate_calls() {
        let phase = 0x1234_5678;
        let (sin_value, cos_value) = sin_cos(phase);

        assert_eq!(sin_value, sin(phase));
        assert_eq!(cos_value, cos(phase));
    }

    #[test]
    fn cosine_is_sine_with_quarter_turn_offset() {
        let phase = 0x1234_5678;

        assert_eq!(cos(phase), sin(phase.wrapping_add(QUARTER_TURN)));
    }

    #[test]
    fn interpolation_matches_float_sine_with_expected_error() {
        let phases = [
            0x0000_0001,
            0x0123_4567,
            0x1FFF_FFFF,
            0x3DA0_0000,
            0x4000_0001,
            0x5555_5555,
            0x7FFF_FFFF,
            0x8000_0001,
            0xAAAA_AAAA,
            0xC000_0001,
            0xFFFF_FFFF,
        ];
        for phase in phases {
            let radians = (f64::from(phase) / FULL_TURN_PHASE_WORDS) * core::f64::consts::TAU;

            assert_abs_diff_eq!(
                f64::from(sin(phase)),
                radians.sin(),
                epsilon = MAX_INTERPOLATION_ERROR
            );
            assert_abs_diff_eq!(
                f64::from(cos(phase)),
                radians.cos(),
                epsilon = MAX_INTERPOLATION_ERROR
            );
        }
    }

    #[cfg(feature = "complex")]
    #[test]
    fn phasor_uses_cosine_as_real_and_sine_as_imaginary() {
        let phase = 0x2000_0000;
        let phasor = self::phasor(phase);

        assert_abs_diff_eq!(phasor.re, cos(phase), epsilon = EPS);
        assert_abs_diff_eq!(phasor.im, sin(phase), epsilon = EPS);
    }
}
