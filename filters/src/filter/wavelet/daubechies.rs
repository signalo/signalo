// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Daubechies filters.

use super::Wavelet;

/// Trait for Daubechies wavelet filters.
pub trait Daubechies: Sized {
    /// Creates a wavelet filter pre-configured with the Daubechies coefficients.
    fn daubechies() -> Self;
}

macro_rules! daubechies_impl_float {
    ($width:expr => [$($scale:expr),* $(,)*]) => {
        impl Daubechies for Wavelet<[f32; $width]> {
            fn daubechies() -> Self {
                let scale: [f32; $width] = [$($scale),*];
                // The translation coefficients are derived by taking the scale coefficients:
                let mut translate = scale.clone();
                // reversing their order:
                translate.reverse();
                // and then inverting the sign of every even-indexed coefficient:
                for (index, coeff) in translate.iter_mut().enumerate() {
                    if index & 2 == 0 {
                        *coeff = 0.0 - *coeff;
                    }
                }
                Wavelet::new(scale, translate)
            }
        }
        impl Daubechies for Wavelet<[f64; $width]> {
            fn daubechies() -> Self {
                let scale: [f64; $width] = [$($scale),*];
                // The translation coefficients are derived by taking the scale coefficients:
                let mut translate = scale.clone();
                // reversing their order:
                translate.reverse();
                // and then inverting the sign of every even-indexed coefficient:
                for (index, coeff) in translate.iter_mut().enumerate() {
                    if index & 2 == 0 {
                        *coeff = 0.0 - *coeff;
                    }
                }
                Wavelet::new(scale, translate)
            }
        }
    };
}

// Source: http://wavelets.pybytes.com/wavelet/db1/

daubechies_impl_float!(2 => [
    0.7071067812,
    0.7071067812,
]);
daubechies_impl_float!(4 => [
    0.4829629131,
    0.8365163037,
    0.2241438680,
    -0.1294095226,

    // -0.1294095226,
    // 0.2241438680,
    // 0.8365163037,
    // 0.4829629131,
]);
daubechies_impl_float!(6 => [
    0.0352262919,
    -0.0854412739,
    -0.1350110200,
    0.4598775021,
    0.8068915093,
    0.3326705530,
]);
daubechies_impl_float!(8 => [
    -0.0105974018,
    0.0328830117,
    0.0308413818,
    -0.1870348117,
    -0.0279837694,
    0.6308807679,
    0.7148465706,
    0.2303778133,
]);
daubechies_impl_float!(10 => [
    0.0033357253,
    -0.0125807520,
    -0.0062414902,
    0.0775714938,
    -0.0322448696,
    -0.2422948871,
    0.1384281459,
    0.7243085284,
    0.6038292698,
    0.1601023980,
]);
daubechies_impl_float!(12 => [
    -0.0010773011,
    0.0047772575,
    0.0005538422,
    -0.0315820393,
    0.0275228655,
    0.0975016056,
    -0.1297668676,
    -0.2262646940,
    0.3152503517,
    0.7511339080,
    0.4946238904,
    0.1115407434,
]);
daubechies_impl_float!(14 => [
    0.0003537138,
    -0.0018016407,
    0.0004295780,
    0.0125509986,
    -0.0165745416,
    -0.0380299369,
    0.0806126092,
    0.0713092193,
    -0.2240361850,
    -0.1439060039,
    0.4697822874,
    0.7291320908,
    0.3965393195,
    0.0778520541,
]);
daubechies_impl_float!(16 => [
    -0.0001174768,
    0.0006754494,
    -0.0003917404,
    -0.0048703530,
    0.0087460940,
    0.0139810279,
    -0.0440882539,
    -0.0173693010,
    0.1287474266,
    0.0004724846,
    -0.2840155430,
    -0.0158291053,
    0.5853546837,
    0.6756307363,
    0.3128715909,
    0.0544158422,
]);
daubechies_impl_float!(18 => [
    0.0000393473,
    -0.0002519632,
    0.0002303858,
    0.0018476469,
    -0.0042815037,
    -0.0047232048,
    0.0223616621,
    0.0002509471,
    -0.0676328291,
    0.0307256815,
    0.1485407493,
    -0.0968407832,
    -0.2932737833,
    0.1331973858,
    0.6572880780,
    0.6048231237,
    0.2438346746,
    0.0380779474,
]);
daubechies_impl_float!(20 => [
    -0.0000132642,
    0.0000935887,
    -0.0001164669,
    -0.0006858567,
    0.0019924053,
    0.0013953517,
    -0.0107331755,
    0.0036065536,
    0.0332126741,
    -0.0294575368,
    -0.0713941472,
    0.0930573646,
    0.1273693403,
    -0.1959462744,
    -0.2498464243,
    0.2811723437,
    0.6884590395,
    0.5272011889,
    0.1881768001,
    0.0266700579,
]);

#[cfg(test)]
mod tests {
    use super::*;

    use signalo_traits::filter::Filter;

    fn get_input() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output_2() -> Vec<(f32, f32)> {
        vec![
            (0.000, 0.000), (0.707, 0.707), (5.657, 4.243), (6.364, -3.536), (4.950, 2.121),
            (9.192, 2.121), (16.971, 5.657), (20.506, -2.121), (22.627, 4.243), (17.678, -9.192),
            (14.142, 5.657), (16.263, -3.536), (12.728, 0.000), (18.385, 5.657), (24.042, 0.000),
            (14.849, -9.192), (11.314, 5.657), (22.627, 5.657), (28.284, 0.000), (19.092, -9.192),
            (9.899, 0.000), (15.556, 5.657), (21.213, 0.000), (17.678, -3.536), (23.335, 9.192),
            (23.335, -9.192), (85.560, 71.418), (205.768, 48.790), (203.647, -50.912),
            (89.095, -63.640), (87.681, 62.225), (78.489, -71.418), (21.920, 14.849),
            (27.577, -9.192), (18.385, 0.000), (24.042, 5.657), (29.698, 0.000), (29.698, 0.000),
            (38.891, 9.192), (29.698, -18.385), (82.731, 71.418), (82.731, -71.418),
            (26.163, 14.849), (31.820, -9.192), (22.627, 0.000), (22.627, 0.000), (84.853, 62.225),
            (81.317, -65.761), (24.749, 9.192), (33.941, 0.000)
        ]
    }
    
    fn get_output_4() -> Vec<(f32, f32)> {
        vec![
            (0.000, 0.000), (0.483, -0.129), (4.217, -1.130), (7.046, -0.991), (5.527, 4.277),
            (7.589, -3.864), (15.281, -0.647), (20.809, -0.991), (22.602, 4.148), (19.635, -1.888),
            (14.357, 6.459), (14.944, -8.460), (14.237, 5.631), (15.945, -3.450), (23.284, -2.828),
            (18.798, 5.546), (10.752, 3.561), (18.394, -10.142), (28.562, 1.035), (23.041, 5.546),
            (11.131, 4.596), (12.081, -7.314), (20.455, -2.828), (19.834, 4.511), (20.894, 0.085),
            (24.370, -5.329), (65.835, -2.196), (179.052, -50.917), (226.319, 33.702),
            (125.019, 70.427), (67.165, -14.342), (81.144, -61.509), (38.169, 75.492),
            (15.431, -54.522), (22.334, 14.738), (20.566, -7.314), (28.941, -2.828),
            (30.734, 3.864), (35.977, -1.682), (34.295, -1.232), (64.238, 2.401),
            (92.437, -35.196), (44.094, 81.771), (19.674, -54.522), (26.577, 14.738),
            (20.945, -6.279), (65.128, -11.388), (93.826, -19.078), (42.033, 73.699),
            (20.674, -49.512)
        ]
    }

    fn get_output_6() -> Vec<(f32, f32)> {
        vec![
            (0.000, 0.000), (0.035, -0.333), (0.161, -1.522), (-0.663, 4.523),
            (-0.480, -3.404), (3.611, -0.406), (6.106, -0.804), (4.253, 4.649),
            (5.777, -3.772), (12.309, 6.056), (18.965, -8.660), (22.861, 4.652),
            (20.073, -0.899), (16.215, -3.240), (15.363, 4.115), (12.452, 4.616),
            (15.860, -9.792), (23.668, 0.664), (18.588, 5.477), (10.239, 3.932),
            (17.159, -7.412), (28.313, -3.130), (24.312, 5.362), (12.566, 2.236),
            (12.806, -7.661), (18.367, 10.136), (21.643, -38.975), (19.143, 23.363),
            (-0.077, 59.233), (33.871, -14.943), (163.284, -84.872), (220.523, 80.298),
            (141.480, -40.224), (96.761, 5.386), (89.655, 3.224), (49.128, -1.824),
            (25.811, 4.623), (21.228, 0.573), (21.040, -5.290), (25.468, 14.532),
            (32.154, -45.743), (29.455, 79.554), (23.721, -50.754), (64.898, 1.563),
            (89.573, 2.766), (53.089, 0.837), (33.556, -28.446), (19.257, 73.128),
            (11.455, -47.165), (63.370, -5.788)
        ]
    }

    fn get_output_8() -> Vec<(f32, f32)> {
        vec![
            (0.000, 0.000), (-0.011, -0.230), (-0.041, -0.898), (0.240, 3.912),
            (0.042, -4.166), (-1.196, 0.461), (0.309, 0.162), (4.775, 3.813),
            (5.579, -4.646), (3.942, 5.061), (6.838, -8.233), (13.597, 6.226),
            (20.460, -1.296), (22.573, -2.441), (18.944, 4.347), (16.604, 1.970),
            (14.165, -9.669), (11.988, 3.984), (18.824, 5.319), (23.795, 0.349),
            (15.289, -8.157), (10.333, 0.372), (20.799, 6.576), (29.024, -0.099),
            (20.867, -7.378), (11.761, 9.988), (13.652, -30.250), (19.268, 32.693),
            (26.171, 37.442), (12.986, -41.516), (-8.919, -64.518), (61.226, 96.761),
            (191.979, -47.797), (202.921, 9.073), (127.217, 4.677), (100.224, -2.006),
            (80.785, 2.949), (42.718, -1.493), (24.626, -4.733), (20.650, 12.251),
            (21.178, -37.420), (28.055, 73.824), (33.968, -63.858), (21.720, 10.579),
            (30.894, 8.525), (77.520, 1.080), (80.922, -21.062), (49.482, 63.737),
            (32.542, -61.277), (9.187, 4.430)
        ]
    }

    #[test]
    fn daubechies_2() {
        let filter: Wavelet<[f32; 2]> = Wavelet::daubechies();
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();

        let expected = get_output_2();

        panic!("{:?}", output);
        
        let output_sums: Vec<_> = output.iter().map(|(sum, _)| *sum).collect();
        let expected_sums: Vec<_> = expected.iter().map(|(sum, _)| *sum).collect();

        let output_differences: Vec<_> = output.iter().map(|(_, diff)| *diff).collect();
        let expected_differences: Vec<_> = expected.iter().map(|(_, diff)| *diff).collect();

        assert_nearly_eq!(output_sums, expected_sums, 0.001);
        assert_nearly_eq!(output_differences, expected_differences, 0.001);
    }

    #[test]
    fn daubechies_4() {
        let filter: Wavelet<[f32; 4]> = Wavelet::daubechies();
        let input = get_input();

        // panic!("{:?}", input);

        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();

        let expected = get_output_4();

        panic!("{:?}", output);

        let output_sums: Vec<_> = output.iter().map(|(sum, _)| *sum).collect();
        let expected_sums: Vec<_> = expected.iter().map(|(sum, _)| *sum).collect();

        let output_differences: Vec<_> = output.iter().map(|(_, diff)| *diff).collect();
        let expected_differences: Vec<_> = expected.iter().map(|(_, diff)| *diff).collect();

        assert_nearly_eq!(output_sums, expected_sums, 0.001);
        assert_nearly_eq!(output_differences, expected_differences, 0.001);
    }

    // #[test]
    // fn daubechies_6() {
    //     let filter: Wavelet<[f32; 6]> = Wavelet::daubechies();
    //     let input = get_input();
    //     let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
    //         Some(filter.filter(input))
    //     }).collect();

    //     let expected = get_output_6();

    //     let output_sums: Vec<_> = output.iter().map(|(sum, _)| *sum).collect();
    //     let expected_sums: Vec<_> = expected.iter().map(|(sum, _)| *sum).collect();

    //     let output_differences: Vec<_> = output.iter().map(|(_, diff)| *diff).collect();
    //     let expected_differences: Vec<_> = expected.iter().map(|(_, diff)| *diff).collect();

    //     assert_nearly_eq!(output_sums, expected_sums, 0.001);
    //     assert_nearly_eq!(output_differences, expected_differences, 0.001);
    // }

    // #[test]
    // fn daubechies_8() {
    //     let filter: Wavelet<[f32; 8]> = Wavelet::daubechies();
    //     let input = get_input();
    //     let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
    //         Some(filter.filter(input))
    //     }).collect();

    //     let expected = get_output_8();

    //     let output_sums: Vec<_> = output.iter().map(|(sum, _)| *sum).collect();
    //     let expected_sums: Vec<_> = expected.iter().map(|(sum, _)| *sum).collect();

    //     let output_differences: Vec<_> = output.iter().map(|(_, diff)| *diff).collect();
    //     let expected_differences: Vec<_> = expected.iter().map(|(_, diff)| *diff).collect();

    //     assert_nearly_eq!(output_sums, expected_sums, 0.001);
    //     assert_nearly_eq!(output_differences, expected_differences, 0.001);
    // }
}
