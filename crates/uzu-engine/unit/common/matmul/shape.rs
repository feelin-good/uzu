use derive_more::Display;

use super::RegistryCase;
use crate::backends::common::{
    gpu_types::QuantizationMethod::{ScaleBias, ScaleSymmetric, ScaleZeroPoint},
    kernel::matmul::QuantParamsLayout::{GroupOutput, OutputGroup},
};

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
#[display("M[{m}]K[{k}]N[{n}]")]
pub struct Shape {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

impl Shape {
    pub const fn new(
        m: u32,
        k: u32,
        n: u32,
    ) -> Self {
        Self {
            m,
            k,
            n,
        }
    }
}

const SHAPES_TINY: &[Shape] = &[Shape::new(64, 64, 64), Shape::new(16, 128, 256), Shape::new(128, 256, 128)];

const SHAPES_UNALIGNED: &[Shape] = &[
    Shape::new(7, 33, 11),
    Shape::new(33, 2048, 2048),
    Shape::new(64, 2048, 33),
    Shape::new(200, 2048, 2048),
    Shape::new(128, 2048, 200),
];

const SHAPES_MEDIUM: &[Shape] = &[Shape::new(128, 2048, 2048), Shape::new(256, 4096, 4096)];

pub fn all_correctness_shapes() -> impl Iterator<Item = Shape> {
    SHAPES_TINY.iter().chain(SHAPES_UNALIGNED.iter()).chain(SHAPES_MEDIUM.iter()).copied()
}

const BENCH_FP_GEMM: &[Shape] = &[
    Shape::new(128, 2048, 8192),
    Shape::new(128, 4096, 14336),
    Shape::new(256, 4096, 4096),
    Shape::new(512, 8192, 2048),
];

pub fn bench_fp_gemm_shapes() -> impl Iterator<Item = Shape> {
    BENCH_FP_GEMM.iter().copied()
}

pub fn bench_quant_gemv_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: u32 = if bits == 4 {
        512
    } else {
        256
    };
    let nk: &[(u32, u32)] = &[(4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];
    let ms = &[1u32, 2, 4];
    nk.iter()
        .filter(move |&&(n, k)| n % 8 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}

pub const REGISTRY_MAX_BATCH: u32 = 32;

#[rustfmt::skip]
const REGISTRY_CASES: &[RegistryCase] = &[
    ("w4zp_g32_tied_readout_n248320_k1024", ScaleZeroPoint, 4, 32, OutputGroup, false, false, 248320, 1024),
    ("w4zp_g32_tied_readout_n248320_k2048", ScaleZeroPoint, 4, 32, OutputGroup, false, false, 248320, 2048),
    ("w4zp_g32_tied_readout_n248320_k2560", ScaleZeroPoint, 4, 32, OutputGroup, false, false, 248320, 2560),
    ("w4zp_g32_linear_n1024_k2048", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 1024, 2048),
    ("w4zp_g32_linear_n1024_k3584", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 1024, 3584),
    ("w4zp_g32_linear_n2048_k2048", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 2048, 2048),
    ("w4zp_g32_linear_n2048_k6144", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 2048, 6144),
    ("w4zp_g32_linear_n2560_k4096", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 2560, 4096),
    ("w4zp_g32_linear_n2560_k9216", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 2560, 9216),
    ("w4zp_g32_linear_n4096_k4096", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 4096, 4096),
    ("w4zp_g32_linear_n4096_k12288", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 4096, 12288),
    ("w4zp_g32_linear_n5120_k1024", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 5120, 1024),
    ("w4zp_g32_linear_n5120_k2048", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 5120, 2048),
    ("w4zp_g32_linear_n7168_k1024", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 7168, 1024),
    ("w4zp_g32_linear_n8224_k1024", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 8224, 1024),
    ("w4zp_g32_linear_n8224_k2048", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 8224, 2048),
    ("w4zp_g32_linear_n10240_k2560", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 10240, 2560),
    ("w4zp_g32_linear_n10240_k4096", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 10240, 4096),
    ("w4zp_g32_linear_n12288_k2048", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 12288, 2048),
    ("w4zp_g32_linear_n12352_k2560", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 12352, 2560),
    ("w4zp_g32_linear_n12352_k4096", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 12352, 4096),
    ("w4zp_g32_linear_n18432_k2560", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 18432, 2560),
    ("w4zp_g32_linear_n24576_k4096", ScaleZeroPoint, 4, 32, GroupOutput, true, true, 24576, 4096),
    ("w4zp_g32_untied_readout_n248320_k4096", ScaleZeroPoint, 4, 32, GroupOutput, true, false, 248320, 4096),
    ("w4zp_g64_tied_readout_n65536_k1024", ScaleZeroPoint, 4, 64, OutputGroup, false, false, 65536, 1024),
    ("w4zp_g64_tied_readout_n65536_k2048", ScaleZeroPoint, 4, 64, OutputGroup, false, false, 65536, 2048),
    ("w4zp_g64_tied_readout_n128000_k2048", ScaleZeroPoint, 4, 64, OutputGroup, false, false, 128000, 2048),
    ("w4zp_g64_linear_n1024_k1024", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 1024, 1024),
    ("w4zp_g64_linear_n1024_k2560", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 1024, 2560),
    ("w4zp_g64_linear_n1024_k4608", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 1024, 4608),
    ("w4zp_g64_linear_n1280_k5120", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 1280, 5120),
    ("w4zp_g64_linear_n1664_k6656", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 1664, 6656),
    ("w4zp_g64_linear_n2048_k1024", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 2048, 1024),
    ("w4zp_g64_linear_n2048_k2048", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 2048, 2048),
    ("w4zp_g64_linear_n2048_k4096", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 2048, 4096),
    ("w4zp_g64_linear_n2048_k5120", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 2048, 5120),
    ("w4zp_g64_linear_n2048_k6656", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 2048, 6656),
    ("w4zp_g64_linear_n2048_k8192", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 2048, 8192),
    ("w4zp_g64_linear_n2048_k10752", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 2048, 10752),
    ("w4zp_g64_linear_n3072_k1024", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 3072, 1024),
    ("w4zp_g64_linear_n3072_k2048", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 3072, 2048),
    ("w4zp_g64_linear_n4096_k2048", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 4096, 2048),
    ("w4zp_g64_linear_n4096_k4096", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 4096, 4096),
    ("w4zp_g64_linear_n4096_k12288", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 4096, 12288),
    ("w4zp_g64_linear_n4096_k32768", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 4096, 32768),
    ("w4zp_g64_linear_n5120_k1024", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 5120, 1024),
    ("w4zp_g64_linear_n5120_k2048", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 5120, 2048),
    ("w4zp_g64_linear_n5120_k4096", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 5120, 4096),
    ("w4zp_g64_linear_n5120_k6144", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 5120, 6144),
    ("w4zp_g64_linear_n5120_k17408", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 5120, 17408),
    ("w4zp_g64_linear_n5120_k25600", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 5120, 25600),
    ("w4zp_g64_linear_n6144_k2048", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 6144, 2048),
    ("w4zp_g64_linear_n6144_k4096", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 6144, 4096),
    ("w4zp_g64_linear_n6144_k5120", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 6144, 5120),
    ("w4zp_g64_linear_n6144_k6656", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 6144, 6656),
    ("w4zp_g64_linear_n6656_k2048", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 6656, 2048),
    ("w4zp_g64_linear_n6656_k4096", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 6656, 4096),
    ("w4zp_g64_linear_n6656_k19968", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 6656, 19968),
    ("w4zp_g64_linear_n6656_k33280", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 6656, 33280),
    ("w4zp_g64_linear_n8704_k6656", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 8704, 6656),
    ("w4zp_g64_linear_n9216_k1024", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 9216, 1024),
    ("w4zp_g64_linear_n10240_k5120", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 10240, 5120),
    ("w4zp_g64_linear_n10240_k6656", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 10240, 6656),
    ("w4zp_g64_linear_n12288_k4096", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 12288, 4096),
    ("w4zp_g64_linear_n14336_k5120", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 14336, 5120),
    ("w4zp_g64_linear_n16384_k2048", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 16384, 2048),
    ("w4zp_g64_linear_n16480_k5120", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 16480, 5120),
    ("w4zp_g64_linear_n21504_k2048", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 21504, 2048),
    ("w4zp_g64_linear_n24576_k4096", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 24576, 4096),
    ("w4zp_g64_linear_n34816_k5120", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 34816, 5120),
    ("w4zp_g64_linear_n39936_k6656", ScaleZeroPoint, 4, 64, GroupOutput, true, true, 39936, 6656),
    ("w4zp_g64_untied_readout_n202048_k6656", ScaleZeroPoint, 4, 64, GroupOutput, true, false, 202048, 6656),
    ("w4zp_g64_untied_readout_n248320_k5120", ScaleZeroPoint, 4, 64, GroupOutput, true, false, 248320, 5120),
    ("w8sym_g32_tied_readout_n248320_k1024", ScaleSymmetric, 8, 32, OutputGroup, false, false, 248320, 1024),
    ("w8sym_g32_tied_readout_n248320_k2048", ScaleSymmetric, 8, 32, OutputGroup, false, false, 248320, 2048),
    ("w8sym_g32_tied_readout_n248320_k2560", ScaleSymmetric, 8, 32, OutputGroup, false, false, 248320, 2560),
    ("w8sym_g32_linear_n1024_k2048", ScaleSymmetric, 8, 32, GroupOutput, true, true, 1024, 2048),
    ("w8sym_g32_linear_n1024_k3584", ScaleSymmetric, 8, 32, GroupOutput, true, true, 1024, 3584),
    ("w8sym_g32_linear_n2048_k2048", ScaleSymmetric, 8, 32, GroupOutput, true, true, 2048, 2048),
    ("w8sym_g32_linear_n2048_k6144", ScaleSymmetric, 8, 32, GroupOutput, true, true, 2048, 6144),
    ("w8sym_g32_linear_n2560_k4096", ScaleSymmetric, 8, 32, GroupOutput, true, true, 2560, 4096),
    ("w8sym_g32_linear_n2560_k9216", ScaleSymmetric, 8, 32, GroupOutput, true, true, 2560, 9216),
    ("w8sym_g32_linear_n5120_k1024", ScaleSymmetric, 8, 32, GroupOutput, true, true, 5120, 1024),
    ("w8sym_g32_linear_n5120_k2048", ScaleSymmetric, 8, 32, GroupOutput, true, true, 5120, 2048),
    ("w8sym_g32_linear_n7168_k1024", ScaleSymmetric, 8, 32, GroupOutput, true, true, 7168, 1024),
    ("w8sym_g32_linear_n8224_k1024", ScaleSymmetric, 8, 32, GroupOutput, true, true, 8224, 1024),
    ("w8sym_g32_linear_n8224_k2048", ScaleSymmetric, 8, 32, GroupOutput, true, true, 8224, 2048),
    ("w8sym_g32_linear_n10240_k2560", ScaleSymmetric, 8, 32, GroupOutput, true, true, 10240, 2560),
    ("w8sym_g32_linear_n12288_k2048", ScaleSymmetric, 8, 32, GroupOutput, true, true, 12288, 2048),
    ("w8sym_g32_linear_n12352_k2560", ScaleSymmetric, 8, 32, GroupOutput, true, true, 12352, 2560),
    ("w8sym_g32_linear_n18432_k2560", ScaleSymmetric, 8, 32, GroupOutput, true, true, 18432, 2560),
    ("w8sym_g64_tied_readout_n65536_k1024", ScaleSymmetric, 8, 64, OutputGroup, false, false, 65536, 1024),
    ("w8sym_g64_tied_readout_n65536_k2048", ScaleSymmetric, 8, 64, OutputGroup, false, false, 65536, 2048),
    ("w8sym_g64_tied_readout_n128000_k2048", ScaleSymmetric, 8, 64, OutputGroup, false, false, 128000, 2048),
    ("w8sym_g64_linear_n1024_k1024", ScaleSymmetric, 8, 64, GroupOutput, true, true, 1024, 1024),
    ("w8sym_g64_linear_n1024_k2560", ScaleSymmetric, 8, 64, GroupOutput, true, true, 1024, 2560),
    ("w8sym_g64_linear_n1024_k4608", ScaleSymmetric, 8, 64, GroupOutput, true, true, 1024, 4608),
    ("w8sym_g64_linear_n2048_k1024", ScaleSymmetric, 8, 64, GroupOutput, true, true, 2048, 1024),
    ("w8sym_g64_linear_n2048_k2048", ScaleSymmetric, 8, 64, GroupOutput, true, true, 2048, 2048),
    ("w8sym_g64_linear_n2048_k8192", ScaleSymmetric, 8, 64, GroupOutput, true, true, 2048, 8192),
    ("w8sym_g64_linear_n2048_k10752", ScaleSymmetric, 8, 64, GroupOutput, true, true, 2048, 10752),
    ("w8sym_g64_linear_n3072_k1024", ScaleSymmetric, 8, 64, GroupOutput, true, true, 3072, 1024),
    ("w8sym_g64_linear_n3072_k2048", ScaleSymmetric, 8, 64, GroupOutput, true, true, 3072, 2048),
    ("w8sym_g64_linear_n4096_k4096", ScaleSymmetric, 8, 64, GroupOutput, true, true, 4096, 4096),
    ("w8sym_g64_linear_n4096_k12288", ScaleSymmetric, 8, 64, GroupOutput, true, true, 4096, 12288),
    ("w8sym_g64_linear_n5120_k1024", ScaleSymmetric, 8, 64, GroupOutput, true, true, 5120, 1024),
    ("w8sym_g64_linear_n5120_k6144", ScaleSymmetric, 8, 64, GroupOutput, true, true, 5120, 6144),
    ("w8sym_g64_linear_n5120_k17408", ScaleSymmetric, 8, 64, GroupOutput, true, true, 5120, 17408),
    ("w8sym_g64_linear_n6144_k2048", ScaleSymmetric, 8, 64, GroupOutput, true, true, 6144, 2048),
    ("w8sym_g64_linear_n6656_k4096", ScaleSymmetric, 8, 64, GroupOutput, true, true, 6656, 4096),
    ("w8sym_g64_linear_n6656_k19968", ScaleSymmetric, 8, 64, GroupOutput, true, true, 6656, 19968),
    ("w8sym_g64_linear_n8704_k6656", ScaleSymmetric, 8, 64, GroupOutput, true, true, 8704, 6656),
    ("w8sym_g64_linear_n9216_k1024", ScaleSymmetric, 8, 64, GroupOutput, true, true, 9216, 1024),
    ("w8sym_g64_linear_n10240_k4096", ScaleSymmetric, 8, 64, GroupOutput, true, true, 10240, 4096),
    ("w8sym_g64_linear_n12352_k4096", ScaleSymmetric, 8, 64, GroupOutput, true, true, 12352, 4096),
    ("w8sym_g64_linear_n14336_k5120", ScaleSymmetric, 8, 64, GroupOutput, true, true, 14336, 5120),
    ("w8sym_g64_linear_n16384_k2048", ScaleSymmetric, 8, 64, GroupOutput, true, true, 16384, 2048),
    ("w8sym_g64_linear_n16480_k5120", ScaleSymmetric, 8, 64, GroupOutput, true, true, 16480, 5120),
    ("w8sym_g64_linear_n21504_k2048", ScaleSymmetric, 8, 64, GroupOutput, true, true, 21504, 2048),
    ("w8sym_g64_linear_n24576_k4096", ScaleSymmetric, 8, 64, GroupOutput, true, true, 24576, 4096),
    ("w8sym_g64_linear_n34816_k5120", ScaleSymmetric, 8, 64, GroupOutput, true, true, 34816, 5120),
    ("w8sym_g64_linear_n39936_k6656", ScaleSymmetric, 8, 64, GroupOutput, true, true, 39936, 6656),
    ("w8sym_g64_untied_readout_n202048_k6656", ScaleSymmetric, 8, 64, GroupOutput, true, false, 202048, 6656),
    ("w8sym_g64_untied_readout_n248320_k4096", ScaleSymmetric, 8, 64, GroupOutput, true, false, 248320, 4096),
    ("w8sym_g64_untied_readout_n248320_k5120", ScaleSymmetric, 8, 64, GroupOutput, true, false, 248320, 5120),
    ("w4bias_g32_diag_n4096_k4096", ScaleBias, 4, 32, OutputGroup, false, false, 4096, 4096),
    ("w4bias_g32_diag_n14336_k4096", ScaleBias, 4, 32, OutputGroup, false, false, 14336, 4096),
    ("w4bias_g32_diag_n4096_k14336", ScaleBias, 4, 32, OutputGroup, false, false, 4096, 14336),
    ("w4bias_g64_diag_n4096_k4096", ScaleBias, 4, 64, OutputGroup, false, false, 4096, 4096),
    ("w4bias_g64_diag_n14336_k4096", ScaleBias, 4, 64, OutputGroup, false, false, 14336, 4096),
    ("w4bias_g64_diag_n4096_k14336", ScaleBias, 4, 64, OutputGroup, false, false, 4096, 14336),
];

pub fn registry_layer_shapes() -> impl Iterator<Item = (&'static RegistryCase, impl Iterator<Item = Shape>)> {
    REGISTRY_CASES.iter().map(|case| (case, (1..=REGISTRY_MAX_BATCH).map(move |m| Shape::new(m, case.8, case.7))))
}
