pub mod bench;
pub mod harness;
pub mod quant;
pub mod shape;

pub use bench::{iter_encode_loop, iter_encode_loop_named};
pub use harness::{Case, cpu_reference, deterministic_input};
pub use quant::{QuantBuffers, QuantInput, quant_arguments, run_quant_cpu, transpose_metadata};
pub use shape::{
    REGISTRY_MAX_BATCH, Shape, all_correctness_shapes, bench_fp_gemm_shapes, bench_quant_gemv_shapes,
    registry_layer_shapes,
};

use crate::backends::common::{gpu_types::QuantizationMethod, kernel::matmul::QuantParamsLayout};
#[cfg(backend = "metal")]
use crate::backends::{
    common::{Backend, kernel::Kernels},
    metal::Metal,
};

pub type RegistryCase = (&'static str, QuantizationMethod, u32, u32, QuantParamsLayout, bool, bool, u32, u32);
#[cfg(backend = "metal")]
pub type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;
