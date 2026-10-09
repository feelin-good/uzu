use super::GemmEngine;
use crate::backends::common::gpu_types::gemm::GemmTiling;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmPlan {
    pub engine: GemmEngine,
    pub tiling: GemmTiling,
    pub split_k: u32,
}
