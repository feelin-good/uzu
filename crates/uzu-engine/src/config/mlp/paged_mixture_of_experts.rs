use uzu_engine_macros::uzu_config;

use crate::config::mlp::dense_mlp::DenseMLPConfig;

/// Mixture of experts whose routed experts live in a file on disk and are paged into a
/// GPU-resident cache on demand (flash paging). The shared expert, router and shared-expert
/// gate stay resident. Written by `tools/flash_paging`.
#[uzu_config(super::MLPConfig)]
pub struct PagedMixtureOfExpertsConfig {
    /// Shared expert (resident). Its weights live under `shared_expert` in the parameter tree.
    pub shared_expert_config: DenseMLPConfig,
    pub shared_expert_hidden_dim: u32,
    /// Routed expert MLP (SiLU-gated), MLX-quantized with `expert_bits`/`expert_group_size`.
    pub expert_config: DenseMLPConfig,
    pub expert_hidden_dim: u32,
    pub expert_bits: u32,
    pub expert_group_size: u32,
    pub num_routed_experts: u32,
    pub num_active_routed_experts: u32,
    /// Renormalize the top-k softmax probabilities so they sum to one.
    pub normalize_top_k: bool,
    /// Absolute path of the expert file and this layer's index in it.
    pub experts_file: String,
    pub layer_index: u32,
    /// Byte size of one (layer, expert) record and the offsets of its parts, in uzu MLX layout.
    pub record_size: u64,
    pub up_weights_offset: u64,
    pub up_scales_offset: u64,
    pub up_biases_offset: u64,
    pub down_weights_offset: u64,
    pub down_scales_offset: u64,
    pub down_biases_offset: u64,
}
