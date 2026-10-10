//! Flash-paged MoE block: routed experts live in a file on disk and are paged into a
//! GPU-resident LRU cache on demand. See FLASH_PAGING.md.
//!
//! Per forward pass and layer:
//! 1. One bf16 matmul computes router logits plus the shared-expert gate logit.
//! 2. `flush_and_wait`, then softmax/top-k/sigmoid run on the CPU over those logits.
//! 3. Rows are bucketed by expert (CPU), gathered on the GPU, and every selected expert runs
//!    as two MLX 4-bit matmuls straight out of its cache slot. Missing experts are pread
//!    into free slots first; if a batch needs more experts than there are slots, the work
//!    is split into waves with a flush in between.
//! 4. The finalize kernel combines the expert rows with their probabilities. The shared
//!    expert takes part as one extra "expert" per token whose weight is its sigmoid gate.

#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::{
    collections::HashMap,
    fs::File,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use half::bf16;
use parking_lot::Mutex;

use super::{MoeBlockError, gather::MoeGather};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        gpu_types::QuantizationMode,
        kernel::{
            MoeFinalizeKernel,
            matmul::{
                ActivationFormat, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput,
                QuantParams, QuantParamsLayout, QuantizedB, QuantizedCorrection,
            },
        },
    },
    config::mlp::{AnyMLPConfig, paged_mixture_of_experts::PagedMixtureOfExpertsConfig},
    data_type::DataType,
    encodable_block::{
        linear::LinearInput,
        mlp::{Mlp, gate_act_mul::MlpGateActMulEncodable},
    },
    parameters::ParameterTree,
};

/// Time spent across all paged layers since the last stats line, in nanoseconds.
static SYNC_NS: AtomicU64 = AtomicU64::new(0);
static IO_NS: AtomicU64 = AtomicU64::new(0);
static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static PASSES: AtomicU64 = AtomicU64::new(0);

/// Default GPU memory for the expert cache across all layers, in MiB.
/// Override with `UZU_EXPERT_CACHE_MB`.
const DEFAULT_EXPERT_CACHE_MB: u64 = 8192;

struct ExpertCache<B: Backend> {
    file: File,
    slots: B::GlobalBuffer,
    slot_count: usize,
    record_size: usize,
    file_base: u64,
    /// expert -> slot
    resident: HashMap<u32, usize>,
    /// slot -> (expert, last use tick)
    slot_owner: Vec<Option<(u32, u64)>>,
    /// Slots referenced by work encoded since the last flush; never evicted.
    pinned: Vec<bool>,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl<B: Backend> ExpertCache<B> {
    /// Returns the slot for `expert`, loading it if needed. `None` means every slot is
    /// pinned and the caller must flush before more experts can be loaded.
    fn acquire(
        &mut self,
        expert: u32,
    ) -> std::io::Result<Option<usize>> {
        self.tick += 1;
        if let Some(&slot) = self.resident.get(&expert) {
            self.hits += 1;
            self.slot_owner[slot] = Some((expert, self.tick));
            self.pinned[slot] = true;
            return Ok(Some(slot));
        }
        let Some(slot) = (0..self.slot_count)
            .filter(|&slot| !self.pinned[slot])
            .min_by_key(|&slot| self.slot_owner[slot].map_or(0, |(_, tick)| tick))
        else {
            return Ok(None);
        };
        if let Some((old, _)) = self.slot_owner[slot].take() {
            self.resident.remove(&old);
        }
        self.misses += 1;
        let range = slot * self.record_size..(slot + 1) * self.record_size;
        let destination = (&mut self.slots).subrange_mut(range).as_slice_mut::<u8>();
        self.file.read_exact_at(destination, self.file_base + expert as u64 * self.record_size as u64)?;
        self.resident.insert(expert, slot);
        self.slot_owner[slot] = Some((expert, self.tick));
        self.pinned[slot] = true;
        Ok(Some(slot))
    }

    fn unpin_all(&mut self) {
        self.pinned.fill(false);
    }
}

/// CPU-written index buffers, grown on demand.
struct Routing<B: Backend> {
    capacity_tokens: u32,
    logits: B::GlobalBuffer,
    bucketed_ids: B::GlobalBuffer,
    sumk: B::GlobalBuffer,
    tok2row: B::GlobalBuffer,
    probs: B::GlobalBuffer,
}

pub struct PagedMoeBlock<B: Backend> {
    name: String,
    shared_expert: Box<dyn Mlp<B>>,
    router: B::GlobalBuffer,
    router_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    up_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    down_kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    gate: MlpGateActMulEncodable<B>,
    gather: MoeGather<B>,
    finalize: <B::Kernels as Kernels>::MoeFinalizeKernel,
    cache: Mutex<ExpertCache<B>>,
    routing: Mutex<Option<Routing<B>>>,
    model_dim: u32,
    hidden_dim: u32,
    num_experts: u32,
    top_k: u32,
    normalize_top_k: bool,
    group_size: u32,
    up_parts: [Range<usize>; 3],
    down_parts: [Range<usize>; 3],
    data_type: DataType,
}

fn invalid<B: Backend>(message: impl Into<String>) -> MoeBlockError<B> {
    MoeBlockError::UnsupportedRouterConfiguration(message.into())
}

impl<B: Backend> PagedMoeBlock<B> {
    pub fn new(
        name: String,
        context: &B::Context,
        config: &PagedMixtureOfExpertsConfig,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, MoeBlockError<B>> {
        if data_type != DataType::BF16 {
            return Err(invalid(format!("paged MoE needs bf16 activations, got {data_type:?}")));
        }
        if config.expert_bits != 4 {
            return Err(invalid(format!("paged MoE supports 4-bit experts, got {}", config.expert_bits)));
        }

        let (shared_expert, _) = <dyn Mlp<B>>::new(
            format!("{name}/shared expert"),
            &AnyMLPConfig::DenseMLPConfig(config.shared_expert_config.clone()),
            model_dim,
            config.shared_expert_hidden_dim,
            context,
            &parameter_tree.subtree("shared_expert"),
            data_type,
        )
        .map_err(|error| invalid(format!("shared expert: {error}")))?;

        let router = parameter_tree
            .subtree("router")
            .leaf("weights")?
            .validate(&[config.num_routed_experts + 1, model_dim], data_type)?
            .read_buffer()?;

        let new_matmul = || {
            <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
                .map_err(MoeBlockError::BackendError)
        };

        let gate = MlpGateActMulEncodable::new(
            format!("{name}/expert gate act mul"),
            context,
            data_type,
            config.expert_config.activation.clone(),
            config.expert_config.gate_clipping,
            config.expert_config.up_clipping,
            config.expert_hidden_dim,
            None,
        )
        .map_err(MoeBlockError::BackendError)?;

        let record_size = config.record_size as usize;
        let cache_mb = std::env::var("UZU_EXPERT_CACHE_MB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_EXPERT_CACHE_MB);
        let layers = std::env::var("UZU_PAGED_LAYERS").ok().and_then(|value| value.parse::<u64>().ok()).unwrap_or(40);
        let slot_count = ((cache_mb << 20) / layers / record_size as u64)
            .clamp(config.num_active_routed_experts as u64, config.num_routed_experts as u64)
            as usize;
        let slots = context.create_buffer(slot_count * record_size).map_err(MoeBlockError::BackendError)?;
        let file = File::open(&config.experts_file)
            .map_err(|error| invalid(format!("cannot open {}: {error}", config.experts_file)))?;
        // Optionally bypass the OS file cache so RAM is not spent on a second copy of experts
        // that already sit in the GPU cache.
        #[cfg(target_os = "macos")]
        if std::env::var_os("UZU_PAGED_NOCACHE").is_some() {
            use std::os::fd::AsRawFd;
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        }

        let expert_bytes = |offset: u64, size: usize| offset as usize..offset as usize + size;
        let (d, h) = (model_dim as usize, config.expert_hidden_dim as usize);
        let groups_up = d / config.expert_group_size as usize;
        let groups_down = h / config.expert_group_size as usize;
        let up_parts = [
            expert_bytes(config.up_weights_offset, 2 * h * d / 2),
            expert_bytes(config.up_scales_offset, groups_up * 2 * h * 2),
            expert_bytes(config.up_biases_offset, groups_up * 2 * h * 2),
        ];
        let down_parts = [
            expert_bytes(config.down_weights_offset, d * h / 2),
            expert_bytes(config.down_scales_offset, groups_down * d * 2),
            expert_bytes(config.down_biases_offset, groups_down * d * 2),
        ];

        Ok(Self {
            name,
            shared_expert,
            router,
            router_kernel: Mutex::new(new_matmul()?),
            up_kernel: Mutex::new(new_matmul()?),
            down_kernel: Mutex::new(new_matmul()?),
            gate,
            gather: MoeGather::new(context, data_type).map_err(MoeBlockError::BackendError)?,
            finalize: <B::Kernels as Kernels>::MoeFinalizeKernel::new(context, data_type)
                .map_err(MoeBlockError::BackendError)?,
            cache: Mutex::new(ExpertCache {
                file,
                slots,
                slot_count,
                record_size,
                file_base: config.layer_index as u64 * config.num_routed_experts as u64 * record_size as u64,
                resident: HashMap::new(),
                slot_owner: vec![None; slot_count],
                pinned: vec![false; slot_count],
                tick: 0,
                hits: 0,
                misses: 0,
            }),
            routing: Mutex::new(None),
            model_dim,
            hidden_dim: config.expert_hidden_dim,
            num_experts: config.num_routed_experts,
            top_k: config.num_active_routed_experts,
            normalize_top_k: config.normalize_top_k,
            group_size: config.expert_group_size,
            up_parts,
            down_parts,
            data_type,
        })
    }

    fn routing_for(
        &self,
        context: &B::Context,
        routing: &mut Option<Routing<B>>,
        tokens: u32,
    ) -> Result<(), B::Error> {
        if routing.as_ref().is_some_and(|routing| routing.capacity_tokens >= tokens) {
            return Ok(());
        }
        let capacity_tokens = tokens.next_power_of_two().max(16);
        let k_total = (self.top_k + 1) as usize;
        let rows = capacity_tokens as usize;
        *routing = Some(Routing {
            capacity_tokens,
            logits: context.create_buffer(rows * (self.num_experts as usize + 1) * 2)?,
            bucketed_ids: context.create_buffer(rows * self.top_k as usize * 4)?,
            sumk: context.create_buffer(4)?,
            tok2row: context.create_buffer(rows * k_total * 4)?,
            probs: context.create_buffer(rows * k_total * 2)?,
        });
        Ok(())
    }

    fn quantized<'a>(
        &self,
        slots: &'a B::GlobalBuffer,
        base: usize,
        parts: &[Range<usize>; 3],
        rows: u32,
        columns: u32,
    ) -> MatmulB<impl BufferRef<Backend = B> + 'a> {
        let at = |range: &Range<usize>| slots.subrange(base + range.start..base + range.end);
        MatmulB::Quantized(QuantizedB {
            codes: at(&parts[0]),
            scales: at(&parts[1]),
            correction: QuantizedCorrection::Biases(at(&parts[2])),
            params: QuantParams::new(QuantParamsLayout::GroupOutput, rows, columns / self.group_size),
            mode: QuantizationMode::U4,
            group_size: self.group_size,
            signed_codes: false,
        })
    }
}

/// Softmax over `logits`, top-k, optional renormalization. Returns (expert, probability).
fn route(
    logits: &[bf16],
    top_k: usize,
    normalize: bool,
) -> Vec<(u32, f32)> {
    let max = logits.iter().map(|value| value.to_f32()).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|value| (value.to_f32() - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.select_nth_unstable_by(top_k - 1, |&a, &b| exps[b as usize].total_cmp(&exps[a as usize]));
    let selected = &order[..top_k];
    let selected_sum: f32 = selected.iter().map(|&expert| exps[expert as usize]).sum();
    let denominator = if normalize { selected_sum } else { sum };
    selected.iter().map(|&expert| (expert, exps[expert as usize] / denominator)).collect()
}

impl<B: Backend> Mlp<B> for PagedMoeBlock<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let started = Instant::now();
        command_buffer.push_debug_group(&self.name);
        let (tokens, d, k) = (batch_dim as usize, self.model_dim as usize, self.top_k as usize);
        let k_total = k + 1;
        let row_bytes = d * self.data_type.size_in_bytes();
        let logit_columns = self.num_experts + 1;

        let mut routing_guard = self.routing.lock();
        self.routing_for(command_buffer.context(), &mut routing_guard, batch_dim)?;
        let routing = routing_guard.as_mut().unwrap();

        // 1. Router + shared gate logits.
        self.router_kernel.lock().encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: 0,
                },
                b: MatmulB::FullPrecision {
                    b: &self.router,
                },
                b_leading_dimension: None,
                b_transpose: true,
                output: MatmulOutput::new(&mut routing.logits, MatmulDOps::none()),
                gather_indices: None::<&B::GlobalBuffer>,
                m: batch_dim,
                n: logit_columns,
                k: self.model_dim,
            },
            command_buffer,
        )?;

        // 2. Read them back and route on the CPU.
        let sync_started = Instant::now();
        command_buffer.flush_and_wait()?;
        SYNC_NS.fetch_add(sync_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let logits = (&routing.logits).as_slice::<bf16>();
        let mut per_expert: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.num_experts as usize];
        let mut probs = vec![bf16::ZERO; tokens * k_total];
        let mut tok2row = vec![-1i32; tokens * k_total];
        for token in 0..tokens {
            let row = &logits[token * logit_columns as usize..(token + 1) * logit_columns as usize];
            for (slot, (expert, probability)) in
                route(&row[..self.num_experts as usize], k, self.normalize_top_k).into_iter().enumerate()
            {
                per_expert[expert as usize].push((token as u32, slot as u32));
                probs[token * k_total + slot] = bf16::from_f32(probability);
            }
            let gate_logit = row[self.num_experts as usize].to_f32();
            probs[token * k_total + k] = bf16::from_f32(1.0 / (1.0 + (-gate_logit).exp()));
        }
        let total_rows = tokens * k;
        let mut bucketed_ids = Vec::with_capacity(total_rows);
        let mut expert_rows: Vec<(u32, Range<usize>)> = Vec::new();
        for (expert, assignments) in per_expert.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let start = bucketed_ids.len();
            for &(token, slot) in assignments {
                tok2row[token as usize * k_total + slot as usize] = bucketed_ids.len() as i32;
                bucketed_ids.push(token as i32);
            }
            expert_rows.push((expert as u32, start..bucketed_ids.len()));
        }
        for token in 0..tokens {
            tok2row[token * k_total + k] = (total_rows + token) as i32;
        }
        (&mut routing.bucketed_ids).subrange_mut(..total_rows * 4).copyin(&bucketed_ids);
        (&mut routing.sumk).copyin(&[total_rows as u32]);
        (&mut routing.tok2row).subrange_mut(..tokens * k_total * 4).copyin(&tok2row);
        (&mut routing.probs).subrange_mut(..tokens * k_total * 2).copyin(&probs);

        // 3. Gather rows by expert; shared expert output goes after the routed rows.
        let x_perm = self.gather.encode(
            &input,
            &routing.bucketed_ids,
            &routing.sumk,
            batch_dim,
            self.top_k,
            self.model_dim,
            command_buffer,
        )?;
        let mut y_partial = command_buffer.allocate_scratch(((total_rows + tokens) * row_bytes).max(row_bytes))?;
        let shared = self.shared_expert.encode(input, batch_dim, command_buffer)?;
        command_buffer
            .encode_copy(&shared, (&mut y_partial).subrange_mut(total_rows * row_bytes..(total_rows + tokens) * row_bytes));

        // 4. Routed experts out of the cache, in waves if the cache is too small for the batch.
        let mut cache = self.cache.lock();
        cache.unpin_all();
        let record_size = cache.record_size;
        for (expert, rows) in &expert_rows {
            let io_started = Instant::now();
            let slot = match cache.acquire(*expert).map_err(|error| panic!("expert read failed: {error}"))? {
                Some(slot) => slot,
                None => {
                    command_buffer.flush_and_wait()?;
                    cache.unpin_all();
                    cache.acquire(*expert).map_err(|error| panic!("expert read failed: {error}"))?.unwrap()
                },
            };
            IO_NS.fetch_add(io_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let m = rows.len() as u32;
            let base = slot * record_size;
            let mut fused_up =
                command_buffer.allocate_scratch_for_shape(&[m, 2 * self.hidden_dim], self.data_type)?;
            self.up_kernel.lock().encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &x_perm,
                        offset: rows.start * row_bytes,
                    },
                    b: self.quantized(&cache.slots, base, &self.up_parts, 2 * self.hidden_dim, self.model_dim),
                    b_leading_dimension: None,
                    b_transpose: true,
                    output: MatmulOutput::new(&mut fused_up, MatmulDOps::none()),
                    gather_indices: None::<&B::GlobalBuffer>,
                    m,
                    n: 2 * self.hidden_dim,
                    k: self.model_dim,
                },
                command_buffer,
            )?;
            let LinearInput::FullPrecision(hidden) =
                self.gate.encode_for_linear(command_buffer, &fused_up, m, ActivationFormat::Bf16)?
            else {
                unreachable!("paged MoE gate is full precision");
            };
            self.down_kernel.lock().encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &hidden,
                        offset: 0,
                    },
                    b: self.quantized(&cache.slots, base, &self.down_parts, self.model_dim, self.hidden_dim),
                    b_leading_dimension: None,
                    b_transpose: true,
                    output: MatmulOutput::new(
                        (&mut y_partial).subrange_mut(rows.start * row_bytes..rows.end * row_bytes),
                        MatmulDOps::none(),
                    ),
                    gather_indices: None::<&B::GlobalBuffer>,
                    m,
                    n: self.model_dim,
                    k: self.hidden_dim,
                },
                command_buffer,
            )?;
        }
        if std::env::var_os("UZU_PAGED_STATS").is_some() && cache.file_base == 0 {
            let total = cache.hits + cache.misses;
            let passes = PASSES.swap(0, Ordering::Relaxed).max(1) as f64;
            let ms = |counter: &AtomicU64| counter.swap(0, Ordering::Relaxed) as f64 / 1e6 / passes;
            eprintln!(
                "[paged moe] layer 0 cache: {} slots, hit rate {:.1}% ({} hits / {} misses) | per pass: moe {:.1} ms, sync {:.1} ms, io {:.1} ms",
                cache.slot_count,
                100.0 * cache.hits as f64 / total.max(1) as f64,
                cache.hits,
                cache.misses,
                ms(&TOTAL_NS),
                ms(&SYNC_NS),
                ms(&IO_NS),
            );
        }
        if cache.file_base == 0 {
            PASSES.fetch_add(1, Ordering::Relaxed);
        }
        drop(cache);

        // 5. Combine.
        let mut output = command_buffer.allocate_scratch_for_shape(&[batch_dim, self.model_dim], self.data_type)?;
        self.finalize.encode(
            &routing.tok2row,
            &routing.probs,
            &y_partial,
            &mut output,
            batch_dim,
            self.model_dim,
            self.top_k + 1,
            command_buffer,
        );
        command_buffer.pop_debug_group();
        TOTAL_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(output)
    }
}
