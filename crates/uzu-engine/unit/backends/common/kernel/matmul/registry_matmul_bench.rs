#![cfg(backend = "metal")]

use criterion::{Criterion, Throughput};
use half::bf16;
use serde_json::json;
use uzu_engine_macros::uzu_bench;
use xxhash_rust::xxh3::xxh3_64;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context,
            kernel::{
                ActivationQuantization, ActivationTransform,
                matmul::{
                    ActivationFormat, MatmulA, MatmulArguments, MatmulDOps, MatmulKernel, MatmulOutput, MatmulShape,
                    QuantParamsLayout,
                },
            },
        },
        metal::{GemmEngine, Metal, MetalContext},
    },
    data_type::DataType,
    tests::{
        cold_pool::ColdPool,
        env_vars,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        matmul::{
            MetalMatmul, QuantBuffers, QuantInput, REGISTRY_MAX_BATCH, RegistryCase, Shape, harness::TestDispatch,
            iter_encode_loop_named, quant_arguments, registry_layer_shapes,
        },
        util::{shared_metal_context, type_short_name},
    },
};

const ROUTES: [(&str, TestDispatch, Option<ActivationFormat>); 4] = [
    ("natural", None, None),
    ("simdgroup", Some(GemmEngine::Simdgroup), Some(ActivationFormat::Bf16)),
    ("mxu", Some(GemmEngine::Mxu), Some(ActivationFormat::Bf16)),
    ("mxu", Some(GemmEngine::Mxu), Some(ActivationFormat::Int8)),
];
const TIMINGS: [(&str, bool); 2] = [("prepared", false), ("with_input_preparation", true)];
const RESIDENCIES: [(&str, bool, &str); 2] = [
    ("streaming", true, "ColdPool rotation over a 256 MiB working set"),
    ("reused", false, "same-address reuse, not guaranteed cache-resident"),
];

fn rht_signs(
    context: &MetalContext,
    len: u32,
) -> <Metal as Backend>::GlobalBuffer {
    let signs: Vec<i32> = (0..len)
        .map(|index| {
            if index % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();
    create_buffer_with_data::<Metal, i32>(context, &signs)
}

fn natural_activation(
    matmul: &MetalMatmul,
    buffers: &QuantBuffers<Metal, bf16>,
    input: &QuantInput<bf16>,
    output: &mut <Metal as Backend>::GlobalBuffer,
    output_signs: Option<&<Metal as Backend>::GlobalBuffer>,
    m: u32,
    context: &MetalContext,
) -> ActivationFormat {
    let arguments = MatmulArguments {
        a: MatmulA::FullPrecision {
            values: &buffers.x,
            offset: 0,
        },
        b: buffers.matmul_b(input),
        b_leading_dimension: None,
        b_transpose: true,
        output: MatmulOutput::new(
            output,
            MatmulDOps {
                rht_factors: output_signs,
                ..MatmulDOps::none()
            },
        ),
        gather_indices: None::<&<Metal as Backend>::GlobalBuffer>,
        m,
        n: input.n,
        k: input.k,
    };
    matmul.select_activation_format(&MatmulShape::from_arguments(&arguments), context)
}

fn encode_cell(
    matmul: &mut MetalMatmul,
    buffers: &QuantBuffers<Metal, bf16>,
    input: &QuantInput<bf16>,
    quantization: Option<ActivationQuantization>,
    rht: &ActivationTransform<Metal>,
    quantizer: Option<&ActivationTransform<Metal>>,
    input_signs: &<Metal as Backend>::GlobalBuffer,
    output_signs: Option<&<Metal as Backend>::GlobalBuffer>,
    input_rht: bool,
    scratch: &mut [<Metal as Backend>::GlobalBuffer; 4],
    output: &mut <Metal as Backend>::GlobalBuffer,
    m: u32,
    route: TestDispatch,
    activation: ActivationFormat,
    with_preparation: bool,
    plan: Option<&mut String>,
    command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
) -> Result<(), String> {
    let [rotated, a_values, a_scales, a_sums] = scratch;
    let k = input.k;
    let a = match activation {
        ActivationFormat::Bf16 if with_preparation => {
            rht.encode_fp(&buffers.x, &mut *rotated, input_signs, m, k, command_buffer);
            MatmulA::FullPrecision {
                values: &*rotated,
                offset: 0,
            }
        },
        ActivationFormat::Bf16 => MatmulA::FullPrecision {
            values: if input_rht {
                &*rotated
            } else {
                &buffers.x
            },
            offset: 0,
        },
        ActivationFormat::Int8 => {
            let (Some(quantization), Some(quantizer)) = (quantization, quantizer) else {
                return Err("production activation policy prepares no int8 input for this device and shape".into());
            };
            if with_preparation {
                quantizer.encode_quantize(
                    &buffers.x,
                    &mut *a_values,
                    &mut *a_scales,
                    quantization.sum_group_size().map(|_| &mut *a_sums),
                    input_signs,
                    m,
                    k,
                    command_buffer,
                );
            }
            MatmulA::Int8Symmetric {
                values: &*a_values,
                scales: &*a_scales,
                group_sums: quantization.sum_group_size().map(|_| &*a_sums),
                scale_group_size: quantization.scale_group_size(),
                code_layout: quantization.code_layout(),
            }
        },
    };
    let arguments = MatmulArguments {
        a,
        b: buffers.matmul_b(input),
        b_leading_dimension: None,
        b_transpose: true,
        output: MatmulOutput::new(
            output,
            MatmulDOps {
                rht_factors: output_signs,
                ..MatmulDOps::none()
            },
        ),
        gather_indices: None::<&<Metal as Backend>::GlobalBuffer>,
        m,
        n: input.n,
        k,
    };
    if let Some(plan) = plan {
        let shape = MatmulShape::from_arguments(&arguments);
        let context = command_buffer.context();
        *plan = match route {
            None => format!(
                "{:?}",
                MetalMatmul::choose_dispatch(
                    &shape,
                    &context.device_name,
                    context.gpu_core_count,
                    context.apple_gpu_family,
                    context.supports_mxu,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                )
            ),
            Some(engine) => format!("{:?}", matmul.plan_for_gemm_engine(&shape, engine, context)),
        };
    }
    match route {
        Some(engine) => matmul.encode_with_gemm_engine(arguments, engine, command_buffer),
        None => matmul.encode(arguments, command_buffer),
    }
    .map_err(|error| format!("{error}"))
}

fn bench_case(
    c: &mut Criterion,
    context: &MetalContext,
    case: &RegistryCase,
    shapes: impl Iterator<Item = Shape>,
) {
    let &(label, method, bits, group_size, params_layout, input_rht, output_rht, n, k) = case;
    let seed = (u64::from(n) << 32) ^ (u64::from(k) << 8) ^ u64::from(bits) ^ u64::from(group_size);
    let with_layout = |input: QuantInput<bf16>| {
        if params_layout == QuantParamsLayout::GroupOutput {
            input.with_group_output()
        } else {
            input
        }
    };
    let mut matmul = MetalMatmul::new(context, DataType::BF16, DataType::BF16, DataType::BF16).expect("matmul kernel");
    let input_signs = rht_signs(context, k);
    let output_signs = output_rht.then(|| rht_signs(context, n));

    let probe = with_layout(QuantInput::<bf16>::new(1, k, 32, group_size, bits, method, seed));
    let mut probe_buffers = QuantBuffers::<Metal, bf16>::allocate(context, &probe);
    let mut probe_arguments = quant_arguments(&mut probe_buffers, &probe);
    probe_arguments.output.ops.rht_factors = output_signs.as_ref();
    let mut a8_candidate = MatmulShape::from_arguments(&probe_arguments);
    a8_candidate.n = n;
    a8_candidate.a_full_precision = false;
    a8_candidate.signed_codes = bits != 4;
    let quantization = matmul.select_activation_quantization(&a8_candidate, context);

    let input = with_layout(QuantInput::<bf16>::new(REGISTRY_MAX_BATCH, k, n, group_size, bits, method, seed));
    let input = if quantization.is_some() && bits != 4 {
        input.with_signed_weight_codes()
    } else {
        input
    };
    let rht = ActivationTransform::<Metal>::input_rht(context, DataType::BF16, false).expect("input rht");
    let quantizer = quantization.map(|quantization| {
        ActivationTransform::<Metal>::quantize(context, DataType::BF16, quantization).expect("quantizer")
    });
    let rows_by_k = (REGISTRY_MAX_BATCH * k) as usize;
    let mut scratch = [
        create_buffer::<Metal, bf16>(context, rows_by_k),
        create_buffer::<Metal, i8>(context, rows_by_k),
        create_buffer::<Metal, f32>(
            context,
            quantization.map_or(1, |quantization| rows_by_k / quantization.scale_group_size() as usize),
        ),
        create_buffer::<Metal, i32>(
            context,
            quantization
                .and_then(ActivationQuantization::sum_group_size)
                .map_or(1, |sum_group| rows_by_k / sum_group as usize),
        ),
    ];
    let output_elements = (REGISTRY_MAX_BATCH * n) as usize;
    let mut output = create_buffer::<Metal, bf16>(context, output_elements);
    let weight_bytes = input.weight_buffer_bytes();
    let reused = QuantBuffers::<Metal, bf16>::allocate(context, &input);
    let mut streaming = ColdPool::new(weight_bytes, || QuantBuffers::<Metal, bf16>::allocate(context, &input));
    {
        let [rotated, a_values, a_scales, a_sums] = &mut scratch;
        let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
        rht.encode_fp(&reused.x, &mut *rotated, &input_signs, REGISTRY_MAX_BATCH, k, &mut command_buffer);
        if let (Some(quantization), Some(quantizer)) = (quantization, quantizer.as_ref()) {
            quantizer.encode_quantize(
                &reused.x,
                &mut *a_values,
                &mut *a_scales,
                quantization.sum_group_size().map(|_| &mut *a_sums),
                &input_signs,
                REGISTRY_MAX_BATCH,
                k,
                &mut command_buffer,
            );
        }
        command_buffer.end_encoding().submit().wait_until_completed().expect("input preparation");
    }

    let group_name = format!("{}/Kernel/RegistryMatmul/{label}", type_short_name::<Metal>());
    let mut group = c.benchmark_group(&group_name);
    for Shape {
        m,
        ..
    } in shapes
    {
        let natural = natural_activation(&matmul, &reused, &input, &mut output, output_signs.as_ref(), m, context);
        let output_prefix = (m * n) as usize;
        for (route_label, route, forced_activation) in ROUTES {
            let activation = forced_activation.unwrap_or(natural);
            let activation_label = match activation {
                ActivationFormat::Bf16 => "bf16",
                ActivationFormat::Int8 => "a8",
            };
            let mut trials = Vec::new();
            for (timing, with_preparation) in TIMINGS {
                let mut plan = String::new();
                let trial = if with_preparation && !input_rht {
                    Err("input rotation is fused into the preceding norm for this role".to_string())
                } else if route == Some(GemmEngine::Mxu) && !context.supports_mxu {
                    Err("device has no MXU".to_string())
                } else {
                    (&mut output).copyin(&vec![bf16::NAN; output_elements]);
                    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
                    let result = encode_cell(
                        &mut matmul,
                        &reused,
                        &input,
                        quantization,
                        &rht,
                        quantizer.as_ref(),
                        &input_signs,
                        output_signs.as_ref(),
                        input_rht,
                        &mut scratch,
                        &mut output,
                        m,
                        route,
                        activation,
                        with_preparation,
                        Some(&mut plan),
                        &mut command_buffer,
                    );
                    command_buffer.end_encoding().submit().wait_until_completed().expect("trial command buffer");
                    result.map(|()| {
                        let mut bytes = buffer_to_vec::<Metal, u8>(&output);
                        bytes.truncate(output_prefix * size_of::<bf16>());
                        bytes
                    })
                };
                trials.push((timing, with_preparation, plan, trial));
            }
            let paired = match (&trials[0].3, &trials[1].3) {
                (Ok(prepared), Ok(inclusive)) => Some(prepared == inclusive),
                _ => None,
            };
            for (timing, with_preparation, plan, trial) in trials {
                let trial = match (trial, paired) {
                    (Ok(_), Some(false)) => {
                        Err("prepared and preparation-inclusive outputs differ: invalid input contract".to_string())
                    },
                    (trial, _) => trial,
                };
                let output_hash = trial.as_ref().ok().map(|bytes| format!("{:016x}", xxh3_64(bytes)));
                let output_bytes = trial.as_ref().map_or(0, Vec::len);
                for (residency, rotate, residency_contract) in RESIDENCIES {
                    let function_id = format!("{route_label}_{activation_label}/{timing}/{residency}/m{m}");
                    let benchmark_path = format!("{group_name}/{function_id}");
                    println!(
                        "{}",
                        json!({
                            "record": "matmul_cell",
                            "benchmark_path": benchmark_path,
                            "status": if trial.is_ok() { "ok" } else { "n/a" },
                            "reason": trial.as_ref().err(),
                            "device": context.device_name,
                            "apple_gpu_family": format!("{:?}", context.apple_gpu_family),
                            "gpu_core_count": context.gpu_core_count,
                            "supports_mxu": context.supports_mxu,
                            "case": label,
                            "method": format!("{method:?}"),
                            "bits": bits,
                            "group_size": group_size,
                            "params_layout": format!("{params_layout:?}"),
                            "signed_codes": input.signed_codes,
                            "input_rht": input_rht,
                            "output_rht": output_rht,
                            "data_type": "BF16",
                            "m": m,
                            "n": n,
                            "k": k,
                            "seed": seed,
                            "route": route_label,
                            "natural_activation": format!("{natural:?}"),
                            "activation": format!("{activation:?}"),
                            "plan": plan,
                            "timing": timing,
                            "output_prefix_bytes": output_bytes,
                            "output_xxh3": output_hash,
                            "prepared_matches_inclusive": paired,
                            "residency": residency,
                            "residency_contract": residency_contract,
                            "weight_bytes": weight_bytes,
                            "measurement": "gpu_ns is one command buffer holding `iterations` encodes: amortized throughput, not dispatch latency",
                        })
                    );
                    if trial.is_err() {
                        continue;
                    }
                    group.throughput(Throughput::Elements(2 * u64::from(m) * u64::from(n) * u64::from(k)));
                    group.bench_function(&function_id, |bencher| {
                        iter_encode_loop_named::<Metal, _>(context, bencher, &benchmark_path, |command_buffer| {
                            let buffers = if rotate {
                                &*streaming.next_mut()
                            } else {
                                &reused
                            };
                            encode_cell(
                                &mut matmul,
                                buffers,
                                &input,
                                quantization,
                                &rht,
                                quantizer.as_ref(),
                                &input_signs,
                                output_signs.as_ref(),
                                input_rht,
                                &mut scratch,
                                &mut output,
                                m,
                                route,
                                activation,
                                with_preparation,
                                None,
                                command_buffer,
                            )
                            .expect("cell was encoded once before timing");
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

#[uzu_bench]
fn bench_registry_matmul(c: &mut Criterion) {
    let filter = std::env::var(env_vars::UZU_MATMUL_CASE_FILTER).unwrap_or_default();
    let context = shared_metal_context();
    let mut selected = 0;
    for (case, shapes) in registry_layer_shapes().filter(|(case, _)| case.0.contains(filter.as_str())) {
        selected += 1;
        bench_case(c, &context, case, shapes);
    }
    assert!(selected > 0, "{}={filter:?} matches no registry matmul case", env_vars::UZU_MATMUL_CASE_FILTER);
}
