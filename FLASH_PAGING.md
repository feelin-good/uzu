# Flash paging fork

This fork of [trymirai/uzu](https://github.com/trymirai/uzu) runs MoE models that are larger than the Mac's RAM on the GPU only. Routed experts stay in a file on disk (an external SSD in our setup) and are paged into a GPU-resident LRU cache on demand. Everything else stays resident.

Target: **Qwen3.6-35B-A3B (MLX 4-bit) on an M4 with 16 GB.** At 4-bit that model is about 19.5 GB: 17.4 GB of routed experts plus about 2 GB of everything else.

## Branch layout

| Branch | What it is |
|---|---|
| `main` | Tracks upstream `trymirai/uzu` `main`, untouched |
| `flash-paging` | Our work, rebased or merged onto upstream as it moves |

To pick up upstream changes: `git fetch upstream && git merge upstream/main` on `flash-paging`.

## How it works

```
token ──► attention / DeltaNet (resident)
      ──► router (resident, bf16) ──► top-8 expert ids ──┐
                                        GPU→CPU sync ◄───┘ (one per MoE layer)
      ──► expert cache: hits use their slot, misses are pread from experts.bin into an LRU slot
      ──► 8 × [MLX 4-bit gate/up matmul → SiLU·mul → down matmul] on the cache slots, weighted sum
      ──► + sigmoid(shared_gate · x) × shared expert (resident)
```

- **Converter** (`tools/flash_paging/`):
  1. `stage_split.py` splits an mlx-community checkpoint three ways:
     - A dense view where the MLP is the shared expert.
     - The router and shared-expert gate, dequantized to bf16.
     - `experts.bin`, with one page-aligned 4-bit record per (layer, expert).
  2. `stage_lalamo.py` runs Mirai's lalamo on the dense view, giving a normal uzu model.
  3. Finalize (to do): rewrite each MLP config to `PagedMixtureOfExpertsConfig` and attach the router, gate and expert file.
- **Runtime** (to do): `PagedMoeBlock`, plus a `flush_and_wait` on the command-buffer encoding so the forward pass can stop at each router and page in experts.

## Changelog

| Date | Change |
|---|---|
| 2026-10-09 | Restored MoE kernels (revert of upstream #886 "Burn MoE"); 44 MoE unit tests pass on M4 |
| 2026-10-10 | Converter stages 1–2. The dense skeleton (1.37 GB) loads and generates in uzu. Its output is meaningless until the routed experts are wired in. |

## Baselines on the same Mac (M4, 16 GB)

| Setup | Prompt | Generation |
|---|---|---|
| llama.cpp, IQ2_M GGUF fully on GPU | 365 tok/s | 30.9 tok/s |
| llama.cpp, IQ3_S GGUF, CPU + mmap from SSD | 67 tok/s | 19.7 tok/s |
| uzu flash paging, MLX 4-bit | to measure | to measure |
