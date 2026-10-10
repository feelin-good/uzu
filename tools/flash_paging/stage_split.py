"""Stage 1 of the flash-paging converter for MLX Qwen3.5/3.6 MoE checkpoints.

Splits an mlx-community `qwen3_5_moe` checkpoint into:

* `dense/`   - a text-only checkpoint that looks like a dense `qwen3_5` model whose
               MLP is the shared expert. lalamo converts this to uzu format as-is.
* `moe_extra.safetensors` - per-layer router and shared-expert-gate weights,
               dequantized to bf16 (they are tiny and the router kernel wants them dense).
* `experts.bin` + `experts.json` - routed experts, still MLX 4-bit, one
               page-aligned record per (layer, expert) so the runtime can pread a single
               expert straight into a GPU cache slot.

Usage: stage_split.py <mlx_checkpoint_dir> <out_dir>
"""

import json
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PAGE = 16384
PREFIX = "language_model."
# Order of the parts inside one expert record. The runtime reads them by offset,
# so this list (and the offsets written to experts.json) is the contract.
EXPERT_PARTS = [
    ("up_proj", "weight"),
    ("up_proj", "scales"),
    ("up_proj", "biases"),
    ("gate_proj", "weight"),
    ("gate_proj", "scales"),
    ("gate_proj", "biases"),
    ("down_proj", "weight"),
    ("down_proj", "scales"),
    ("down_proj", "biases"),
]


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def dequantize(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    per_word = 32 // bits
    w = weight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(per_word, dtype=torch.int64) * bits
    q = ((w.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(*weight.shape[:-1], -1).to(torch.float32)
    q = q.reshape(*q.shape[:-1], -1, group)
    out = q * scales.to(torch.float32).unsqueeze(-1) + biases.to(torch.float32).unsqueeze(-1)
    return out.reshape(*weight.shape[:-1], -1).to(torch.bfloat16)


def main(src: Path, out: Path) -> None:
    config = json.loads((src / "config.json").read_text())
    text = dict(config.get("text_config", config))
    quant = config["quantization"]
    group, bits = quant["group_size"], quant["bits"]
    num_layers, num_experts = text["num_hidden_layers"], text["num_experts"]

    shards = sorted(src.glob("model-*.safetensors"))
    where: dict[str, Path] = {}
    for shard in shards:
        header, _ = read_header(shard)
        for name in header:
            where[name] = shard

    def get(name: str) -> torch.Tensor:
        with safe_open(where[name], framework="pt") as f:
            return f.get_tensor(name)

    dense_dir = out / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dense view: drop vision tower, routed experts, router and shared gate;
    #    rename shared_expert.* to the plain dense-MLP names.
    dense: dict[str, torch.Tensor] = {}
    for name in where:
        if not name.startswith(PREFIX):
            continue
        if ".mlp.switch_mlp." in name or ".mlp.gate." in name or ".mlp.shared_expert_gate." in name:
            continue
        dense[name.replace(".mlp.shared_expert.", ".mlp.")] = get(name)
    save_file(dense, str(dense_dir / "model.safetensors"), metadata={"format": "mlx"})
    del dense

    dense_config = dict(config)
    dense_text = dict(text)
    dense_text["model_type"] = "qwen3_5"
    dense_text["intermediate_size"] = text["shared_expert_intermediate_size"]
    for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size"):
        dense_text.pop(key, None)
    dense_config["text_config"] = dense_text
    dense_config["model_type"] = "qwen3_5"
    dense_config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    dense_config["quantization"] = {"group_size": group, "bits": bits, "mode": quant.get("mode", "affine")}
    dense_config["quantization_config"] = dense_config["quantization"]
    (dense_dir / "config.json").write_text(json.dumps(dense_config, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        if (src / name).exists():
            shutil.copy(src / name, dense_dir / name)

    # 2. Router + shared-expert gate, dequantized to bf16.
    extra: dict[str, torch.Tensor] = {}
    for layer in range(num_layers):
        base = f"{PREFIX}model.layers.{layer}.mlp."
        for part, key in (("gate", "router"), ("shared_expert_gate", "shared_expert_gate")):
            spec = quant.get(f"{base}{part}", {"group_size": group, "bits": bits})
            extra[f"layers.{layer}.{key}"] = dequantize(
                get(f"{base}{part}.weight"),
                get(f"{base}{part}.scales"),
                get(f"{base}{part}.biases"),
                spec["bits"],
                spec["group_size"],
            )
    save_file(extra, str(out / "moe_extra.safetensors"))

    # 3. Routed experts: one page-aligned record per (layer, expert).
    layout = None
    record_size = None
    with (out / "experts.bin").open("wb") as f:
        for layer in range(num_layers):
            base = f"{PREFIX}model.layers.{layer}.mlp.switch_mlp."
            tensors = {(p, k): get(f"{base}{p}.{k}") for p, k in EXPERT_PARTS}
            if layout is None:
                layout, offset = [], 0
                for p, k in EXPERT_PARTS:
                    t = tensors[(p, k)][0]
                    size = t.numel() * t.element_size()
                    layout.append({"name": f"{p}.{k}", "offset": offset, "size": size,
                                   "dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape)})
                    offset = (offset + size + 255) // 256 * 256
                record_size = (offset + PAGE - 1) // PAGE * PAGE
            for expert in range(num_experts):
                record = bytearray(record_size)
                for entry, (p, k) in zip(layout, EXPERT_PARTS):
                    raw = tensors[(p, k)][expert].contiguous().view(torch.uint8).numpy().tobytes()
                    assert len(raw) == entry["size"]
                    record[entry["offset"]:entry["offset"] + len(raw)] = raw
                f.write(record)
            print(f"layer {layer}: {num_experts} experts written", flush=True)

    (out / "experts.json").write_text(json.dumps({
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_active_experts": text["num_experts_per_tok"],
        "model_dim": text["hidden_size"],
        "expert_hidden_dim": text["moe_intermediate_size"],
        "bits": bits,
        "group_size": group,
        "record_size": record_size,
        "parts": layout,
    }, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
