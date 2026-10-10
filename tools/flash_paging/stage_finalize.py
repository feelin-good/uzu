"""Stage 3 of the flash-paging converter: assemble the final uzu model directory.

Run with lalamo's environment:  <lalamo>/.venv/bin/python stage_finalize.py <stage_dir>
Reads <stage_dir>/{uzu, moe_extra.safetensors, experts.json, experts.bin} and writes
<stage_dir>/model, which the flash-paging fork of uzu loads with `--model <stage_dir>/model`.
experts.bin stays where it is; config.json references it by absolute path.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYER_PREFIX = "decoder.transformer.layers."


def main(stage_dir: Path) -> None:
    src = stage_dir / "uzu"
    out = stage_dir / "model"
    out.mkdir(exist_ok=True)
    experts = json.loads((stage_dir / "experts.json").read_text())
    offsets = {part["name"]: part["offset"] for part in experts["parts"]}

    config = json.loads((src / "config.json").read_text())
    layers = config["decoder_config"]["transformer_config"]["layer_configs"]
    assert len(layers) == experts["num_layers"]
    for index, layer in enumerate(layers):
        dense = layer["mlp_config"]
        assert dense["type"] == "DenseMLPConfig", dense["type"]
        layer["mlp_config"] = {
            "type": "PagedMixtureOfExpertsConfig",
            "shared_expert_config": dense,
            "shared_expert_hidden_dim": experts["expert_hidden_dim"],
            "expert_config": dense,
            "expert_hidden_dim": experts["expert_hidden_dim"],
            "expert_bits": experts["bits"],
            "expert_group_size": experts["group_size"],
            "num_routed_experts": experts["num_experts"],
            "num_active_routed_experts": experts["num_active_experts"],
            "normalize_top_k": True,
            "experts_file": str((stage_dir / "experts.bin").resolve()),
            "layer_index": index,
            "record_size": experts["record_size"],
            "up_weights_offset": offsets["up_projection.weights"],
            "up_scales_offset": offsets["up_projection.scales"],
            "up_biases_offset": offsets["up_projection.biases"],
            "down_weights_offset": offsets["down_projection.weights"],
            "down_scales_offset": offsets["down_projection.scales"],
            "down_biases_offset": offsets["down_projection.biases"],
        }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    def rename(name: str) -> str:
        if name.startswith(LAYER_PREFIX) and ".mlp." in name:
            return name.replace(".mlp.", ".mlp.shared_expert.", 1)
        return name

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src / "model.safetensors", framework="pt") as f:
        metadata = {rename(key): value for key, value in (f.metadata() or {}).items()}
        for name in f.keys():
            tensors[rename(name)] = f.get_tensor(name)
    with safe_open(stage_dir / "moe_extra.safetensors", framework="pt") as f:
        for index in range(experts["num_layers"]):
            router = f.get_tensor(f"layers.{index}.router")
            shared_gate = f.get_tensor(f"layers.{index}.shared_expert_gate")
            tensors[f"{LAYER_PREFIX}{index}.mlp.router.weights"] = torch.cat([router, shared_gate]).contiguous()
    save_file(tensors, str(out / "model.safetensors"), metadata=metadata)

    for name in ("tokenizer.json", "encoding.json"):
        shutil.copy(src / name, out / name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
