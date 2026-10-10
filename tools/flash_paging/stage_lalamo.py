"""Stage 2 of the flash-paging converter: run lalamo on the dense view from stage 1.

Run with lalamo's environment:  <lalamo>/.venv/bin/python stage_lalamo.py <stage1_out_dir>
Writes <stage1_out_dir>/uzu (config.json, model.safetensors, tokenizer.json).
"""

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lalamo.commands import convert
from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.qwen import QWEN_END_OF_THINKING_TAG
from lalamo.model_import.model_specs.reasoning_configs import BOOLEAN_REASONING_DEFAULT_ON_CONFIG
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.origins import Origin, WeightFormat, load_weight_paths


@dataclass(frozen=True)
class LocalOrigin(Origin):
    path: str

    def resolve_file(self, file_spec: FileSpec, progress_callback: Callable | None = None) -> Path:
        return Path(self.path) / file_spec.filename

    def get_weights(self, progress_callback: Callable | None = None) -> Sequence:
        return load_weight_paths(sorted(Path(self.path).glob("*.safetensors")), WeightFormat.SAFETENSORS)

    @property
    def description(self) -> str:
        return self.path


def main(stage_dir: Path) -> None:
    dense = stage_dir / "dense"
    spec = LanguageModelSpec(
        vendor="Alibaba",
        family="Qwen3.6",
        name="Qwen3.6-35B-A3B-FlashPaged",
        size="35B",
        origin=LocalOrigin(str(dense)),
        config_type=HFQwen35Config,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=QWEN_END_OF_THINKING_TAG,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    )
    convert(spec, stage_dir / "uzu")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
