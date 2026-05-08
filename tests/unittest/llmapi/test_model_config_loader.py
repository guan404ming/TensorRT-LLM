# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import yaml

from tensorrt_llm.llmapi.model_config_loader import (
    DEFAULT_MODEL_CONFIGS_DIR,
    load_model_defaults,
    merge_model_defaults,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))


def test_returns_empty_when_no_files(tmp_path):
    assert load_model_defaults("LlamaForCausalLM", search_dirs=[tmp_path]) == {}


def test_returns_empty_when_arch_is_empty(tmp_path):
    _write_yaml(tmp_path / "LlamaForCausalLM.yaml", {"max_seq_len": 4096})
    assert load_model_defaults("", search_dirs=[tmp_path]) == {}


def test_loads_model_yaml_only(tmp_path):
    _write_yaml(tmp_path / "LlamaForCausalLM.yaml", {"max_seq_len": 4096})
    assert load_model_defaults("LlamaForCausalLM", search_dirs=[tmp_path]) == {"max_seq_len": 4096}


def test_ad_default_skipped_unless_requested(tmp_path):
    _write_yaml(tmp_path / "LlamaForCausalLM_ad_default.yaml", {"ad_only": True})
    _write_yaml(tmp_path / "LlamaForCausalLM.yaml", {"max_seq_len": 4096})

    pytorch_result = load_model_defaults("LlamaForCausalLM", search_dirs=[tmp_path])
    assert pytorch_result == {"max_seq_len": 4096}

    autodeploy_result = load_model_defaults(
        "LlamaForCausalLM",
        include_ad_default=True,
        search_dirs=[tmp_path],
    )
    assert autodeploy_result == {"max_seq_len": 4096, "ad_only": True}


def test_model_yaml_overrides_ad_default(tmp_path):
    _write_yaml(
        tmp_path / "LlamaForCausalLM_ad_default.yaml",
        {"max_seq_len": 1024, "ad_only": True},
    )
    _write_yaml(tmp_path / "LlamaForCausalLM.yaml", {"max_seq_len": 4096})

    result = load_model_defaults(
        "LlamaForCausalLM",
        include_ad_default=True,
        search_dirs=[tmp_path],
    )
    assert result == {"max_seq_len": 4096, "ad_only": True}


def test_deep_merge_for_nested_dicts(tmp_path):
    _write_yaml(
        tmp_path / "LlamaForCausalLM_ad_default.yaml",
        {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.8,
            }
        },
    )
    _write_yaml(
        tmp_path / "LlamaForCausalLM.yaml",
        {"kv_cache_config": {"free_gpu_memory_fraction": 0.9}},
    )

    result = load_model_defaults(
        "LlamaForCausalLM",
        include_ad_default=True,
        search_dirs=[tmp_path],
    )
    assert result == {
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.9,
        }
    }


def test_later_search_dir_overrides_earlier(tmp_path):
    base = tmp_path / "base"
    overlay = tmp_path / "overlay"
    _write_yaml(base / "LlamaForCausalLM.yaml", {"max_seq_len": 4096, "untouched": True})
    _write_yaml(overlay / "LlamaForCausalLM.yaml", {"max_seq_len": 8192})

    result = load_model_defaults("LlamaForCausalLM", search_dirs=[base, overlay])
    assert result == {"max_seq_len": 8192, "untouched": True}


def test_default_search_dir_resolves_to_package_dir():
    assert DEFAULT_MODEL_CONFIGS_DIR.name == "model_configs"
    assert DEFAULT_MODEL_CONFIGS_DIR.parent.name == "tensorrt_llm"


def test_empty_yaml_treated_as_empty_dict(tmp_path):
    (tmp_path / "LlamaForCausalLM.yaml").write_text("")
    assert load_model_defaults("LlamaForCausalLM", search_dirs=[tmp_path]) == {}


def test_non_mapping_yaml_raises(tmp_path):
    (tmp_path / "LlamaForCausalLM.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_model_defaults("LlamaForCausalLM", search_dirs=[tmp_path])


def test_merge_model_defaults_right_wins_and_skips_empty():
    yaml_layer = {
        "max_seq_len": 4096,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.8,
        },
    }
    code_layer = {"kv_cache_config": {"free_gpu_memory_fraction": 0.9}}

    assert merge_model_defaults() == {}
    assert merge_model_defaults({}, None, {}) == {}
    assert merge_model_defaults(yaml_layer, {}) == yaml_layer
    assert merge_model_defaults({}, code_layer) == code_layer
    assert merge_model_defaults(yaml_layer, code_layer) == {
        "max_seq_len": 4096,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.9,
        },
    }


def test_missing_search_dir_silently_skipped(tmp_path):
    real = tmp_path / "real"
    missing = tmp_path / "does_not_exist"
    _write_yaml(real / "LlamaForCausalLM.yaml", {"max_seq_len": 4096})

    assert load_model_defaults("LlamaForCausalLM", search_dirs=[missing, real]) == {
        "max_seq_len": 4096
    }
