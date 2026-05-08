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
from typing import Any, Dict, Optional, Sequence

import yaml

from .llm_utils import _deep_merge

__all__ = ["DEFAULT_MODEL_CONFIGS_DIR", "load_model_defaults", "merge_model_defaults"]

DEFAULT_MODEL_CONFIGS_DIR: Path = Path(__file__).resolve().parent.parent / "model_configs"


def _load_yaml_if_exists(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Model config {path} must be a YAML mapping at the top level, "
            f"got {type(loaded).__name__}."
        )
    return loaded


def load_model_defaults(
    arch: str,
    *,
    include_ad_default: bool = False,
    search_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """Return merged per-model defaults for ``arch``.

    Loads ``<arch>.yaml`` and, when ``include_ad_default``, ``<arch>_ad_default.yaml``.
    Right-side precedence: model.yaml beats ad_default.yaml; later
    ``search_dirs`` override earlier ones. Missing files are skipped.
    """
    if not arch:
        return {}

    if search_dirs is None:
        search_dirs = (DEFAULT_MODEL_CONFIGS_DIR,)

    merged: Dict[str, Any] = {}
    for directory in search_dirs:
        directory = Path(directory)
        if include_ad_default:
            ad_layer = _load_yaml_if_exists(directory / f"{arch}_ad_default.yaml")
            if ad_layer:
                merged = _deep_merge(merged, ad_layer)

        model_layer = _load_yaml_if_exists(directory / f"{arch}.yaml")
        if model_layer:
            merged = _deep_merge(merged, model_layer)

    return merged


def merge_model_defaults(*layers: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge defaults layers with right-side precedence; empty layers are skipped."""
    result: Dict[str, Any] = {}
    for layer in layers:
        if layer:
            result = _deep_merge(result, layer)
    return result
