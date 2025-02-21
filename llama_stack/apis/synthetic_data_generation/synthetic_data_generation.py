# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type, webmethod


class FilteringFunction(Enum):
    """The type of filtering function."""

    none = "none"
    random = "random"
    top_k = "top_k"
    top_p = "top_p"
    top_k_top_p = "top_k_top_p"
    sigmoid = "sigmoid"


@json_schema_type
class SyntheticDataGenerationRequest(BaseModel):
    """Request to generate synthetic data"""

    input_data: List[Dict[str, Any]]
    config: Dict[str, Any]  # Provider-specific configuration
    model: Optional[str] = None


@json_schema_type
class SyntheticDataGenerationResponse(BaseModel):
    """Response containing generated data and execution statistics"""

    synthetic_data: List[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]] = None


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic-data-generation/generate")
    def synthetic_data_generate(
        self,
        input_data: List[Dict[str, Any]],
        config: Dict[str, Any],
        model: Optional[str] = None,
    ) -> SyntheticDataGenerationResponse: ...

    @webmethod(route="/synthetic-data-generation/capabilities")
    def get_capabilities(self) -> Dict[str, Any]: ...
