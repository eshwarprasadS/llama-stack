# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import InstructlabSDGConfig


async def get_provider_impl(config: InstructlabSDGConfig, _deps) -> Any:
    from .instructlab_sdg import InstructlabSDGImpl

    impl = InstructlabSDGImpl(config)
    await impl.initialize()
    return impl
