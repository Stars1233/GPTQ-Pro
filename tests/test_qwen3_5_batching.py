# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest


pytest.importorskip("transformers.models.qwen3_5")

from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel


def test_qwen3_5_disables_batch_quantization():
    assert Qwen3_5QModel.support_batch_quantize is False
