# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.utils.perplexity import Perplexity


class _DummyTokenizer:
    def __init__(self, input_ids, pad_token_id: int = 0):
        self._input_ids = torch.tensor([input_ids], dtype=torch.long)
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id
        self.bos_token_id = None
        self.model_max_length = 0

    def __call__(self, _text, truncation=False, return_tensors="pt"):
        return SimpleNamespace(input_ids=self._input_ids.clone())


class _DummyModel(nn.Module):
    def __init__(self, vocab_size: int, mode: str):
        super().__init__()
        self.vocab_size = vocab_size
        self.mode = mode
        self.register_parameter("_device_anchor", nn.Parameter(torch.zeros(1)))

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        logits = torch.zeros((*input_ids.shape, self.vocab_size), dtype=torch.float32, device=input_ids.device)
        if self.mode == "uniform":
            return SimpleNamespace(logits=logits)

        if self.mode != "perfect":
            raise ValueError(f"Unknown dummy mode: {self.mode}")

        next_tokens = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, next_tokens.unsqueeze(-1), 12.0)
        return SimpleNamespace(logits=logits)


def _make_perplexity(monkeypatch, *, input_ids, vocab_size: int, mode: str) -> Perplexity:
    monkeypatch.setattr(Perplexity, "_prepare_data", lambda self: "stub")
    tokenizer = _DummyTokenizer(input_ids=input_ids)
    model = _DummyModel(vocab_size=vocab_size, mode=mode)
    return Perplexity(model=model, tokenizer=tokenizer, dataset_path="unused")


def test_perplexity_matches_uniform_distribution_reference(monkeypatch):
    ppl = _make_perplexity(
        monkeypatch,
        input_ids=[0, 1, 2, 3, 4, 5, 0, 1, 2],
        vocab_size=6,
        mode="uniform",
    )

    scores = ppl.calculate(n_ctx=4, n_batch=8)

    assert scores
    assert scores[-1] == pytest.approx(6.0, rel=0.0, abs=1e-6)


def test_perplexity_is_invariant_to_smaller_batch_token_budget(monkeypatch):
    ppl = _make_perplexity(
        monkeypatch,
        input_ids=[0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
        vocab_size=6,
        mode="uniform",
    )

    small_budget_scores = ppl.calculate(n_ctx=4, n_batch=2)
    large_budget_scores = ppl.calculate(n_ctx=4, n_batch=16)

    assert small_budget_scores
    assert large_budget_scores
    assert small_budget_scores[-1] == pytest.approx(large_budget_scores[-1], rel=0.0, abs=1e-6)


def test_perplexity_includes_final_partial_window(monkeypatch):
    ppl = _make_perplexity(
        monkeypatch,
        input_ids=[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4],
        vocab_size=6,
        mode="perfect",
    )

    scores = ppl.calculate(n_ctx=4, n_batch=4)

    assert len(scores) == 3
    assert scores[-1] < 1.001
