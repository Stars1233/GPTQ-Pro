# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
import sys
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from logbar import LogBar


logger = LogBar.shared()


class Perplexity:
    """
    A helper for calculating next-token perplexity over a text corpus.
    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset_path: str = "wikitext",
        dataset_name: str | None = None,
        split: str = "test",
        text_column: str = "text",
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()

    def _prepare_data(self) -> str:
        """
        Load the requested dataset and concatenate a bounded number of samples.
        """
        if self._dataset_path == "wikitext":
            self._dataset_name = "wikitext-2-raw-v1"

        length = 512 if self._dataset_path == "wikitext" else 2048
        if self._dataset_path.startswith("/") or self._dataset_path.startswith("./"):
            if self._dataset_path.endswith(".gz"):
                data = load_dataset(self._dataset_name, data_files=self._dataset_path, split=self._split)
            else:
                data = load_from_disk(self._dataset_path)[self._split]
        else:
            data = load_dataset(self._dataset_path, self._dataset_name, split=self._split)

        datas: List[str] = []
        for sample in data:
            text = sample[self._text_column]
            if len(text) >= length:
                datas.append(" \n" if text == "" else text)
                if len(datas) >= 1024:
                    break

        return "".join(datas)

    def _model_device(self) -> torch.device:
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            try:
                resolved = torch.device(model_device)
                if resolved.type != "meta":
                    return resolved
            except (RuntimeError, TypeError, ValueError):
                pass

        try:
            first_param = next(self._model.parameters())
        except (AttributeError, StopIteration):
            return torch.device("cpu")

        return first_param.device if first_param.device.type != "meta" else torch.device("cpu")

    def _tokenize(self) -> torch.Tensor:
        self._tokenizer.model_max_length = sys.maxsize
        return self._tokenizer(self._text, truncation=False, return_tensors="pt").input_ids

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        return int(pad_token_id)

    def _build_windows(self, tokens: torch.Tensor, starts: Iterable[int], n_ctx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        windows = []
        for start in starts:
            window = tokens[start : min(start + n_ctx + 1, tokens.shape[0])]
            if window.numel() >= 2:
                windows.append(window)

        if not windows:
            empty = torch.empty(0, dtype=tokens.dtype)
            return empty, empty, empty

        max_len = max(window.numel() - 1 for window in windows)
        batch_size = len(windows)
        pad_token_id = self._pad_token_id()

        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=tokens.dtype)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for row, window in enumerate(windows):
            inputs = window[:-1]
            targets = window[1:]
            seq_len = inputs.numel()
            input_ids[row, :seq_len] = inputs
            labels[row, :seq_len] = targets.long()
            attention_mask[row, :seq_len] = 1

        return input_ids, attention_mask, labels

    def calculate(self, n_ctx: int = 512, n_batch: int = 512) -> List[float]:
        """
        Calculate cumulative perplexity values across the evaluation corpus.

        `n_ctx` is the maximum context length per sequence window. `n_batch`
        acts as an approximate token budget per forward pass; windows are batched
        together in groups of `max(1, n_batch // n_ctx)`.
        """
        if n_ctx < 2:
            raise ValueError("Perplexity.calculate: `n_ctx` must be >= 2.")
        if n_batch <= 0:
            raise ValueError("Perplexity.calculate: `n_batch` must be > 0.")

        flat_tokens = self._tokenize()[0].cpu()
        if flat_tokens.numel() < 2:
            return []

        device = self._model_device()
        windows_per_forward = max(1, n_batch // n_ctx)
        starts = list(range(0, flat_tokens.numel() - 1, n_ctx))

        nll = 0.0
        count = 0
        all_perplexity: List[float] = []

        with logger.pb(range(0, len(starts), windows_per_forward)).title("Perplexity: - ").manual() as pb:
            for offset in pb:
                input_ids, attention_mask, labels = self._build_windows(
                    flat_tokens,
                    starts[offset : offset + windows_per_forward],
                    n_ctx,
                )
                if input_ids.numel() == 0:
                    continue

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                with torch.inference_mode():
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits.float()
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                valid_tokens = int(torch.count_nonzero(labels != -100).item())
                if valid_tokens == 0:
                    continue

                nll += float(loss.item())
                count += valid_tokens

                curr_ppl = math.exp(nll / count)
                all_perplexity.append(curr_ppl)
                pb.title(f"Perplexity: {curr_ppl:.4f}").draw()

        return all_perplexity
