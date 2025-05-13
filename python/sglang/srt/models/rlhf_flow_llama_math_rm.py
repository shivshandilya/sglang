# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig, AutoTokenizer

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM, LlamaModel
from sglang.srt.utils import add_prefix


class RLHFlowLlamaMathRM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.num_labels = config.num_labels
        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False, offset=3)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.candidate_tokens = [10, 12]
        self.eos_token_id = config.eos_token_id

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        """
        Assumes that each sample is formatted like a chat where the final assistant message is "+" or "-"
        and we extract logits for that token prediction.
        """
        # Get hidden states from base model
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        logits= self.lm_head(hidden_states.to(self.lm_head.weight.dtype))

        logits = self.pooler(logits, forward_batch).embeddings

        target_logits= logits[:, self.candidate_tokens]

        plus_step_scores = target_logits.softmax(dim=-1)[:, 0].unsqueeze(1)

        return EmbeddingPoolerOutput(plus_step_scores)
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return LlamaForCausalLM.load_weights(self, weights)

EntryClass = [
    RLHFlowLlamaMathRM,
]
