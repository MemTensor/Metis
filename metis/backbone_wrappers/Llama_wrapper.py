from typing import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt
from transformers import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.utils import TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from typing_extensions import Unpack

from ..utils import CausalLMWrapperForMetis, DecoderLayerWrapperForMetis


class LlamaDecoderLayerForMetis(DecoderLayerWrapperForMetis):
    def __init__(self, config, raw_decoder):
        super().__init__(config, raw_decoder)

    def before_mixin(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self_attn = self.raw_decoder.self_attn

        def _attn(hidden_states):
            residual = hidden_states
            normed = self.raw_decoder.input_layernorm(hidden_states)

            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            query_states = self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
            key_states = self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
            value_states = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

            memory_for_query = query_states.clone()

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self_attn.layer_idx
                )

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self_attn.config._attn_implementation, eager_attention_forward
            )

            attn_output, _ = attention_interface(
                self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self_attn.attention_dropout,
                scaling=self_attn.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self_attn.o_proj(attn_output)
            return memory_for_query, attn_output, residual

        if getattr(self, "_use_gradient_checkpointing", False) and self.training:
            memory_for_query, attn_output, residual = _ckpt(_attn, hidden_states, use_reentrant=False)
        else:
            memory_for_query, attn_output, residual = _attn(hidden_states)

        return memory_for_query, attn_output, {"residual": residual}, self_attn.o_proj

    def after_mixin(
        self,
        memory_carrier,
        cache_dict,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = cache_dict["residual"]
        hidden_states = residual + memory_carrier

        def _mlp(hidden_states):
            residual = hidden_states
            normed = self.raw_decoder.post_attention_layernorm(hidden_states)
            return residual + self.raw_decoder.mlp(normed)

        if getattr(self, "_use_gradient_checkpointing", False) and self.training:
            hidden_states = _ckpt(_mlp, hidden_states, use_reentrant=False)
        else:
            hidden_states = _mlp(hidden_states)

        return hidden_states


class LlamaCausalLMForMetis(CausalLMWrapperForMetis):
    def __init__(self, config):
        super().__init__(config)

        self.model = LlamaModel(config.backbone_configs)
        self.vocab_size = config.backbone_configs.vocab_size
        self.lm_head = nn.Linear(
            config.backbone_configs.hidden_size,
            config.backbone_configs.vocab_size,
            bias=False,
        )

        self.model.post_init()

    def get_decoder_layer_by_id(self, layer_id: int):
        return self.model.layers[layer_id]

    @merge_with_config_defaults
    @capture_outputs
    def forward_with_memory(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.model.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        all_hidden_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids=position_ids)

        for layer_idx, _decoder_layer in enumerate(self.model.layers[: self.model.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = self._metis_blocks_ref[layer_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )
