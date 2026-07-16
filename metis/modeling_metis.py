from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_metis import MetisConfig
from .utils import TrajectoryGenerationMixin, create_metis_causallm
from .dev_beta.metis_block import create_metis_block


class MetisPreTrainedModel(PreTrainedModel):
    config_class = MetisConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MetisBlock"]
    _skip_keys_device_placement = "past_key_values"
    _tied_weights_keys = []

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)


class MetisModel(MetisPreTrainedModel):
    def __init__(self, config: MetisConfig):
        super().__init__(config)

        self.metis_backbone = create_metis_causallm(config)

        self.metis_blocks = nn.ModuleList(
            [create_metis_block(config, i, self.metis_backbone.get_decoder_layer_by_id(i))
             for i in range(self.metis_backbone.model.config.num_hidden_layers)]
        )
        self.metis_backbone.register_metis_blocks(self.metis_blocks)

        self.post_init()

    def forward(self, **kwargs) -> BaseModelOutputWithPast:
        return self.metis_backbone.forward_with_memory(**kwargs)


class MetisForCausalLM(MetisPreTrainedModel, TrajectoryGenerationMixin):
    def __init__(self, config: MetisConfig):
        super().__init__(config)
        self.model = MetisModel(config)
        self.post_init()

    @torch.no_grad()
    def reset(self):
        for layer in self.model.metis_blocks:
            if layer.local_memory is None:
                continue
            layer.local_memory.reset()

    # Backward-compat alias.
    reset_memory = reset

    def _commit_memory(self, outputs, attention_mask=None):
        """Write per-layer hidden states into local memory with optional mask.

        attention_mask is passed through to hyper_memory.update_local_memory
        so mask-aware variants (e.g. LinearLastMetisHyperMemory) can select
        the last *real* token rather than the last position.
        """
        offset = self.config.memory_configs.get('commit_hidden_offset', 0)
        if offset not in (0, 1):
            raise ValueError(f"commit_hidden_offset must be 0 or 1, got {offset!r}")
        all_hidden = outputs.hidden_states
        for k, layer in enumerate(self.model.metis_blocks):
            if layer.local_memory is None:
                continue
            layer_h = all_hidden[k + offset]
            layer.hyper_memory.update_local_memory(
                layer_h, layer.local_memory, attention_mask=attention_mask,
            )

    def commit(self, outputs):
        """Public commit wrapper.  For mask-aware writes use _commit_memory directly."""
        self._commit_memory(outputs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        commit_memory: bool = False,
        attention_mask_1d: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if commit_memory:
            kwargs['output_hidden_states'] = True

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.model.metis_backbone.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if commit_memory:
            mask_1d = attention_mask_1d
            if mask_1d is None and attention_mask is not None and attention_mask.ndim == 2:
                mask_1d = attention_mask
            self._commit_memory(outputs, attention_mask=mask_1d)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
