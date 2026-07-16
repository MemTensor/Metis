# Define some utils for constructing Metis.
from transformers import GenerationMixin

class TrajectoryGenerationMixin(GenerationMixin):
    def reset(self):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    def step_generate(self, input_ids, **kwargs):
        outputs = self.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_hidden_states=False,
            **kwargs
        )

        trajectory_ids = outputs.sequences

        # Re-forward the trajectory to get the final hidden states.
        final_outputs = self.model.forward(
            input_ids=trajectory_ids,
            output_hidden_states=True, 
            use_cache=False 
        )

        self.commit(final_outputs)
        
        return trajectory_ids
    
import torch
import torch.nn as nn
import importlib

def create_metis_decoder_layer(config, raw_decoder):
    module = importlib.import_module('metis.backbone_wrappers.%s_wrapper' % config.backbone_meta['backbone_type'])
    decoder_layer_class = getattr(module, '%sDecoderLayerForMetis' % config.backbone_meta['backbone_type'])

    return decoder_layer_class(config, raw_decoder)

def create_metis_causallm(config):
    module = importlib.import_module('metis.backbone_wrappers.%s_wrapper' % config.backbone_meta['backbone_type'])
    model_class = getattr(module, '%sCausalLMForMetis' % config.backbone_meta['backbone_type'])
    
    return model_class(config)

class DecoderLayerWrapperForMetis(nn.Module):
    def __init__(self, config, raw_decoder):
        super().__init__()
        self.config = config
        self._raw_decoder_ref = [raw_decoder]
    
    @property
    def raw_decoder(self):
        return self._raw_decoder_ref[0]

    def before_mixin(self, **kwargs):
        raise NotImplementedError
    
    def after_mixin(self, memory_carrier, cache_dict, **kwargs) -> torch.Tensor:
        raise NotImplementedError

class CausalLMWrapperForMetis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def register_metis_blocks(self, metis_blocks):
        # Store as a plain list so PyTorch does NOT register these as submodules
        # of this wrapper. The MetisModel already owns the ModuleList; registering
        # it here too would create duplicate state-dict keys (shared-tensor error).
        self._metis_blocks_ref = list(metis_blocks)

    def get_decoder_layer_by_id(self, layer_id: int):
        raise NotImplementedError
    
    def forward_with_memory(self, **kwargs):
        raise NotImplementedError

