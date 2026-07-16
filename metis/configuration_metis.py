from transformers import PretrainedConfig, AutoConfig


class MetisConfig(PretrainedConfig):
    model_type = "metis"

    def __init__(
        self,
        backbone_meta: dict | None = None,
        backbone_configs=None,
        memory_configs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_meta = dict(backbone_meta or {
            'backbone_type': "Qwen3_5",
            'backbone_path': "",
        })

        if backbone_configs is not None:
            # When loading from saved config JSON, backbone_configs arrives
            # as a plain dict.  Convert it back to the proper config class.
            if isinstance(backbone_configs, dict) and "model_type" in backbone_configs:
                cfg = dict(backbone_configs)
                model_type = cfg.pop("model_type")
                backbone_configs = AutoConfig.for_model(model_type, **cfg)
            self.backbone_configs = backbone_configs
        else:
            # Eagerly load only when an explicit path is provided.
            # During default construction (e.g. HuggingFace internal
            # diff/serialization), backbone_path is "" so this is skipped.
            backbone_path = self.backbone_meta.get('backbone_path', '')
            if backbone_path:
                self.backbone_configs = AutoConfig.from_pretrained(backbone_path)
            else:
                self.backbone_configs = None

        self.memory_configs = dict(memory_configs or {
            'metis_block_type': 'NormedReweightLearnedQueryMetisBlock',
            'metis_hyper_memory_type': 'StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory',
            'metis_local_memory_type': 'NormalizedDeltaNetMetisLocalMemory',
            'update_ratio': 0.9,
            'commit_hidden_offset': 0,
            'metis_reweight_gamma': 0.9,
            'gated_delta_alpha_init': 1.0,
            'gated_delta_beta_init': 1.0,
        })

        # Expose backbone fields that HuggingFace internals (e.g. cache init,
        # generation utils) expect to find directly on the top-level config.
        if self.backbone_configs is not None:
            text_cfg = getattr(self.backbone_configs, 'text_config', self.backbone_configs)
            self.num_hidden_layers = text_cfg.num_hidden_layers
            self.num_attention_heads = text_cfg.num_attention_heads
            self.hidden_size = text_cfg.hidden_size
            self.bos_token_id = text_cfg.bos_token_id
            self.eos_token_id = text_cfg.eos_token_id
            self.pad_token_id = text_cfg.pad_token_id
