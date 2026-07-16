from transformers import GradientCheckpointingLayer
import torch
import torch.nn as nn
from .metis_hyper_memory import create_metis_hyper_memory
from .metis_local_memory import create_metis_local_memory
from ..utils import create_metis_decoder_layer
from abc import ABC


def create_metis_block(config, layer_idx, raw_decoder):
    if getattr(raw_decoder, "layer_type", "full_attention") == "linear_attention":
        return NonMemoryMetisBlock(config, layer_idx, raw_decoder)
    return eval(config.memory_configs['metis_block_type'])(config, layer_idx, raw_decoder)


def _is_boundary_memory_layer(config, layer_idx: int) -> bool:
    text_cfg = getattr(config.backbone_configs, 'text_config', config.backbone_configs)
    num_layers = getattr(text_cfg, "num_hidden_layers", getattr(config, "num_hidden_layers", 0))
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is None:
        return layer_idx in {0, num_layers - 1}
    memory_layers = [
        i for i, lt in enumerate(layer_types)
        if lt != "linear_attention"
    ]
    return bool(memory_layers) and layer_idx in {memory_layers[0], memory_layers[-1]}

class NonMemoryMetisBlock(GradientCheckpointingLayer):
    """Block with no memory modules — used for debugging / ablation.
    Passes through the backbone decoder with no memory read or write."""

    def __init__(self, config, layer_idx, raw_decoder):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.local_memory = None
        self.hyper_memory = None
        self.monitor_branch_norms = False
        self.last_memory_branch_norm = None
        self.last_attention_branch_norm = None
        self.last_memory_attention_norm_ratio = None
        self._backbone_decoder_ref = [create_metis_decoder_layer(config, raw_decoder)]

    @property
    def backbone_decoder(self):
        return self._backbone_decoder_ref[0]

    def forward(self, hidden_states, **kwargs) -> torch.Tensor:
        _, memory_carrier_before_mixin, cache_dict, _ = \
            self.backbone_decoder.before_mixin(hidden_states, **kwargs)
        return self.backbone_decoder.after_mixin(
            memory_carrier_before_mixin, cache_dict, hidden_states, **kwargs,
        )


# ── Basic blocks ────────────────────────────────────────────────────────────

class MetisBlockBase(GradientCheckpointingLayer, ABC):
    def __init__(self, config, layer_idx: int, raw_decoder):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.local_memory = create_metis_local_memory(config)
        self.hyper_memory = create_metis_hyper_memory(config)
        self._backbone_decoder_ref = [create_metis_decoder_layer(config, raw_decoder)]
        self.hyper_memory.register_raw_decoder(self._backbone_decoder_ref)

        # Branch norm monitoring (on boundary full-attention layers only).
        self.monitor_branch_norms = _is_boundary_memory_layer(config, layer_idx)
        self.last_memory_branch_norm = None
        self.last_attention_branch_norm = None
        self.last_memory_attention_norm_ratio = None
        self.last_memory_query_norm = None

    @property
    def backbone_decoder(self):
        return self._backbone_decoder_ref[0]

    def memory_integrate(self, memory_branch, memory_carrier_before_mixin):
        raise NotImplementedError

    @torch.no_grad()
    def _record_branch_norms(self, memory_branch, memory_carrier_before_mixin):
        if not self.monitor_branch_norms:
            return
        mem_norm = memory_branch.detach().float().norm()
        attn_norm = memory_carrier_before_mixin.detach().float().norm()
        self.last_memory_branch_norm = mem_norm.item()
        self.last_attention_branch_norm = attn_norm.item()
        self.last_memory_attention_norm_ratio = (
            mem_norm / attn_norm.clamp_min(1e-12)
        ).item()

    @torch.no_grad()
    def _record_query_norm(self, query_for_memory):
        self.last_memory_query_norm = (
            query_for_memory.detach().float().norm(dim=-1).mean().item()
        )

    def _init_learned_query(self, raw_decoder):
        text_cfg = getattr(self.config.backbone_configs, 'text_config', self.config.backbone_configs)
        hidden_dim = text_cfg.hidden_size
        self.query_num_heads = text_cfg.num_attention_heads
        self.query_head_dim = getattr(text_cfg, "head_dim", hidden_dim // self.query_num_heads)
        mem_dim = self.query_num_heads * self.query_head_dim

        self_attn = getattr(raw_decoder, "self_attn", None)
        q_proj = getattr(self_attn, "q_proj", None) if self_attn is not None else None
        has_q_bias = q_proj is not None and getattr(q_proj, "bias", None) is not None
        self.query_proj = nn.Linear(hidden_dim, mem_dim, bias=has_q_bias)
        nn.init.normal_(self.query_proj.weight, std=hidden_dim ** -0.5)
        if self.query_proj.bias is not None:
            nn.init.zeros_(self.query_proj.bias)

        ln = getattr(self_attn, "q_norm", None) or raw_decoder.input_layernorm
        norm_cls = type(ln)
        eps = getattr(ln, "variance_epsilon", getattr(ln, "eps", 1e-6))
        self.query_norm = norm_cls(self.query_head_dim, eps=eps)

    def _make_learned_query(self, hidden_states):
        input_shape = hidden_states.shape[:-1]
        query_for_memory = self.query_proj(hidden_states).view(
            *input_shape, self.query_num_heads, self.query_head_dim,
        )
        query_for_memory = self.query_norm(query_for_memory).transpose(1, 2).contiguous()
        self._record_query_norm(query_for_memory)
        return query_for_memory

    def forward(self, hidden_states, **kwargs) -> torch.Tensor:
        query_for_memory, memory_carrier_before_mixin, cache_dict, o_proj_weight = \
            self.backbone_decoder.before_mixin(hidden_states, **kwargs)
        if query_for_memory is None:
            return self.backbone_decoder.after_mixin(
                memory_carrier_before_mixin, cache_dict, hidden_states, **kwargs,
            )
        self._record_query_norm(query_for_memory)
        memory_branch = o_proj_weight(self.local_memory.read(query_for_memory))
        self._record_branch_norms(memory_branch, memory_carrier_before_mixin)
        memory_carrier_after_mixin = self.memory_integrate(memory_branch, memory_carrier_before_mixin)
        return self.backbone_decoder.after_mixin(
            memory_carrier_after_mixin, cache_dict, hidden_states, **kwargs,
        )

class NaiveMetisBlock(MetisBlockBase):
    def memory_integrate(self, memory_branch, memory_carrier_before_mixin):
        return memory_branch + memory_carrier_before_mixin

# ── Normed blocks ───────────────────────────────────────────────────────────

class NormedNaiveMetisBlock(NaiveMetisBlock):
    """NaiveMetisBlock with RMSNorm applied to the memory readout.

    Inserts mem_norm between local_memory.read() and o_proj:
        before: memory_branch = o_proj( M.read(q) )
        after:  memory_branch = o_proj( mem_norm( M.read(q) ) )

    mem_norm_init defaults to 1.0 (safe backbone-style init).  Set
    memory_configs['mem_norm_init'] to a float to override (e.g. 0.2).
    """

    def __init__(self, config, layer_idx: int, raw_decoder):
        super().__init__(config, layer_idx, raw_decoder)
        text_cfg = getattr(config.backbone_configs, 'text_config', config.backbone_configs)
        mem_dim = text_cfg.num_attention_heads * text_cfg.head_dim

        ln = raw_decoder.input_layernorm
        norm_cls = type(ln)
        eps = getattr(ln, "variance_epsilon", getattr(ln, "eps", 1e-6))
        self.mem_norm = norm_cls(mem_dim, eps=eps)

        init_val = float(config.memory_configs.get('mem_norm_init', 1.0))
        if hasattr(self.mem_norm, "weight") and init_val != 1.0:
            with torch.no_grad():
                self.mem_norm.weight.fill_(init_val)

    def forward(self, hidden_states, **kwargs) -> torch.Tensor:
        query_for_memory, memory_carrier_before_mixin, cache_dict, o_proj_weight = \
            self.backbone_decoder.before_mixin(hidden_states, **kwargs)
        if query_for_memory is None:
            return self.backbone_decoder.after_mixin(
                memory_carrier_before_mixin, cache_dict, hidden_states, **kwargs,
            )
        self._record_query_norm(query_for_memory)
        normed_mem = self.mem_norm(self.local_memory.read(query_for_memory))
        memory_branch = o_proj_weight(normed_mem)
        self._record_branch_norms(memory_branch, memory_carrier_before_mixin)
        memory_carrier_after_mixin = self.memory_integrate(memory_branch, memory_carrier_before_mixin)
        return self.backbone_decoder.after_mixin(
            memory_carrier_after_mixin, cache_dict, hidden_states, **kwargs,
        )


# ── Reweight blocks ─────────────────────────────────────────────────────────

class NormedReweightMetisBlock(NormedNaiveMetisBlock):
    """Normed memory + reweighted integration.
    Attention = gamma * Original Attention + (1 - gamma) * Memory Attention."""

    def __init__(self, config, layer_idx, raw_decoder):
        super().__init__(config, layer_idx, raw_decoder)
        self.gamma = config.memory_configs.get('metis_reweight_gamma', 0.9)

    def memory_integrate(self, memory_branch, memory_carrier_before_mixin):
        return self.gamma * memory_carrier_before_mixin + (1 - self.gamma) * memory_branch


class NormedReweightLearnedQueryMetisBlock(NormedReweightMetisBlock):
    """Normed reweight block with a dedicated trainable memory-read query."""

    def __init__(self, config, layer_idx: int, raw_decoder):
        super().__init__(config, layer_idx, raw_decoder)
        self._init_learned_query(raw_decoder)

    def forward(self, hidden_states, **kwargs) -> torch.Tensor:
        query_for_memory, memory_carrier_before_mixin, cache_dict, o_proj_weight = \
            self.backbone_decoder.before_mixin(hidden_states, **kwargs)
        if query_for_memory is None:
            return self.backbone_decoder.after_mixin(
                memory_carrier_before_mixin, cache_dict, hidden_states, **kwargs,
            )
        query_for_memory = self._make_learned_query(hidden_states)
        normed_mem = self.mem_norm(self.local_memory.read(query_for_memory))
        memory_branch = o_proj_weight(normed_mem)
        self._record_branch_norms(memory_branch, memory_carrier_before_mixin)
        memory_carrier_after_mixin = self.memory_integrate(memory_branch, memory_carrier_before_mixin)
        return self.backbone_decoder.after_mixin(
            memory_carrier_after_mixin, cache_dict, hidden_states, **kwargs,
        )


class NormedGatedMetisBlock(NormedNaiveMetisBlock):
    """NormedNaiveMetisBlock with a query-conditioned single-layer linear gate.

    Replaces the fixed ``gamma`` scalar with a single-layer MLP (no bias) that
    takes the block input (``hidden_states``) and produces a per-token blending
    coefficient:

        g      = sigmoid( gamma_gate_proj(hidden_states) )   # [B, T, 1]
        output = g * attn_out + (1 - g) * memory_branch

    Weights are initialised with a small normal std (0.01) so ``sigmoid`` output
    starts near 0.5 for all positions.

    The parent's ``mem_norm`` / ``mem_norm_init`` logic is fully preserved.
    """

    def __init__(self, config, layer_idx: int, raw_decoder):
        super().__init__(config, layer_idx, raw_decoder)
        text_cfg = getattr(config.backbone_configs, 'text_config', config.backbone_configs)
        hidden_dim = text_cfg.hidden_size

        self.gamma_gate_proj = nn.Linear(hidden_dim, 1, bias=False)
        # Small init so sigmoid output starts near 0.5 (≈ uniform gate).
        nn.init.normal_(self.gamma_gate_proj.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states, **kwargs) -> torch.Tensor:
        query_for_memory, memory_carrier_before_mixin, cache_dict, o_proj_weight = \
            self.backbone_decoder.before_mixin(hidden_states, **kwargs)
        if query_for_memory is None:
            return self.backbone_decoder.after_mixin(
                memory_carrier_before_mixin, cache_dict, hidden_states, **kwargs,
            )
        self._record_query_norm(query_for_memory)
        normed_mem = self.mem_norm(self.local_memory.read(query_for_memory))
        memory_branch = o_proj_weight(normed_mem)
        self._record_branch_norms(memory_branch, memory_carrier_before_mixin)
        # Per-token gate conditioned on the block input hidden state.
        g = torch.sigmoid(self.gamma_gate_proj(hidden_states))   # [B, T, 1]
        memory_carrier_after_mixin = g * memory_carrier_before_mixin + (1.0 - g) * memory_branch
        return self.backbone_decoder.after_mixin(
            memory_carrier_after_mixin, cache_dict, hidden_states, **kwargs,
        )


class NormedGatedLearnedQueryMetisBlock(NormedGatedMetisBlock):
    """NormedGatedMetisBlock where ``query_for_memory`` is derived from a
    dedicated learnable projection of ``hidden_states`` rather than the
    backbone attention Q output.

    Architecture:
        query_for_memory = query_norm( query_proj(hidden_states) )   # [B, H, T, D]
        normed_mem       = mem_norm( local_memory.read(query_for_memory) )
        memory_branch    = o_proj(normed_mem)
        g                = sigmoid( gamma_gate_proj(hidden_states) ) # [B, T, 1]
        output           = g * attn_out + (1 - g) * memory_branch

    ``query_proj`` maps ``hidden_dim → num_heads × head_dim``; its bias setting
    follows the backbone q_proj.
    ``query_norm`` is the same RMSNorm class as the backbone's attention Q norm,
    applied in head space so the memory read is scale-normalised.

    Initialisation: ``query_proj`` uses a 1/√hidden_dim normal std, matching
    standard attention Q-projection scale; ``query_norm`` starts at weight 1.
    """

    def __init__(self, config, layer_idx: int, raw_decoder):
        super().__init__(config, layer_idx, raw_decoder)
        self._init_learned_query(raw_decoder)

    def forward(self, hidden_states, **kwargs) -> torch.Tensor:
        query_for_memory, memory_carrier_before_mixin, cache_dict, o_proj_weight = \
            self.backbone_decoder.before_mixin(hidden_states, **kwargs)
        if query_for_memory is None:
            return self.backbone_decoder.after_mixin(
                memory_carrier_before_mixin, cache_dict, hidden_states, **kwargs,
            )
        # Replace backbone Q with a learned projection from hidden_states.
        query_for_memory = self._make_learned_query(hidden_states)
        normed_mem = self.mem_norm(self.local_memory.read(query_for_memory))
        memory_branch = o_proj_weight(normed_mem)
        self._record_branch_norms(memory_branch, memory_carrier_before_mixin)
        g = torch.sigmoid(self.gamma_gate_proj(hidden_states))        # [B, T, 1]
        memory_carrier_after_mixin = g * memory_carrier_before_mixin + (1.0 - g) * memory_branch
        return self.backbone_decoder.after_mixin(
            memory_carrier_after_mixin, cache_dict, hidden_states, **kwargs,
        )


class NormedSwiGLUGatedMetisBlock(NormedNaiveMetisBlock):
    """NormedNaiveMetisBlock with a two-layer SwiGLU gate MLP.

    The blending gate is computed by a SwiGLU MLP over the block input:

        h = SiLU( gate_proj(hidden_states) ) * up_proj(hidden_states)   # [B, T, D_gate]
        g = sigmoid( down_proj(h) )                                      # [B, T, 1]
        output = g * attn_out + (1 − g) * memory_branch

    Intermediate dimension ``D_gate`` defaults to ``hidden_dim // 4`` and can
    be overridden via ``memory_configs['swiglu_gate_hidden_dim']``.

    Initialization: ``gate_proj`` / ``up_proj`` use small-std normal so the
    SwiGLU activations start near zero; ``down_proj`` is zeroed so
    ``sigmoid(0) = 0.5`` at the first step (balanced blending).
    """

    def __init__(self, config, layer_idx: int, raw_decoder):
        super().__init__(config, layer_idx, raw_decoder)
        text_cfg = getattr(config.backbone_configs, 'text_config', config.backbone_configs)
        hidden_dim = text_cfg.hidden_size
        gate_dim = int(config.memory_configs.get('swiglu_gate_hidden_dim', hidden_dim // 4))

        self.gamma_gate_proj = nn.Linear(hidden_dim, gate_dim, bias=False)
        self.gamma_up_proj   = nn.Linear(hidden_dim, gate_dim, bias=False)
        self.gamma_down_proj = nn.Linear(gate_dim,   1,        bias=False)

        nn.init.normal_(self.gamma_gate_proj.weight, std=0.01)
        nn.init.normal_(self.gamma_up_proj.weight,   std=0.01)
        nn.init.zeros_(self.gamma_down_proj.weight)

    def forward(self, hidden_states, **kwargs) -> torch.Tensor:
        query_for_memory, memory_carrier_before_mixin, cache_dict, o_proj_weight = \
            self.backbone_decoder.before_mixin(hidden_states, **kwargs)
        if query_for_memory is None:
            return self.backbone_decoder.after_mixin(
                memory_carrier_before_mixin, cache_dict, hidden_states, **kwargs,
            )
        self._record_query_norm(query_for_memory)
        normed_mem = self.mem_norm(self.local_memory.read(query_for_memory))
        memory_branch = o_proj_weight(normed_mem)
        self._record_branch_norms(memory_branch, memory_carrier_before_mixin)
        h = nn.functional.silu(self.gamma_gate_proj(hidden_states)) * self.gamma_up_proj(hidden_states)
        g = torch.sigmoid(self.gamma_down_proj(h))                   # [B, T, 1]
        memory_carrier_after_mixin = g * memory_carrier_before_mixin + (1.0 - g) * memory_branch
        return self.backbone_decoder.after_mixin(
            memory_carrier_after_mixin, cache_dict, hidden_states, **kwargs,
        )
