"""Utilities for building MetisForCausalLM from a pretrained backbone."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .configuration_metis import MetisConfig
from .modeling_metis import MetisForCausalLM

# Map CLI-style lowercase names to the CamelCase names used in backbone_meta.
_BACKBONE_TYPE_MAP: dict[str, str] = {
    "qwen3_5": "Qwen3_5",
    "qwen3":   "Qwen3",
    "llama":   "Llama",
}


def _fit_rows(src: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Pad or tile source rows to match the target dimension.

    When src_rows < target_rows and target_rows is a multiple of src_rows,
    each source row is repeated to fill the GQA→MHA gap (repeat_interleave).
    Otherwise the first ``min(src_rows, target_rows)`` rows are copied.
    """
    src_rows, cols = src.shape
    if src_rows >= target_rows:
        return src[:target_rows]
    if target_rows % src_rows == 0:
        return src.repeat_interleave(target_rows // src_rows, dim=0)
    return src[:target_rows]


def _extract_backbone_query_rows(
    q_tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor | None:
    """Extract the query rows from a backbone q_proj tensor.

    Qwen3.5 stores q_proj output as per-head ``[query | gate]`` chunks with
    size ``head_dim * 2``.  Older backbones may store query rows directly.
    """
    if q_tensor.shape[0] == num_heads * head_dim:
        return q_tensor
    if q_tensor.shape[0] == num_heads * head_dim * 2:
        return q_tensor.view(num_heads, head_dim * 2, *q_tensor.shape[1:])[:, :head_dim].reshape(
            num_heads * head_dim,
            *q_tensor.shape[1:],
        )
    return None


@torch.no_grad()
def _init_learned_query_from_backbone(model: MetisForCausalLM) -> None:
    """Initialize trainable memory-read query layers from backbone q_proj/q_norm."""
    inited = 0
    skipped = 0
    for block in model.model.metis_blocks:
        query_proj = getattr(block, "query_proj", None)
        query_norm = getattr(block, "query_norm", None)
        if query_proj is None or query_norm is None:
            skipped += 1
            continue

        raw_decoder = block.backbone_decoder.raw_decoder
        self_attn = getattr(raw_decoder, "self_attn", None)
        src_q_proj = getattr(self_attn, "q_proj", None) if self_attn is not None else None
        src_q_norm = getattr(self_attn, "q_norm", None) if self_attn is not None else None
        if src_q_proj is None:
            skipped += 1
            continue

        num_heads = getattr(block, "query_num_heads", None)
        head_dim = getattr(block, "query_head_dim", None)
        if num_heads is None or head_dim is None:
            skipped += 1
            continue

        src_weight = _extract_backbone_query_rows(src_q_proj.weight, num_heads, head_dim)
        if src_weight is None or src_weight.shape != query_proj.weight.shape:
            skipped += 1
            continue

        query_proj.weight.copy_(src_weight.to(device=query_proj.weight.device, dtype=query_proj.weight.dtype))

        if query_proj.bias is not None:
            src_bias = getattr(src_q_proj, "bias", None)
            if src_bias is not None:
                src_bias = _extract_backbone_query_rows(src_bias, num_heads, head_dim)
                if src_bias is not None and src_bias.shape == query_proj.bias.shape:
                    query_proj.bias.copy_(src_bias.to(device=query_proj.bias.device, dtype=query_proj.bias.dtype))
                else:
                    query_proj.bias.zero_()
            else:
                query_proj.bias.zero_()

        if query_norm is not None and src_q_norm is not None:
            try:
                query_norm.load_state_dict(src_q_norm.state_dict(), strict=True)
            except RuntimeError:
                skipped += 1
                continue

        inited += 1

    print(f"[weight_utils] Learned memory queries initialized from backbone q_proj/q_norm: {inited} layers (skipped {skipped}).")


@torch.no_grad()
def _init_hyper_memory_from_backbone(model: MetisForCausalLM) -> None:
    """Initialize hyper-memory W_k / W_v from decoder projections.

    Supports both layer types in Qwen3.5-style hybrid models:

    full_attention layers  → W_k ← self_attn.k_proj
                             W_v ← self_attn.v_proj
    linear_attention layers→ W_k ← in_proj_qkv key portion
                             W_v ← in_proj_qkv value portion

    Uses k_proj (not q_proj) for W_k init — W_k projects to key space
    so k_proj weights are the semantically correct starting point.
    """
    inited = 0
    skipped = 0
    for block in model.model.metis_blocks:
        hm = getattr(block, "hyper_memory", None)
        if hm is None:
            skipped += 1
            continue

        raw_decoder = block.backbone_decoder.raw_decoder
        w_k = hm.W_k.weight          # (kv_dim, hidden_size)
        w_v = hm.W_v.weight          # (kv_dim, hidden_size)
        k_rows, k_cols = w_k.shape
        v_rows, v_cols = w_v.shape

        src_k = src_v = None

        self_attn = getattr(raw_decoder, "self_attn", None)
        if self_attn is not None:
            # full_attention: k_proj → W_k,  v_proj → W_v
            src_k = getattr(self_attn, "k_proj", None)
            src_v = getattr(self_attn, "v_proj", None)

        if src_k is None or src_v is None:
            linear_attn = getattr(raw_decoder, "linear_attn", None)
            in_proj_qkv = getattr(linear_attn, "in_proj_qkv", None) if linear_attn is not None else None
            in_proj_z   = getattr(linear_attn, "in_proj_z",   None) if linear_attn is not None else None
            if in_proj_qkv is not None and in_proj_z is not None:
                # in_proj_qkv output layout: [Q(key_dim) | K(key_dim) | V(value_dim)]
                qkv_out   = in_proj_qkv.weight.shape[0]
                value_dim = in_proj_z.weight.shape[0]
                key_total = qkv_out - value_dim            # key_dim * 2

                qkv_w = in_proj_qkv.weight
                src_k_w = qkv_w[:key_total]                # Q+K portion → W_k
                src_v_w = qkv_w[key_total:]                # V   portion → W_v

                if src_k_w.shape[1] != k_cols or src_v_w.shape[1] != v_cols:
                    skipped += 1
                    continue

                w_k.zero_()
                w_v.zero_()
                w_k[: min(src_k_w.shape[0], k_rows)].copy_(src_k_w[: min(src_k_w.shape[0], k_rows)])
                w_v[: min(src_v_w.shape[0], v_rows)].copy_(src_v_w[: min(src_v_w.shape[0], v_rows)])
                inited += 1
                continue
            skipped += 1
            continue

        k_w = src_k.weight   # (k_out, hidden_size)
        v_w = src_v.weight   # (v_out, hidden_size)

        if k_w.shape[1] != k_cols or v_w.shape[1] != v_cols:
            skipped += 1
            continue

        w_k.zero_()
        w_v.zero_()
        # Use _fit_rows to handle GQA→MHA dimension mismatch.
        w_k[: min(k_w.shape[0], k_rows)].copy_(_fit_rows(k_w, k_rows)[: min(k_w.shape[0], k_rows)])
        w_v[: min(v_w.shape[0], v_rows)].copy_(_fit_rows(v_w, v_rows)[: min(v_w.shape[0], v_rows)])
        inited += 1

    print(f"[weight_utils] Hyper-memory initialized from backbone projections: {inited} layers (skipped {skipped}).")


def load_metis_from_backbone(
    backbone_path: str,
    backbone_type: str = "qwen3_5",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    metis_block_type: str = "NormedReweightLearnedQueryMetisBlock",
    metis_hyper_memory_type: str = "StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory",
    metis_local_memory_type: str = "NormalizedDeltaNetMetisLocalMemory",
    update_ratio: float = 0.9,
    commit_hidden_offset: int = 0,
    mem_norm_init: float = 1.0,
    uniform_num_selected: int = 16,
    stride_interval: int = 8,
    pool_temperature: float = 1.0,
    gumbel_topk_noise: bool = True,
    alpha_top_p: float = 0.9,
    alpha_min_tokens: int = 1,
    alpha_max_tokens: int = 0,
    alpha_max_fraction: float = 0.0,
    gated_delta_alpha_init: float = 1.0,
    gated_delta_beta_init: float = 1.0,
    qk_kernel_type: str = "elu_plus_one",
    metis_reweight_gamma: float = 0.9,
) -> tuple[MetisForCausalLM, "AutoTokenizer"]:
    """Build a MetisForCausalLM and load pretrained backbone weights into it.

    Args:
        backbone_path: Local path (or HF hub id) to the backbone model.
        backbone_type: One of ``qwen3_5`` / ``qwen3`` / ``llama`` (lowercase).
        device: Target device.
        dtype: Weight dtype (e.g. torch.float16 / torch.bfloat16).
        metis_block_type: Which MetisBlock variant to use.
        metis_hyper_memory_type: Which HyperMemory variant to use.
        metis_local_memory_type: Which LocalMemory variant to use.
        update_ratio: Blend factor for memory write (1.0 = full update).
        commit_hidden_offset: 0 → layer input, 1 → layer output.
        mem_norm_init: Initial value for mem_norm RMSNorm weight.
        uniform_num_selected: N for uniformly-spaced token selection.
        stride_interval: K for stride-based token selection.
        pool_temperature: Softmax temperature for attention-pooling.
        gumbel_topk_noise: Whether GumbelTopK injects noise during training.
        alpha_top_p: Probability mass for AlphaTopP token selection.
        alpha_min_tokens: Minimum tokens for AlphaTopP.
        alpha_max_tokens: Optional fixed cap for AlphaTopP.
        alpha_max_fraction: Optional length-relative cap for AlphaTopP.
        gated_delta_alpha_init: Initial sigmoid value for gated-delta alpha.
        gated_delta_beta_init: Initial sigmoid value for gated-delta beta.
        qk_kernel_type: Feature map for kernelized q/k memory variants.
        metis_reweight_gamma: Gate weight for reweight blocks.

    Returns:
        (model, tokenizer)
    """
    backbone_type_camel = _BACKBONE_TYPE_MAP.get(backbone_type, backbone_type)

    config = MetisConfig(
        backbone_meta={
            "backbone_type": backbone_type_camel,
            "backbone_path": backbone_path,
        },
        memory_configs={
            "metis_block_type": metis_block_type,
            "metis_hyper_memory_type": metis_hyper_memory_type,
            "metis_local_memory_type": metis_local_memory_type,
            "update_ratio": update_ratio,
            "commit_hidden_offset": commit_hidden_offset,
            "mem_norm_init": mem_norm_init,
            "uniform_num_selected": uniform_num_selected,
            "stride_interval": stride_interval,
            "pool_temperature": pool_temperature,
            "gumbel_topk_noise": gumbel_topk_noise,
            "alpha_top_p": alpha_top_p,
            "alpha_min_tokens": alpha_min_tokens,
            "alpha_max_tokens": alpha_max_tokens,
            "alpha_max_fraction": alpha_max_fraction,
            "gated_delta_alpha_init": gated_delta_alpha_init,
            "gated_delta_beta_init": gated_delta_beta_init,
            "qk_kernel_type": qk_kernel_type,
            "metis_reweight_gamma": metis_reweight_gamma,
        },
    )

    model = MetisForCausalLM(config)

    print(f"[weight_utils] Loading backbone weights from {backbone_path} …")
    backbone = AutoModelForCausalLM.from_pretrained(backbone_path, dtype=dtype)
    mb = model.model.metis_backbone
    mb.model.load_state_dict(backbone.model.state_dict())
    mb.lm_head.load_state_dict(backbone.lm_head.state_dict())

    _init_learned_query_from_backbone(model)
    _init_hyper_memory_from_backbone(model)

    del backbone
    torch.cuda.empty_cache()
    print("[weight_utils] Backbone weights loaded.")

    model = model.to(device=device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(backbone_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
