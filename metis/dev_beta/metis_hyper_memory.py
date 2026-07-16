import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

def _qk_kernel(x: torch.Tensor, kernel_type: str = "elu_plus_one") -> torch.Tensor:
    if kernel_type == "elu_plus_one":
        return F.elu(x) + 1.0
    if kernel_type == "relu_square":
        return F.relu(x).square()
    if kernel_type == "softplus":
        return F.softplus(x)
    raise ValueError(f"Unsupported qk kernel type: {kernel_type}")

def create_metis_hyper_memory(config):
    return eval(config.memory_configs['metis_hyper_memory_type'])(config)

class MetisHyperMemoryBase(nn.Module, ABC):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # Qwen 3.5 has text config, but Qwen 3 does not.
        self.text_cfg = getattr(config.backbone_configs, 'text_config', config.backbone_configs)
        # Reference set later by MetisBlock via register_raw_decoder().
        # Stored as a list to avoid registering the backbone decoder as a submodule.
        self._backbone_decoder_ref: list | None = None

    def register_raw_decoder(self, _backbone_decoder_ref: list) -> None:
        """Called by MetisBlock to give HyperMemory access to the backbone decoder.

        Enables hyper memory variants to apply backbone-style normalisations
        (e.g. input_layernorm) on hidden states before computing W_k/W_v.
        """
        self._backbone_decoder_ref = _backbone_decoder_ref

    @property
    def backbone_decoder(self):
        if self._backbone_decoder_ref is None:
            raise RuntimeError("backbone_decoder not registered; call register_raw_decoder first")
        return self._backbone_decoder_ref[0]

    def update_local_memory(self, raw_info, local_memory) -> None:
        local_memory.write(self.get_new_info_for_local_memory(raw_info))

    def get_new_info_for_local_memory(self, raw_info):
        raise NotImplementedError


class LinearLastMetisHyperMemory(MetisHyperMemoryBase):
    """Additive memory update using the last token's hidden state.

    M_new = (1 - update_ratio) * M_old + update_ratio * (W_k(h_norm)^T @ W_v(h_norm))

    where h_norm = backbone.input_layernorm(h_last).  Applying the backbone's
    RMSNorm before W_k / W_v bounds pre-projection magnitudes (mirroring how
    the backbone's own attention consumes its input).

    All token-selection subclasses (Uniform / Stride / AllTokens) follow the
    same layernorm-then-project pattern.  The exception is
    ``NormalizedLinearLastMetisHyperMemory``, which keeps the legacy
    L2-normalize-on-output behaviour.

    update_ratio is read from memory_configs (default: 1.0).
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        hidden_size = self.text_cfg.hidden_size

        num_q_heads  = self.text_cfg.num_attention_heads
        num_kv_heads = getattr(self.text_cfg, "num_key_value_heads", num_q_heads)
        head_dim     = getattr(self.text_cfg, "head_dim", hidden_size // num_q_heads)

        # W_k / W_v output dim must equal the local memory matrix's kv_dim so
        # that write vectors align with read queries.  Default = GQA layout
        # (num_kv_heads * head_dim).  Switch to MHA layout (num_q_heads * head_dim)
        # when the user chose the legacy MHA local memory.
        local_mem_type = config.memory_configs.get('metis_local_memory_type', '')
        if local_mem_type.startswith('MHA'):
            self.kv_dim = num_q_heads * head_dim
        else:
            self.kv_dim = num_kv_heads * head_dim

        self.update_ratio = config.memory_configs.get('update_ratio', 1.0)

        self.W_k = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.W_v = nn.Linear(hidden_size, self.kv_dim, bias=False)

    def get_new_info_for_local_memory(self, raw_info: torch.Tensor,
                                       attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the memory delta from the last *real* token of each sample.

        Args:
            raw_info:       hidden states  (b, s, hidden_size)
            attention_mask: binary mask    (b, s)  — 1 for real tokens, 0 for pad.
                            When None, the last position is used (safe for unbatched
                            or already-trimmed sequences).

        Returns:
            delta: (b, D, D) — outer product of write key and write value.
        """
        if attention_mask is not None:
            # last real token index per sample: sum of 1s minus 1
            last_idx = attention_mask.sum(dim=1) - 1          # (b,)
            b = raw_info.size(0)
            h_last = raw_info[torch.arange(b, device=raw_info.device),
                               last_idx, :].unsqueeze(1)       # (b, 1, hidden_size)
        else:
            h_last = raw_info[:, -1:, :]                      # (b, 1, hidden_size)
        h_last = self.backbone_decoder.raw_decoder.input_layernorm(h_last)
        write_key = self.W_k(h_last)                          # (b, 1, kv_dim)
        write_value = self.W_v(h_last)                        # (b, 1, kv_dim)
        # (b, kv_dim, 1) @ (b, 1, kv_dim) -> (b, kv_dim, kv_dim)
        return torch.matmul(write_key.transpose(-1, -2), write_value)

    def update_local_memory(self, raw_info: torch.Tensor, local_memory,
                            attention_mask: torch.Tensor | None = None) -> None:
        """Blend the existing memory with the new additive update."""
        delta = self.get_new_info_for_local_memory(raw_info, attention_mask)
        if local_memory.state is not None:
            new_state = (1.0 - self.update_ratio) * local_memory.state + self.update_ratio * delta
        else:
            new_state = self.update_ratio * delta
        local_memory.write(new_state)


class NormalizedLinearLastMetisHyperMemory(LinearLastMetisHyperMemory):
    """Exception class: legacy L2-normalized W_k/W_v output, no input_layernorm.

    Unlike all other subclasses (which apply backbone.input_layernorm to
    hidden states before W_k / W_v), this class operates on raw hidden
    states and L2-normalises the projection *outputs*:

        write_key   = F.normalize(W_k(h_last), dim=-1)   # ‖·‖ = 1
        write_value = F.normalize(W_v(h_last), dim=-1)   # ‖·‖ = 1
        ‖delta‖_F   = ‖write_key‖ · ‖write_value‖ = 1

    The per-step memory increment is bounded by update_ratio.  Kept
    primarily for reproducing earlier experiments.
    """

    def get_new_info_for_local_memory(self, raw_info: torch.Tensor,
                                       attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1          # (b,)
            b = raw_info.size(0)
            h_last = raw_info[torch.arange(b, device=raw_info.device),
                               last_idx, :].unsqueeze(1)       # (b, 1, hidden_size)
        else:
            h_last = raw_info[:, -1:, :]                      # (b, 1, hidden_size)
        write_key   = F.normalize(self.W_k(h_last), dim=-1)   # (b, 1, D), ‖·‖=1
        write_value = F.normalize(self.W_v(h_last), dim=-1)   # (b, 1, D), ‖·‖=1
        # (b, D, 1) @ (b, 1, D) -> (b, D, D),  ‖delta‖_F ≤ 1
        return torch.matmul(write_key.transpose(-1, -2), write_value)


class UniformNormalizedMetisHyperMemory(LinearLastMetisHyperMemory):
    """Memory update using uniformly sampled tokens with backbone-norm pre-projection.

    This class selects N = ``uniform_num_selected`` tokens evenly spaced
    across the real sequence (always including the last real token),
    applies the backbone's input_layernorm to the selected hidden states,
    then projects with W_k / W_v.  Each selected token contributes one
    rank-1 outer product to the memory delta:

        step  = L / N                 (L = real sequence length)
        idx_j = round(j * step)  for j in 0..N-1
        idx_{N-1} = L - 1             (force-include last)

        h_normed   = input_layernorm(h[idx])         # (b, N, hidden)
        write_key  = W_k(h_normed)                   # (b, N, kv_dim)
        write_val  = W_v(h_normed)                   # (b, N, kv_dim)
        delta      = write_key.T @ write_val         # (b, kv_dim, kv_dim)

    Configurable via ``memory_configs``:
        - ``uniform_num_selected``  (int, default 16): number of tokens N

    When the real sequence is shorter than N, all real tokens are used
    and the last one is repeated to fill the remaining slots.
    """

    DEFAULT_NUM_SELECTED: int = 16

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_selected = int(
            config.memory_configs.get('uniform_num_selected', self.DEFAULT_NUM_SELECTED)
        )

    def _select_tokens(
        self,
        hidden_states: torch.Tensor,            # (b, s, hidden_size)
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:                           # (b, N, hidden_size)
        b, S, hidden_size = hidden_states.shape
        N = self.num_selected
        device = hidden_states.device

        # Real sequence length per sample.
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long()          # (b,)
        else:
            lengths = torch.full((b,), S, dtype=torch.long, device=device)

        # Build per-sample index tensors (b, N).
        indices_list = []
        for bi in range(b):
            L = lengths[bi].item()
            if L <= N:
                # Fewer real tokens than slots: use all, repeat last to pad.
                idx = list(range(L)) + [L - 1] * (N - L)
            else:
                # Uniformly spaced: step = L/N, always land last on L-1.
                step = L / N
                idx = [min(int(i * step), L - 1) for i in range(N)]
                idx[-1] = L - 1
            indices_list.append(idx)

        indices = torch.tensor(indices_list, dtype=torch.long, device=device)
        idx_exp = indices.unsqueeze(-1).expand(b, N, hidden_size)  # (b, N, hidden)
        return hidden_states.gather(1, idx_exp)                    # (b, N, hidden)

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,                 # (b, s, hidden_size)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                           # (b, kv_dim, kv_dim)
        h_sel = self._select_tokens(raw_info, attention_mask)         # (b, N, hidden)
        # Apply backbone's RMSNorm (same one that gates the layer's attention).
        h_sel = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        write_key   = self.W_k(h_sel)                                 # (b, N, kv_dim)
        write_value = self.W_v(h_sel)                                 # (b, N, kv_dim)
        # (b, kv_dim, N) @ (b, N, kv_dim) -> (b, kv_dim, kv_dim)
        return torch.matmul(write_key.transpose(-1, -2), write_value)


class StrideNormalizedMetisHyperMemory(LinearLastMetisHyperMemory):
    """Memory update using stride-based token selection with backbone-norm pre-projection.

    Unlike ``UniformNormalizedMetisHyperMemory`` (fixed-N evenly-spaced),
    this class selects **every K-th real token** from each sample, plus
    the last real token.  K is configurable; the number of selected tokens
    per sample varies with sequence length:

        L  = real (non-padding) sequence length
        K  = ``stride_interval``  (config, default 16)
        idx = [0, K, 2K, ...]  intersected with [0, L-1],  union {L-1}

    For mixed-length batches the per-sample selection counts differ; padded
    slots in the resulting (b, N_max, hidden) tensor are masked to zero so
    they contribute nothing to the rank-1 outer products.

    Configurable via ``memory_configs``:
        - ``stride_interval``  (int, default 16): K, the spacing between picks
    """

    DEFAULT_STRIDE: int = 16

    def __init__(self, config) -> None:
        super().__init__(config)
        self.stride = int(config.memory_configs.get('stride_interval', self.DEFAULT_STRIDE))
        if self.stride <= 0:
            raise ValueError(f"stride_interval must be > 0, got {self.stride}")

    def _select_tokens_with_mask(
        self,
        hidden_states: torch.Tensor,            # (b, s, hidden_size)
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:    # (b, N_max, hidden), (b, N_max)
        b, S, hidden_size = hidden_states.shape
        device = hidden_states.device
        K = self.stride

        # Real length per sample — only non-padding tokens are eligible.
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long().tolist()
        else:
            lengths = [S] * b

        # Per-sample stride-K indices, always force-including the last real token.
        per_sample_idx: list[list[int]] = []
        for L in lengths:
            if L <= 0:
                # Edge case: empty sample.  Use index 0 (will be masked out).
                per_sample_idx.append([0])
                continue
            idx = list(range(0, L, K))
            if idx[-1] != L - 1:
                idx.append(L - 1)
            per_sample_idx.append(idx)

        N_max = max(len(idx) for idx in per_sample_idx)

        # Right-pad each sample's index list with 0 (a real position) and
        # record a 0/1 mask so padded slots contribute zero to the outer product.
        indices_padded: list[list[int]] = []
        masks: list[list[float]] = []
        for idx, L in zip(per_sample_idx, lengths):
            n_valid = len(idx) if L > 0 else 0
            pad_n = N_max - len(idx)
            indices_padded.append(idx + [0] * pad_n)
            masks.append([1.0] * n_valid + [0.0] * (N_max - n_valid))

        indices = torch.tensor(indices_padded, dtype=torch.long, device=device)
        mask    = torch.tensor(masks, dtype=hidden_states.dtype, device=device)

        idx_exp = indices.unsqueeze(-1).expand(b, N_max, hidden_size)  # (b, N_max, hidden)
        h_sel   = hidden_states.gather(1, idx_exp)                     # (b, N_max, hidden)
        return h_sel, mask

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,                 # (b, s, hidden_size)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                           # (b, kv_dim, kv_dim)
        h_sel, mask = self._select_tokens_with_mask(raw_info, attention_mask)
        # Apply backbone's RMSNorm before the W_k / W_v projections.
        h_sel = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        write_key   = self.W_k(h_sel)                                   # (b, N, kv_dim)
        write_value = self.W_v(h_sel)                                   # (b, N, kv_dim)
        # Zero-out padded slots so they contribute nothing to the matmul.
        mask = mask.unsqueeze(-1)                                       # (b, N, 1)
        write_key   = write_key   * mask
        write_value = write_value * mask
        # (b, kv_dim, N) @ (b, N, kv_dim) -> (b, kv_dim, kv_dim)
        return torch.matmul(write_key.transpose(-1, -2), write_value)


class FullTokensNormalizedv3MetisHyperMemory(LinearLastMetisHyperMemory):
    """Memory update using all real tokens with v3-style normalization.

    Every non-padding token contributes one rank-1 outer product:

        h_normed   = input_layernorm(h)             # (b, s, hidden)
        write_key  = W_k(h_normed)                  # (b, s, kv_dim)
        write_val  = W_v(h_normed)                  # (b, s, kv_dim)
        delta      = write_key.T @ write_val / (L * sqrt(D))

    This matches ``StrideNormalizedv3MetisHyperMemory``'s normalization while
    selecting the full real-token sequence instead of stride-sampled tokens.
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,                 # (b, s, hidden_size)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                           # (b, kv_dim, kv_dim)
        h = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        write_key   = self.W_k(h)                                # (b, s, kv_dim)
        write_value = self.W_v(h)                                # (b, s, kv_dim)

        if attention_mask is not None:
            # Broadcast mask over hidden dim so pad positions contribute 0.
            mask = attention_mask.unsqueeze(-1).to(write_key.dtype)  # (b, s, 1)
            L_prime = attention_mask.sum(dim=1).clamp(min=1)         # (b,)
            write_key   = write_key   * mask
            write_value = write_value * mask
        else:
            L_prime = torch.full(
                (raw_info.size(0),),
                raw_info.size(1),
                dtype=write_key.dtype,
                device=raw_info.device,
            ).clamp(min=1)

        # (b, kv_dim, s) @ (b, s, kv_dim) -> (b, kv_dim, kv_dim)
        delta = torch.matmul(write_key.transpose(-1, -2), write_value)
        scale = L_prime.to(delta.dtype) * (self.kv_dim ** 0.5)
        return delta / scale.view(-1, 1, 1)


class KeyNormTokenAggMetisHyperMemory(LinearLastMetisHyperMemory):
    """Shared write path for token-aggregation experiments.

    Subclasses choose or pool hidden states into ``(h_tokens, mask)``.  This
    base class then applies the same key-normalized DeltaNet write protocol as
    ``FullTokensKeyNormMetisHyperMemory``:

        k = normalize(W_k(input_layernorm(h))) / sqrt(D)
        v = W_v(input_layernorm(h))
        state = mean_t(k_t^T @ v_t)
        key_state = mean_t(k_t)
    """

    def _delta_from_normed_tokens(
        self,
        h_normed: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        write_key = F.normalize(self.W_k(h_normed), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h_normed)

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).to(write_key.dtype)
            lengths = mask.sum(dim=1).clamp(min=1).to(write_key.dtype)
            write_key = write_key * mask_exp
            write_value = write_value * mask_exp
        else:
            lengths = torch.full(
                (h_normed.size(0),),
                h_normed.size(1),
                dtype=write_key.dtype,
                device=h_normed.device,
            ).clamp(min=1)

        delta_state = torch.matmul(write_key.transpose(-1, -2), write_value)
        ones = torch.ones(
            write_key.size(0),
            write_key.size(1),
            1,
            device=write_key.device,
            dtype=write_key.dtype,
        )
        delta_key_state = torch.matmul(write_key.transpose(-1, -2), ones)
        scale = lengths.view(-1, 1, 1)
        return delta_state / scale, delta_key_state / scale

    def _delta_from_raw_tokens(
        self,
        h_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(h_tokens)
        return self._delta_from_normed_tokens(h_normed, mask)

    def _write_keynorm_update(
        self,
        raw_info: torch.Tensor,
        local_memory,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        delta_state, delta_key_state = self.get_new_info_for_local_memory(raw_info, attention_mask)
        if getattr(local_memory, "key_state", None) is None:
            new_state = self.update_ratio * delta_state
            new_key_state = self.update_ratio * delta_key_state
        else:
            new_state = (1.0 - self.update_ratio) * local_memory.state + self.update_ratio * delta_state
            new_key_state = (
                (1.0 - self.update_ratio) * local_memory.key_state
                + self.update_ratio * delta_key_state
            )
        local_memory.write(new_state, new_key_state)

    def update_local_memory(
        self,
        raw_info: torch.Tensor,
        local_memory,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self._write_keynorm_update(raw_info, local_memory, attention_mask)


class MeanPoolKeyNormMetisHyperMemory(KeyNormTokenAggMetisHyperMemory):
    """Mean-pool all real hidden states into one write token."""

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(raw_info.dtype)
            lengths = attention_mask.sum(dim=1).clamp(min=1).to(raw_info.dtype)
            h_pool = (raw_info * mask).sum(dim=1, keepdim=True) / lengths.view(-1, 1, 1)
        else:
            h_pool = raw_info.mean(dim=1, keepdim=True)
        return self._delta_from_raw_tokens(h_pool)


class StridePoolKeyNormMetisHyperMemory(KeyNormTokenAggMetisHyperMemory):
    """Mean-pool every stride-sized chunk into one write token per chunk."""

    DEFAULT_STRIDE: int = 8

    def __init__(self, config) -> None:
        super().__init__(config)
        self.stride = int(config.memory_configs.get("stride_interval", self.DEFAULT_STRIDE))
        if self.stride <= 0:
            raise ValueError(f"stride_interval must be > 0, got {self.stride}")

    def _pool_stride_windows(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long().tolist()
        else:
            lengths = [S] * b

        per_sample: list[torch.Tensor] = []
        masks: list[list[float]] = []
        max_chunks = 1
        for bi, L in enumerate(lengths):
            L = max(int(L), 1)
            chunks = []
            for start in range(0, L, self.stride):
                end = min(start + self.stride, L)
                chunks.append(hidden_states[bi, start:end].mean(dim=0))
            sample = torch.stack(chunks, dim=0)
            per_sample.append(sample)
            max_chunks = max(max_chunks, sample.size(0))

        padded = []
        for sample in per_sample:
            pad_n = max_chunks - sample.size(0)
            if pad_n > 0:
                pad = torch.zeros(pad_n, hidden_size, device=device, dtype=dtype)
                sample = torch.cat([sample, pad], dim=0)
            padded.append(sample)
            masks.append([1.0] * (sample.size(0) - pad_n) + [0.0] * pad_n)

        return torch.stack(padded, dim=0), torch.tensor(masks, device=device, dtype=dtype)

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_pool, mask = self._pool_stride_windows(raw_info, attention_mask)
        return self._delta_from_raw_tokens(h_pool, mask)


class AttentionPoolKeyNormMetisHyperMemory(KeyNormTokenAggMetisHyperMemory):
    """Learn a global attention pooling query and write one pooled token."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.pool_score = nn.Linear(self.text_cfg.hidden_size, 1, bias=False)
        self.pool_temperature = float(config.memory_configs.get("pool_temperature", 1.0))
        if self.pool_temperature <= 0:
            raise ValueError(f"pool_temperature must be > 0, got {self.pool_temperature}")

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        scores = self.pool_score(h_normed).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores / self.pool_temperature, dim=1).unsqueeze(1)
        h_pool = torch.matmul(weights, h_normed)
        return self._delta_from_normed_tokens(h_pool)


class WindowAttentionPoolKeyNormMetisHyperMemory(AttentionPoolKeyNormMetisHyperMemory):
    """Soft-select one pooled write token per stride-sized window.

    This is the differentiable replacement for hard top-k token selection used
    by the token-aggregation experiments.  For stride ``R`` it writes roughly
    ``ceil(L / R)`` tokens, matching stride/top-k compression, but the scorer
    receives gradients from every real token in each window during both
    training and inference.
    """

    DEFAULT_STRIDE: int = 8

    def __init__(self, config) -> None:
        super().__init__(config)
        self.stride = int(config.memory_configs.get("stride_interval", self.DEFAULT_STRIDE))
        if self.stride <= 0:
            raise ValueError(f"stride_interval must be > 0, got {self.stride}")

    def _pool_attention_windows(
        self,
        h_normed: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, hidden_size = h_normed.shape
        if attention_mask is None:
            attention_mask = torch.ones(b, S, device=h_normed.device, dtype=torch.long)
        pad_n = (-S) % self.stride
        if pad_n > 0:
            h_normed = F.pad(h_normed, (0, 0, 0, pad_n))
            attention_mask = F.pad(attention_mask, (0, pad_n))

        W = h_normed.size(1) // self.stride
        h_win = h_normed.view(b, W, self.stride, hidden_size)
        scores = self.pool_score(h_win).squeeze(-1)
        mask_win = attention_mask.view(b, W, self.stride).bool()

        valid_window = mask_win.any(dim=2)
        scores = scores.masked_fill(~mask_win, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores / self.pool_temperature, dim=2)
        weights = weights.masked_fill(~mask_win, 0.0)

        h_pool = (weights.unsqueeze(-1).to(h_win.dtype) * h_win).sum(dim=2)
        return h_pool, valid_window.to(h_normed.dtype)

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        h_pool, mask = self._pool_attention_windows(h_normed, attention_mask)
        return self._delta_from_normed_tokens(h_pool, mask)


class TopKKeyNormMetisHyperMemory(KeyNormTokenAggMetisHyperMemory):
    """Learn token scores, then write the top ceil(L / stride_interval) tokens."""

    DEFAULT_STRIDE: int = 8

    def __init__(self, config) -> None:
        super().__init__(config)
        self.stride = int(config.memory_configs.get("stride_interval", self.DEFAULT_STRIDE))
        if self.stride <= 0:
            raise ValueError(f"stride_interval must be > 0, got {self.stride}")
        self.pool_score = nn.Linear(self.text_cfg.hidden_size, 1, bias=False)
        self.pool_temperature = float(config.memory_configs.get("pool_temperature", 1.0))
        if self.pool_temperature <= 0:
            raise ValueError(f"pool_temperature must be > 0, got {self.pool_temperature}")

    def _select_topk(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, S, hidden_size = raw_info.shape
        device = raw_info.device
        dtype = raw_info.dtype
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        scores = self.pool_score(h_normed).squeeze(-1)
        if attention_mask is not None:
            lengths_t = attention_mask.sum(dim=1).long().clamp(min=1)
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        else:
            lengths_t = torch.full((b,), S, dtype=torch.long, device=device)

        k_per_sample = torch.div(lengths_t + self.stride - 1, self.stride, rounding_mode="floor")
        k_max = int(k_per_sample.max().item())
        selected = []
        selected_scores = []
        masks = []
        for bi in range(b):
            k = int(k_per_sample[bi].item())
            idx = torch.topk(scores[bi], k=k, dim=0).indices.sort().values
            h_sel = raw_info[bi].index_select(0, idx)
            score_sel = scores[bi].index_select(0, idx)
            pad_n = k_max - k
            if pad_n > 0:
                h_pad = torch.zeros(pad_n, hidden_size, device=device, dtype=dtype)
                s_pad = torch.zeros(pad_n, device=device, dtype=scores.dtype)
                h_sel = torch.cat([h_sel, h_pad], dim=0)
                score_sel = torch.cat([score_sel, s_pad], dim=0)
            selected.append(h_sel)
            selected_scores.append(score_sel)
            masks.append([1.0] * k + [0.0] * pad_n)

        return (
            torch.stack(selected, dim=0),
            torch.tensor(masks, device=device, dtype=dtype),
            torch.stack(selected_scores, dim=0),
        )

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_sel, mask, _scores = self._select_topk(raw_info, attention_mask)
        return self._delta_from_raw_tokens(h_sel, mask)


class SoftTopKKeyNormMetisHyperMemory(TopKKeyNormMetisHyperMemory):
    """All-token soft select used for both training and inference.

    This is the no-hard-selection counterpart of top-k writes:

        soft = softmax(scores / tau) * K
        state = sum_i soft_i * outer(k_i, v_i) / K

    Forward and backward are both soft.  The total gate mass is K, so after the
    final divide-by-K this is a convex weighted sum of per-token outer products
    and stays on the same scale as K hard selected tokens averaged by K.
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, _hidden_size = raw_info.shape
        device = raw_info.device
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        scores = self.pool_score(h_normed).squeeze(-1)

        if attention_mask is not None:
            lengths_t = attention_mask.sum(dim=1).long().clamp(min=1)
            valid_mask = attention_mask.bool()
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        else:
            lengths_t = torch.full((b,), S, dtype=torch.long, device=device)
            valid_mask = torch.ones(b, S, device=device, dtype=torch.bool)

        k_per_sample = torch.div(lengths_t + self.stride - 1, self.stride, rounding_mode="floor")
        gate = torch.softmax(scores / self.pool_temperature, dim=1)
        gate = gate.masked_fill(~valid_mask, 0.0)
        gate = gate * k_per_sample.to(gate.dtype).unsqueeze(1)

        write_key = F.normalize(self.W_k(h_normed), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h_normed)
        gate_exp = gate.unsqueeze(-1).to(write_key.dtype)
        write_key = write_key * gate_exp

        delta_state = torch.matmul(write_key.transpose(-1, -2), write_value)
        ones = torch.ones(b, S, 1, device=device, dtype=write_key.dtype)
        delta_key_state = torch.matmul(write_key.transpose(-1, -2), ones)
        scale = k_per_sample.clamp(min=1).to(write_key.dtype).view(-1, 1, 1)
        return delta_state / scale, delta_key_state / scale


class StraightThroughTopKKeyNormMetisHyperMemory(TopKKeyNormMetisHyperMemory):
    """Hard top-k forward with softmax surrogate gradients for the scorer."""

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, _hidden_size = raw_info.shape
        device = raw_info.device
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        scores = self.pool_score(h_normed).squeeze(-1)

        if attention_mask is not None:
            lengths_t = attention_mask.sum(dim=1).long().clamp(min=1)
            valid_mask = attention_mask.bool()
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        else:
            lengths_t = torch.full((b,), S, dtype=torch.long, device=device)
            valid_mask = torch.ones(b, S, device=device, dtype=torch.bool)

        k_per_sample = torch.div(lengths_t + self.stride - 1, self.stride, rounding_mode="floor")
        soft = torch.softmax(scores / self.pool_temperature, dim=1) * k_per_sample.to(scores.dtype).unsqueeze(1)
        soft = soft.masked_fill(~valid_mask, 0.0)

        hard = torch.zeros_like(scores)
        for bi in range(b):
            k = int(k_per_sample[bi].item())
            idx = torch.topk(scores[bi], k=k, dim=0).indices
            hard[bi].scatter_(0, idx, 1.0)

        gate = hard.detach() - soft.detach() + soft
        write_key = F.normalize(self.W_k(h_normed), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h_normed)
        gate_exp = gate.unsqueeze(-1).to(write_key.dtype)
        write_key = write_key * gate_exp

        delta_state = torch.matmul(write_key.transpose(-1, -2), write_value)
        ones = torch.ones(b, S, 1, device=device, dtype=write_key.dtype)
        delta_key_state = torch.matmul(write_key.transpose(-1, -2), ones)
        scale = k_per_sample.clamp(min=1).to(write_key.dtype).view(-1, 1, 1)
        return delta_state / scale, delta_key_state / scale


class GumbelTopKKeyNormMetisHyperMemory(TopKKeyNormMetisHyperMemory):
    """Continuous Gumbel-TopK approximation that writes K soft-selected tokens."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.gumbel_topk_noise = bool(config.memory_configs.get("gumbel_topk_noise", True))
        self.gumbel_eps = float(config.memory_configs.get("gumbel_eps", 1e-6))

    def _sample_gumbel(self, scores: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(scores).clamp_(self.gumbel_eps, 1.0 - self.gumbel_eps)
        return -torch.log(-torch.log(uniform))

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, _hidden_size = raw_info.shape
        device = raw_info.device
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        scores = self.pool_score(h_normed).squeeze(-1)

        if attention_mask is not None:
            lengths_t = attention_mask.sum(dim=1).long().clamp(min=1)
            valid_mask = attention_mask.bool()
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        else:
            lengths_t = torch.full((b,), S, dtype=torch.long, device=device)
            valid_mask = torch.ones(b, S, device=device, dtype=torch.bool)

        k_per_sample = torch.div(lengths_t + self.stride - 1, self.stride, rounding_mode="floor")
        k_max = int(k_per_sample.max().item())
        logits = scores
        if self.training and self.gumbel_topk_noise:
            logits = logits + self._sample_gumbel(scores)

        remaining = valid_mask.to(scores.dtype)
        selections = []
        for _ in range(k_max):
            masked_logits = logits + torch.log(remaining.clamp(min=self.gumbel_eps))
            weights = torch.softmax(masked_logits / self.pool_temperature, dim=1)
            weights = weights.masked_fill(~valid_mask, 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=self.gumbel_eps)
            selections.append(weights)
            remaining = remaining * (1.0 - weights).clamp(min=0.0)

        selection = torch.stack(selections, dim=1)
        row_mask = (
            torch.arange(k_max, device=device).unsqueeze(0)
            < k_per_sample.unsqueeze(1)
        ).to(h_normed.dtype)
        h_pool = torch.matmul(selection.to(h_normed.dtype), h_normed)
        return self._delta_from_normed_tokens(h_pool, row_mask)


class AlphaTopPKeyNormMetisHyperMemory(TopKKeyNormMetisHyperMemory):
    """Adaptive top-p/nucleus token selection with selected soft weights.

    Select the smallest set whose scorer probability mass reaches
    ``alpha_top_p``.  The selected tokens are written as a convex weighted sum
    of per-token outer products:

        weights_i = p_i / sum_{j in S_alpha} p_j
        state = sum_{i in S_alpha} weights_i * outer(k_i, v_i)
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.alpha_top_p = float(config.memory_configs.get("alpha_top_p", 0.9))
        if not 0.0 < self.alpha_top_p <= 1.0:
            raise ValueError(f"alpha_top_p must be in (0, 1], got {self.alpha_top_p}")
        self.alpha_min_tokens = int(config.memory_configs.get("alpha_min_tokens", 1))
        if self.alpha_min_tokens <= 0:
            raise ValueError(f"alpha_min_tokens must be > 0, got {self.alpha_min_tokens}")
        self.alpha_max_tokens = int(config.memory_configs.get("alpha_max_tokens", 0))
        if self.alpha_max_tokens < 0:
            raise ValueError(f"alpha_max_tokens must be >= 0, got {self.alpha_max_tokens}")
        self.alpha_max_fraction = float(config.memory_configs.get("alpha_max_fraction", 0.0))
        if not 0.0 <= self.alpha_max_fraction <= 1.0:
            raise ValueError(f"alpha_max_fraction must be in [0, 1], got {self.alpha_max_fraction}")
        self.last_alpha_stats: dict[str, float] = {}

    def _alpha_top_p_mask(
        self,
        probs: torch.Tensor,
        valid_mask: torch.Tensor,
        lengths_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, S = probs.shape
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=1)
        cum = sorted_probs.cumsum(dim=1)
        k_raw = (cum <= self.alpha_top_p).sum(dim=1) + 1
        k_raw = torch.minimum(k_raw, lengths_t)

        k_min = torch.minimum(
            torch.full_like(lengths_t, self.alpha_min_tokens),
            lengths_t,
        )
        k_max = lengths_t.clone()
        if self.alpha_max_fraction > 0.0:
            frac_cap = torch.ceil(lengths_t.to(probs.dtype) * self.alpha_max_fraction).long()
            k_max = torch.minimum(k_max, frac_cap.clamp(min=1))
        if self.alpha_max_tokens > 0:
            fixed_cap = torch.full_like(lengths_t, self.alpha_max_tokens)
            k_max = torch.minimum(k_max, fixed_cap.clamp(min=1))
        k_max = torch.maximum(k_max, k_min)
        k_alpha = torch.minimum(torch.maximum(k_raw, k_min), k_max)

        rank = torch.arange(S, device=probs.device).unsqueeze(0)
        keep_sorted = rank < k_alpha.unsqueeze(1)
        hard = torch.zeros_like(probs)
        hard.scatter_(1, sorted_idx, keep_sorted.to(probs.dtype))
        hard = hard.masked_fill(~valid_mask, 0.0)

        selected_mass = (probs * hard).sum(dim=1, keepdim=True).clamp(min=1e-6)
        return hard, k_alpha, selected_mass

    def _record_alpha_stats(
        self,
        probs: torch.Tensor,
        hard: torch.Tensor,
        k_alpha: torch.Tensor,
        selected_mass: torch.Tensor,
        lengths_t: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            probs_f = probs.detach().float()
            k_f = k_alpha.detach().float()
            lengths_f = lengths_t.detach().float().clamp(min=1)
            entropy = -(probs_f * probs_f.clamp(min=1e-12).log()).sum(dim=1)
            self.last_alpha_stats = {
                "k_mean": float(k_f.mean().item()),
                "k_min": float(k_f.min().item()),
                "k_max": float(k_f.max().item()),
                "k_ratio": float((k_f / lengths_f).mean().item()),
                "score_entropy": float(entropy.mean().item()),
                "p_max": float(probs_f.max(dim=1).values.mean().item()),
                "selected_mass": float(selected_mass.detach().float().mean().item()),
                "alpha": self.alpha_top_p,
            }

    def _alpha_weights(
        self,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        lengths_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(scores / self.pool_temperature, dim=1)
        probs = probs.masked_fill(~valid_mask, 0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        hard, k_alpha, selected_mass = self._alpha_top_p_mask(probs, valid_mask, lengths_t)
        self._record_alpha_stats(probs, hard, k_alpha, selected_mass, lengths_t)
        weights = probs * hard / selected_mass
        return weights, probs

    def _delta_from_weights(
        self,
        h_normed: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        write_key = F.normalize(self.W_k(h_normed), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h_normed)
        weight_exp = weights.unsqueeze(-1).to(write_key.dtype)
        write_key = write_key * weight_exp
        delta_state = torch.matmul(write_key.transpose(-1, -2), write_value)
        ones = torch.ones(
            write_key.size(0),
            write_key.size(1),
            1,
            device=write_key.device,
            dtype=write_key.dtype,
        )
        delta_key_state = torch.matmul(write_key.transpose(-1, -2), ones)
        return delta_state, delta_key_state

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, _hidden_size = raw_info.shape
        device = raw_info.device
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        scores = self.pool_score(h_normed).squeeze(-1)
        if attention_mask is not None:
            lengths_t = attention_mask.sum(dim=1).long().clamp(min=1)
            valid_mask = attention_mask.bool()
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        else:
            lengths_t = torch.full((b,), S, dtype=torch.long, device=device)
            valid_mask = torch.ones(b, S, device=device, dtype=torch.bool)

        weights, _probs = self._alpha_weights(scores, valid_mask, lengths_t)
        return self._delta_from_weights(h_normed, weights)


class StraightThroughAlphaTopPKeyNormMetisHyperMemory(AlphaTopPKeyNormMetisHyperMemory):
    """Alpha top-p forward with full-softmax surrogate gradients."""

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, _hidden_size = raw_info.shape
        device = raw_info.device
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        scores = self.pool_score(h_normed).squeeze(-1)
        if attention_mask is not None:
            lengths_t = attention_mask.sum(dim=1).long().clamp(min=1)
            valid_mask = attention_mask.bool()
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        else:
            lengths_t = torch.full((b,), S, dtype=torch.long, device=device)
            valid_mask = torch.ones(b, S, device=device, dtype=torch.bool)

        hard_weights, probs = self._alpha_weights(scores, valid_mask, lengths_t)
        soft_weights = probs.masked_fill(~valid_mask, 0.0)
        soft_weights = soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weights = hard_weights.detach() - soft_weights.detach() + soft_weights
        return self._delta_from_weights(h_normed, weights)


class GatedDeltaRuleMixin:
    """Mixin implementing the gated delta rule memory recurrence.

    The paper formula is written for column-vector reads:

        S_t = S_{t-1}(alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T

    Metis stores row-vector memories read as ``q @ M``.  The equivalent
    single-token update is:

        M_t = alpha_t (I - beta_t k_t k_t^T) M_{t-1} + beta_t k_t v_t^T

    For a selected token set, this implementation applies the batched parallel
    approximation ``sum_t beta_t k_t k_t^T`` / ``sum_t beta_t k_t v_t^T`` in one
    write.  The same erase/write rule is applied to ``key_state`` when the
    paired local memory keeps one for key-normalized reads.
    """

    @staticmethod
    def _logit_clamped(value: float) -> float:
        eps = 1e-4
        p = min(max(float(value), eps), 1.0 - eps)
        return math.log(p / (1.0 - p))

    def _init_gated_delta_rule(self) -> None:
        hidden_size = self.text_cfg.hidden_size
        self.gated_delta_alpha = nn.Linear(hidden_size, 1, bias=True)
        self.gated_delta_beta = nn.Linear(hidden_size, 1, bias=True)

        nn.init.zeros_(self.gated_delta_alpha.weight)
        nn.init.zeros_(self.gated_delta_beta.weight)

        alpha_init = self.config.memory_configs.get("gated_delta_alpha_init", 1.0)
        beta_init = self.config.memory_configs.get("gated_delta_beta_init", 1.0)
        nn.init.constant_(self.gated_delta_alpha.bias, self._logit_clamped(alpha_init))
        nn.init.constant_(self.gated_delta_beta.bias, self._logit_clamped(beta_init))

    def _alpha_top_p_normed_weights(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None,
        straight_through: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, _hidden_size = raw_info.shape
        device = raw_info.device
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        scores = self.pool_score(h_normed).squeeze(-1)
        if attention_mask is not None:
            lengths_t = attention_mask.sum(dim=1).long().clamp(min=1)
            valid_mask = attention_mask.bool()
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        else:
            lengths_t = torch.full((b,), S, dtype=torch.long, device=device)
            valid_mask = torch.ones(b, S, device=device, dtype=torch.bool)

        hard_weights, probs = self._alpha_weights(scores, valid_mask, lengths_t)
        if not straight_through:
            return h_normed, hard_weights

        soft_weights = probs.masked_fill(~valid_mask, 0.0)
        soft_weights = soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weights = hard_weights.detach() - soft_weights.detach() + soft_weights
        return h_normed, weights

    def _apply_gated_delta_rule_update(
        self,
        h_normed: torch.Tensor,
        weights: torch.Tensor,
        local_memory,
    ) -> None:
        write_key = F.normalize(self.W_k(h_normed), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h_normed)

        weights = weights.to(write_key.dtype)
        weight_mass = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        alpha_gate = torch.sigmoid(self.gated_delta_alpha(h_normed).squeeze(-1))
        beta_gate = torch.sigmoid(self.gated_delta_beta(h_normed).squeeze(-1))
        alpha = (weights * alpha_gate).sum(dim=1) / weight_mass.squeeze(1)
        beta = weights * (self.update_ratio * beta_gate)
        beta_exp = beta.unsqueeze(-1)

        bsz = write_key.size(0)
        state = local_memory.state
        if state is None:
            state = torch.zeros(
                bsz, self.kv_dim, self.kv_dim,
                device=write_key.device,
                dtype=write_key.dtype,
            )

        key_state = getattr(local_memory, "key_state", None)

        # M_t = alpha * (M - K^T beta (K M)) + K^T beta V
        key_memory = torch.matmul(write_key, state)
        erase_state = torch.matmul(write_key.transpose(-1, -2), beta_exp * key_memory)
        add_state = torch.matmul(write_key.transpose(-1, -2), beta_exp * write_value)
        new_state = alpha.view(bsz, 1, 1) * (state - erase_state) + add_state

        has_key_state = hasattr(local_memory, "key_state")
        if key_state is None and has_key_state:
            key_state = torch.zeros(
                bsz, self.kv_dim, 1,
                device=write_key.device,
                dtype=write_key.dtype,
            )

        if key_state is not None:
            key_memory_mass = torch.matmul(write_key, key_state)
            erase_key_state = torch.matmul(
                write_key.transpose(-1, -2),
                beta_exp * key_memory_mass,
            )
            add_key_state = torch.matmul(write_key.transpose(-1, -2), beta_exp)
            new_key_state = (
                alpha.view(bsz, 1, 1) * (key_state - erase_key_state)
                + add_key_state
            )
            local_memory.write(new_state, new_key_state)
        else:
            local_memory.write(new_state)


class AlphaTopPGatedDeltaRuleMetisHyperMemory(
    GatedDeltaRuleMixin,
    AlphaTopPKeyNormMetisHyperMemory,
):
    """AlphaTopP token selection with gated-delta local-memory writes."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._init_gated_delta_rule()

    def update_local_memory(
        self,
        raw_info: torch.Tensor,
        local_memory,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        h_normed, weights = self._alpha_top_p_normed_weights(
            raw_info, attention_mask, straight_through=False,
        )
        self._apply_gated_delta_rule_update(h_normed, weights, local_memory)


class StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory(
    AlphaTopPGatedDeltaRuleMetisHyperMemory,
):
    """Straight-through AlphaTopP selection with gated-delta writes."""

    def update_local_memory(
        self,
        raw_info: torch.Tensor,
        local_memory,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        h_normed, weights = self._alpha_top_p_normed_weights(
            raw_info, attention_mask, straight_through=True,
        )
        self._apply_gated_delta_rule_update(h_normed, weights, local_memory)


class LastTokenGatedDeltaRuleMetisHyperMemory(
    GatedDeltaRuleMixin,
    LinearLastMetisHyperMemory,
):
    """Last-real-token selection with gated-delta memory updates."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._init_gated_delta_rule()

    def update_local_memory(
        self,
        raw_info: torch.Tensor,
        local_memory,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        batch_size = raw_info.size(0)
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1).long().clamp(min=1) - 1
        else:
            last_idx = torch.full(
                (batch_size,),
                raw_info.size(1) - 1,
                dtype=torch.long,
                device=raw_info.device,
            )
        h_last = raw_info[
            torch.arange(batch_size, device=raw_info.device),
            last_idx,
            :,
        ].unsqueeze(1)
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(h_last)
        weights = torch.ones(
            batch_size,
            1,
            device=raw_info.device,
            dtype=h_normed.dtype,
        )
        self._apply_gated_delta_rule_update(h_normed, weights, local_memory)


class WeightedTopKKeyNormMetisHyperMemory(TopKKeyNormMetisHyperMemory):
    """Top-k token write with learned softmax weights over selected tokens."""

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_sel, mask, scores = self._select_topk(raw_info, attention_mask)
        h_normed = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        write_key = F.normalize(self.W_k(h_normed), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h_normed)

        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1).to(write_key.dtype)
        mask_exp = mask.unsqueeze(-1).to(write_key.dtype)
        write_key = write_key * mask_exp
        write_value = write_value * weights * mask_exp

        delta_state = torch.matmul(write_key.transpose(-1, -2), write_value)
        delta_key_state = torch.matmul(write_key.transpose(-1, -2), weights * mask_exp)
        return delta_state, delta_key_state


class Conv1dPoolKeyNormMetisHyperMemory(StridePoolKeyNormMetisHyperMemory):
    """Depthwise conv1d pooling with kernel=stride=stride_interval."""

    def __init__(self, config) -> None:
        super().__init__(config)
        hidden_size = self.text_cfg.hidden_size
        self.pool_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.stride,
            stride=self.stride,
            groups=hidden_size,
            bias=False,
        )
        nn.init.constant_(self.pool_conv.weight, 1.0 / self.stride)

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, S, hidden_size = raw_info.shape
        pad_n = (-S) % self.stride
        if pad_n > 0:
            raw_info = F.pad(raw_info, (0, 0, 0, pad_n))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_n))

        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).to(raw_info.dtype)
            raw_info = raw_info * mask_exp
            denom = F.avg_pool1d(
                attention_mask.unsqueeze(1).to(raw_info.dtype),
                kernel_size=self.stride,
                stride=self.stride,
                count_include_pad=False,
            ).squeeze(1) * self.stride
        else:
            denom = torch.full(
                (b, raw_info.size(1) // self.stride),
                self.stride,
                device=raw_info.device,
                dtype=raw_info.dtype,
            )

        h_pool = self.pool_conv(raw_info.transpose(1, 2)).transpose(1, 2)
        h_pool = h_pool * (self.stride / denom.clamp(min=1).unsqueeze(-1))
        mask = (denom > 0).to(raw_info.dtype)
        return self._delta_from_raw_tokens(h_pool, mask)


class MixedKeyNormMetisHyperMemory(StridePoolKeyNormMetisHyperMemory):
    """Stride-window pooled writes mixed with one global mean-pooled write token.

    This uses the same stride-window pooling as ``StridePoolKeyNormMetisHyperMemory``
    and appends one all-sequence mean token:

        tokens = [mean(h[0:K]), mean(h[K:2K]), ..., mean(h[0:L])]
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_stride, stride_mask = self._pool_stride_windows(raw_info, attention_mask)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(raw_info.dtype)
            lengths = attention_mask.sum(dim=1).clamp(min=1).to(raw_info.dtype)
            h_mean = (raw_info * mask).sum(dim=1, keepdim=True) / lengths.view(-1, 1, 1)
        else:
            h_mean = raw_info.mean(dim=1, keepdim=True)

        h_mix = torch.cat([h_stride, h_mean], dim=1)
        mean_mask = torch.ones(
            stride_mask.size(0),
            1,
            device=stride_mask.device,
            dtype=stride_mask.dtype,
        )
        mix_mask = torch.cat([stride_mask, mean_mask], dim=1)
        return self._delta_from_raw_tokens(h_mix, mix_mask)


class FullTokensKeyNormMetisHyperMemory(LinearLastMetisHyperMemory):
    """Full-token write path for key-normalized DeltaNet memory.

    This mirrors the recent metis_modular normalization scheme while keeping it
    opt-in as a separate dev_beta class:

        h_normed    = input_layernorm(h)
        write_key   = normalize(W_k(h_normed)) / sqrt(D)
        write_value = W_v(h_normed)
        state       = mean_t(write_key_t^T @ write_value_t)
        key_state   = mean_t(write_key_t)

    ``NormalizedDeltaNetMetisLocalMemory`` uses ``key_state`` at read time to
    divide memory outputs by ``q @ key_state + 1``.
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone_decoder.raw_decoder.input_layernorm(raw_info)
        write_key = F.normalize(self.W_k(h), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(write_key.dtype)
            lengths = attention_mask.sum(dim=1).clamp(min=1).to(write_key.dtype)
            write_key = write_key * mask
            write_value = write_value * mask
        else:
            lengths = torch.full(
                (raw_info.size(0),),
                raw_info.size(1),
                dtype=write_key.dtype,
                device=raw_info.device,
            ).clamp(min=1)

        delta_state = torch.matmul(write_key.transpose(-1, -2), write_value)
        ones = torch.ones(
            write_key.size(0),
            write_key.size(1),
            1,
            device=write_key.device,
            dtype=write_key.dtype,
        )
        delta_key_state = torch.matmul(write_key.transpose(-1, -2), ones)
        scale = lengths.view(-1, 1, 1)
        return delta_state / scale, delta_key_state / scale

    def update_local_memory(
        self,
        raw_info: torch.Tensor,
        local_memory,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        delta_state, delta_key_state = self.get_new_info_for_local_memory(raw_info, attention_mask)
        if getattr(local_memory, "key_state", None) is None:
            new_state = self.update_ratio * delta_state
            new_key_state = self.update_ratio * delta_key_state
        else:
            new_state = (1.0 - self.update_ratio) * local_memory.state + self.update_ratio * delta_state
            new_key_state = (
                (1.0 - self.update_ratio) * local_memory.key_state
                + self.update_ratio * delta_key_state
            )
        local_memory.write(new_state, new_key_state)


class StrideKeyNormMetisHyperMemory(StrideNormalizedMetisHyperMemory):
    """Stride-token write path for key-normalized DeltaNet memory.

    This is the stride-sampled counterpart of
    ``FullTokensKeyNormMetisHyperMemory``:

        h_sel       = input_layernorm(h[stride_indices])
        write_key   = normalize(W_k(h_sel)) / sqrt(D)
        write_value = W_v(h_sel)
        state       = mean_selected(write_key_t^T @ write_value_t)
        key_state   = mean_selected(write_key_t)

    It should be paired with ``NormalizedDeltaNetMetisLocalMemory`` so reads
    can use ``key_state`` for the q @ key_state + 1 normalization factor.
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_sel, mask = self._select_tokens_with_mask(raw_info, attention_mask)
        lengths = mask.sum(dim=1).clamp(min=1).to(raw_info.dtype)

        h_sel = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        write_key = F.normalize(self.W_k(h_sel), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h_sel)

        mask_exp = mask.unsqueeze(-1).to(write_key.dtype)
        write_key = write_key * mask_exp
        write_value = write_value * mask_exp

        delta_state = torch.matmul(write_key.transpose(-1, -2), write_value)
        ones = torch.ones(
            write_key.size(0),
            write_key.size(1),
            1,
            device=write_key.device,
            dtype=write_key.dtype,
        )
        delta_key_state = torch.matmul(write_key.transpose(-1, -2), ones)
        scale = lengths.to(delta_state.dtype).view(-1, 1, 1)
        return delta_state / scale, delta_key_state / scale

    def update_local_memory(
        self,
        raw_info: torch.Tensor,
        local_memory,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        delta_state, delta_key_state = self.get_new_info_for_local_memory(raw_info, attention_mask)
        if getattr(local_memory, "key_state", None) is None:
            new_state = self.update_ratio * delta_state
            new_key_state = self.update_ratio * delta_key_state
        else:
            new_state = (1.0 - self.update_ratio) * local_memory.state + self.update_ratio * delta_state
            new_key_state = (
                (1.0 - self.update_ratio) * local_memory.key_state
                + self.update_ratio * delta_key_state
            )
        local_memory.write(new_state, new_key_state)


class StrideKernelKeyNormMetisHyperMemory(StrideKeyNormMetisHyperMemory):
    """Stride keynorm write path with a kernel feature map on write keys.

    Paired with ``KernelizedDeltaNetMetisLocalMemory``.  The same q/k feature
    map should be used on both sides:

        phi(k)     = kernel(W_k(input_layernorm(h_sel)))
        state      = mean_selected(phi(k)_t^T @ v_t)
        key_state  = mean_selected(phi(k)_t)

    Configurable via ``memory_configs['qk_kernel_type']``:
        - ``elu_plus_one`` (default)
        - ``relu_square``
        - ``softplus``
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.qk_kernel_type = config.memory_configs.get("qk_kernel_type", "elu_plus_one")

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_sel, mask = self._select_tokens_with_mask(raw_info, attention_mask)
        lengths = mask.sum(dim=1).clamp(min=1).to(raw_info.dtype)

        h_sel = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        write_key = _qk_kernel(self.W_k(h_sel), self.qk_kernel_type)
        write_value = self.W_v(h_sel)

        mask_exp = mask.unsqueeze(-1).to(write_key.dtype)
        write_key = write_key * mask_exp
        write_value = write_value * mask_exp

        delta_state = torch.matmul(write_key.transpose(-1, -2), write_value)
        ones = torch.ones(
            write_key.size(0),
            write_key.size(1),
            1,
            device=write_key.device,
            dtype=write_key.dtype,
        )
        delta_key_state = torch.matmul(write_key.transpose(-1, -2), ones)
        scale = lengths.to(delta_state.dtype).view(-1, 1, 1)
        return delta_state / scale, delta_key_state / scale


class StrideL2NormMetisHyperMemory(StrideNormalizedMetisHyperMemory):
    """Stride-token write path with L2-normalized keys scaled by sqrt(D).

    Pair this with ``L2NormalizedDeltaNetMetisLocalMemory``.  Unlike
    ``StrideKeyNormMetisHyperMemory``, this class does not apply a kernel,
    does not produce ``key_state``, and therefore has no key-state denominator
    at read time:

        k      = normalize(W_k(input_layernorm(h_sel))) / sqrt(D)
        q      = normalize(q)  # in the paired local memory
        state  = mean_selected(k_t^T @ v_t)
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_sel, mask = self._select_tokens_with_mask(raw_info, attention_mask)
        lengths = mask.sum(dim=1).clamp(min=1).to(raw_info.dtype)

        h_sel = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        write_key = F.normalize(self.W_k(h_sel), dim=-1) / (self.kv_dim ** 0.5)
        write_value = self.W_v(h_sel)

        mask_exp = mask.unsqueeze(-1).to(write_key.dtype)
        write_key = write_key * mask_exp
        write_value = write_value * mask_exp

        delta = torch.matmul(write_key.transpose(-1, -2), write_value)
        return delta / lengths.to(delta.dtype).view(-1, 1, 1)


class StrideNormalizedv3MetisHyperMemory(StrideNormalizedMetisHyperMemory):
    """Stride-based memory update scaled by 1 / (L' * sqrt(D)).
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,                 # (b, s, hidden_size)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                           # (b, kv_dim, kv_dim)
        h_sel, mask = self._select_tokens_with_mask(raw_info, attention_mask)
        # mask: (b, N_max)  — 1.0 for valid tokens, 0.0 for padding
        L_prime = mask.sum(dim=1).clamp(min=1)              # (b,)  actual token count
        h_sel = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        write_key   = self.W_k(h_sel)                       # (b, N_max, kv_dim)
        write_value = self.W_v(h_sel)                       # (b, N_max, kv_dim)
        # Zero-out padded slots so they contribute nothing to the outer product.
        mask_exp    = mask.unsqueeze(-1)                    # (b, N_max, 1)
        write_key   = write_key   * mask_exp
        write_value = write_value * mask_exp
        # (b, kv_dim, N_max) @ (b, N_max, kv_dim) -> (b, kv_dim, kv_dim)
        delta = torch.matmul(write_key.transpose(-1, -2), write_value)
        # Scale per sample: divide by L' * sqrt(D)
        scale = L_prime * (self.kv_dim ** 0.5)              # (b,)
        scale = scale.view(-1, 1, 1)                        # (b, 1, 1)  broadcast
        return delta / scale

class StrideNormalizedv4MetisHyperMemory(StrideNormalizedMetisHyperMemory):
    """Stride-based memory update scaled by 1 / (L' * sqrt(D)).
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,                 # (b, s, hidden_size)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                           # (b, kv_dim, kv_dim)
        h_sel, mask = self._select_tokens_with_mask(raw_info, attention_mask)
        # mask: (b, N_max)  — 1.0 for valid tokens, 0.0 for padding
        L_prime = mask.sum(dim=1).clamp(min=1)              # (b,)  actual token count
        h_sel = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        write_key   = self.W_k(h_sel)                       # (b, N_max, kv_dim)
        write_value = self.W_v(h_sel)                       # (b, N_max, kv_dim)
        # Zero-out padded slots so they contribute nothing to the outer product.
        mask_exp    = mask.unsqueeze(-1)                    # (b, N_max, 1)
        write_key   = write_key   * mask_exp
        write_value = write_value * mask_exp
        # (b, kv_dim, N_max) @ (b, N_max, kv_dim) -> (b, kv_dim, kv_dim)
        delta = torch.matmul(write_key.transpose(-1, -2), write_value)
        # Scale per sample: divide by L' * sqrt(D)
        scale = L_prime * (self.kv_dim)              # (b,)
        scale = scale.view(-1, 1, 1)                        # (b, 1, 1)  broadcast
        return delta / scale


class StrideNormalizedv5MetisHyperMemory(StrideNormalizedMetisHyperMemory):
    """Stride-based memory update: L2-normalize write vectors, then divide by L'.

        delta = F.normalize(W_k H, dim=-1).T  @  F.normalize(W_v H, dim=-1)  /  L'

    Differences vs v3 (which divides raw projections by L' * sqrt(D)):
      - Each token's write_key / write_value is L2-normalised to unit norm before
        the outer product, so every rank-1 contribution has ||·||_F = 1 exactly.
      - Dividing by L' averages the L' unit outer products.
      - Result: ||delta||_F <= 1 always, independent of D, sequence length,
        and weight magnitudes.

    Note: mask is applied AFTER F.normalize so that batch-padding slots (filled
    with a copy of position 0) are first given unit norm and then zeroed out.
    Applying mask before normalize would produce 0/0 for zero vectors.
    """

    def get_new_info_for_local_memory(
        self,
        raw_info: torch.Tensor,                 # (b, s, hidden_size)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                           # (b, kv_dim, kv_dim)
        h_sel, mask = self._select_tokens_with_mask(raw_info, attention_mask)
        # mask: (b, N_max) — 1.0 for valid tokens, 0.0 for batch-padding slots
        L_prime = mask.sum(dim=1).clamp(min=1)              # (b,)  actual token count
        h_sel = self.backbone_decoder.raw_decoder.input_layernorm(h_sel)
        # L2-normalize each token's projection to unit norm along kv_dim axis
        write_key   = F.normalize(self.W_k(h_sel), dim=-1)  # (b, N_max, kv_dim), ‖·‖=1
        write_value = F.normalize(self.W_v(h_sel), dim=-1)  # (b, N_max, kv_dim), ‖·‖=1
        # Zero-out batch-padding slots after normalization to avoid 0/0 issues
        mask_exp    = mask.unsqueeze(-1)                    # (b, N_max, 1)
        write_key   = write_key   * mask_exp
        write_value = write_value * mask_exp
        # (b, kv_dim, N_max) @ (b, N_max, kv_dim) -> (b, kv_dim, kv_dim)
        # ||delta||_F <= L' (sum of L' unit outer products), divide by L' to average
        delta = torch.matmul(write_key.transpose(-1, -2), write_value)
        return delta / L_prime.view(-1, 1, 1)               # (b, 1, 1) broadcast
