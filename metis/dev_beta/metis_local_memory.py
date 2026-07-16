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

def create_metis_local_memory(config):
    return eval(config.memory_configs['metis_local_memory_type'])(config)

class MetisLocalMemoryBase(nn.Module, ABC):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # Qwen 3.5 has text config, but Qwen 3 does not.
        self.text_cfg = getattr(config.backbone_configs, 'text_config', config.backbone_configs)

    def initialize(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def read(self, query_for_memory):
        raise NotImplementedError

    def write(self, new_info) -> None:
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError


class DeltaNetMetisLocalMemory(MetisLocalMemoryBase):
    """Linear (DeltaNet-style) memory matrix of shape (b, D, D).

    Read:  output = Q_flat @ M,  where Q_flat = (b, s, D)
    Write: M = new_state  (forget + additive update computed by HyperMemory)
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        num_q_heads = self.text_cfg.num_attention_heads
        # If num_key_value_heads is not set, use num_attention_heads (MHA).
        num_kv_heads = getattr(self.text_cfg, "num_key_value_heads", num_q_heads)
        head_dim = getattr(self.text_cfg, "head_dim", self.text_cfg.hidden_size // num_q_heads)

        self.q_dim = num_q_heads * head_dim
        self.kv_dim = self._compute_kv_dim(num_q_heads, num_kv_heads, head_dim)
        self.num_kv_groups = self.q_dim // self.kv_dim

        self._state: torch.Tensor | None = None

    @staticmethod
    def _compute_kv_dim(num_q_heads: int, num_kv_heads: int, head_dim: int) -> int:
        """GQA layout: kv_dim = num_kv_heads * head_dim."""
        return num_kv_heads * head_dim

    def initialize(self) -> None:
        self._state = None

    def reset(self) -> None:
        self._state = None

    def _ensure_ready(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self._state is None or self._state.shape[0] != batch_size:
            self._state = torch.zeros(
                batch_size, self.kv_dim, self.kv_dim, device=device, dtype=dtype,
            )

    def read(self, query_for_memory: torch.Tensor) -> torch.Tensor:
        """Linear memory read: output = Q_flat @ M.

        Args:
            query_for_memory: (b, h, s, d)

        Returns:
            (b, s, D) — memory readout, ready to be fused with attention output.
        """
        bsz, _h, seq_len, _d = query_for_memory.shape
        self._ensure_ready(bsz, query_for_memory.device, query_for_memory.dtype)
        # (b, h, s, d) → (b, s, q_dim)
        q_flat = query_for_memory.transpose(1, 2).reshape(bsz, seq_len, -1)
        if self.num_kv_groups > 1:
            # GQA mode.
            q_2d = q_flat.view(bsz, seq_len * self.num_kv_groups, self.kv_dim)
            out_2d = torch.matmul(q_2d, self._state)
            return out_2d.view(bsz, seq_len, self.q_dim).contiguous()
        else:
            # For MHA mode
            # (b, s, q_dim) @ (b, q_dim, q_dim) → (b, s, q_dim)
            return torch.matmul(q_flat, self._state).contiguous()

    def write(self, new_state: torch.Tensor) -> None:
        self._state = new_state  # no detach here, so gradients flow through W_k / W_v

    @property
    def state(self) -> torch.Tensor | None:
        return self._state

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    def norm(self) -> float:
        return self._state.norm().item() if self._state is not None else 0.0

class MHADeltaNetMetisLocalMemory(DeltaNetMetisLocalMemory):
    """Legacy MHA-style memory: kv_dim = num_q_heads * head_dim (no GQA grouping).

    Memory matrix is (b, q_dim, q_dim) — for Qwen3.5-4B that's 4096×4096.
    Read collapses to a single MHA matmul: (b, s, q_dim) @ (b, q_dim, q_dim).

    Use this for loading checkpoints trained before the GQA refactor
    (e.g. experiments/4.17-* and 4.18-*).
    """

    @staticmethod
    def _compute_kv_dim(num_q_heads: int, num_kv_heads: int, head_dim: int) -> int:
        return num_q_heads * head_dim


class NormalizedDeltaNetMetisLocalMemory(DeltaNetMetisLocalMemory):
    """DeltaNet memory with metis_modular-style key normalization.

    Read path:
        q = normalize(q)
        y = q @ state
        y = y / (q @ key_state + 1)

    The paired hyper-memory class ``FullTokensKeyNormMetisHyperMemory`` writes
    both ``state`` and ``key_state``.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._key_state: torch.Tensor | None = None

    def initialize(self) -> None:
        self._state = None
        self._key_state = None

    def reset(self) -> None:
        self._state = None
        self._key_state = None

    def _ensure_ready(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self._state is None or self._state.shape[0] != batch_size:
            self._state = torch.zeros(
                batch_size, self.kv_dim, self.kv_dim, device=device, dtype=dtype,
            )
            self._key_state = torch.zeros(
                batch_size, self.kv_dim, 1, device=device, dtype=dtype,
            )

    def read(self, query_for_memory: torch.Tensor) -> torch.Tensor:
        bsz, _h, seq_len, _d = query_for_memory.shape
        self._ensure_ready(bsz, query_for_memory.device, query_for_memory.dtype)
        query_for_memory = F.normalize(query_for_memory, dim=-1)
        q_flat = query_for_memory.transpose(1, 2).reshape(bsz, seq_len, -1)

        if self.num_kv_groups > 1:
            q_2d = q_flat.view(bsz, seq_len * self.num_kv_groups, self.kv_dim)
            out_2d = torch.matmul(q_2d, self._state)
            # if self._key_state is not None:
            norm_factor = torch.matmul(q_2d, self._key_state)
            # print(norm_factor[0])
            out_2d = out_2d / (norm_factor + 1.0)
            return out_2d.view(bsz, seq_len, self.q_dim).contiguous()

        out = torch.matmul(q_flat, self._state)
        # if self._key_state is not None:
        norm_factor = torch.matmul(q_flat, self._key_state)
        # print(norm_factor.shape)
        out = out / (norm_factor + 1.0)
        return out.contiguous()

    def write(self, new_state: torch.Tensor, key_state: torch.Tensor) -> None:
        self._state = new_state
        self._key_state = key_state

    @property
    def key_state(self) -> torch.Tensor | None:
        return self._key_state

    @property
    def is_initialized(self) -> bool:
        return self._state is not None and self._key_state is not None


class KernelizedDeltaNetMetisLocalMemory(NormalizedDeltaNetMetisLocalMemory):
    """DeltaNet memory read path with a kernel feature map on queries.

    Pair this with ``StrideKernelKeyNormMetisHyperMemory`` so the same feature
    map is applied to q and k before the key-state normalization:

        phi(q) = kernel(q)
        y      = phi(q) @ state
        y      = y / (phi(q) @ key_state + 1)
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.qk_kernel_type = config.memory_configs.get("qk_kernel_type", "elu_plus_one")

    def read(self, query_for_memory: torch.Tensor) -> torch.Tensor:
        bsz, _h, seq_len, _d = query_for_memory.shape
        self._ensure_ready(bsz, query_for_memory.device, query_for_memory.dtype)
        q_flat = query_for_memory.transpose(1, 2).reshape(bsz, seq_len, -1)

        if self.num_kv_groups > 1:
            q_2d = q_flat.view(bsz, seq_len * self.num_kv_groups, self.kv_dim)
            q_2d = _qk_kernel(q_2d, self.qk_kernel_type)
            out_2d = torch.matmul(q_2d, self._state)
            if self._key_state is not None:
                norm_factor = torch.matmul(q_2d, self._key_state)
                out_2d = out_2d / norm_factor
            return out_2d.view(bsz, seq_len, self.q_dim).contiguous()

        q_flat = _qk_kernel(q_flat, self.qk_kernel_type)
        out = torch.matmul(q_flat, self._state)
        if self._key_state is not None:
            norm_factor = torch.matmul(q_flat, self._key_state)
            out = out / norm_factor
        return out.contiguous()


class L2NormalizedDeltaNetMetisLocalMemory(DeltaNetMetisLocalMemory):
    """DeltaNet memory with L2-normalized queries and no key-state denominator.

    Pair this with ``StrideL2NormMetisHyperMemory``:

        q = normalize(q)
        y = q @ state

    This keeps the read-side q scale controlled by L2 normalization while the
    write-side hyper memory normalizes k.
    """

    def read(self, query_for_memory: torch.Tensor) -> torch.Tensor:
        bsz, _h, seq_len, _d = query_for_memory.shape
        self._ensure_ready(bsz, query_for_memory.device, query_for_memory.dtype)
        query_for_memory = F.normalize(query_for_memory, dim=-1)
        q_flat = query_for_memory.transpose(1, 2).reshape(bsz, seq_len, -1)

        if self.num_kv_groups > 1:
            q_2d = q_flat.view(bsz, seq_len * self.num_kv_groups, self.kv_dim)
            out_2d = torch.matmul(q_2d, self._state)
            return out_2d.view(bsz, seq_len, self.q_dim).contiguous()

        return torch.matmul(q_flat, self._state).contiguous()


class OneStepAblationMetisLocalMemory(MetisLocalMemoryBase):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def initialize(self) -> None:
        self.memory_state = None

    def reset(self) -> None:
        self.initialize()

    def read(self, query_for_memory):
        return self.memory_state

    def write(self, new_info) -> None:
        self.memory_state = new_info
