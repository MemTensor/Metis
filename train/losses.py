"""Task-specific loss functions for Metis memory training.

Three separate interfaces for future customization:
  - loss_fact_recall:    Task 0 — fact recall / reconstruction
  - loss_memory_operation: Task 1 — remember / forget / update / reflection
  - loss_long_term:      Task 2 — distract-augmented long-term memory
"""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor,
                    ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Shared shifted cross-entropy loss for causal LM."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def loss_fact_recall(logits: torch.Tensor, labels: torch.Tensor,
                     ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Task 0: Fact recall — standard CE on query chunk's assistant response."""
    return _causal_lm_loss(logits, labels, ignore_index)


def loss_memory_operation(logits: torch.Tensor, labels: torch.Tensor,
                          operation_type: str | None = None,
                          ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Task 1: Memory operation — operation-aware loss.

    Currently uses the same CE backbone.  Can be extended for:
      - forget: add contrastive penalty for the forgotten fact
      - update: add consistency penalty with prior knowledge
    """
    return _causal_lm_loss(logits, labels, ignore_index)


def loss_long_term(logits: torch.Tensor, labels: torch.Tensor,
                   ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Task 2: Long-term memory — distract-augmented loss.

    Currently uses the same CE backbone.  Can be extended for:
      - higher weight on the final query chunk
      - penalty for distract-induced drift
    """
    return _causal_lm_loss(logits, labels, ignore_index)


# Dispatch table.
_LOSS_FN = {
    0: loss_fact_recall,
    1: loss_memory_operation,
    2: loss_long_term,
}


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, task_id: int,
                 operation: str | None = None,
                 ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Dispatch to the appropriate task-specific loss."""
    fn = _LOSS_FN.get(task_id, loss_fact_recall)
    if task_id == 1 and operation is not None:
        return fn(logits, labels, operation_type=operation, ignore_index=ignore_index)
    return fn(logits, labels, ignore_index=ignore_index)
