"""Lightweight Mixture of Optimized Expert Engines model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise ImportError("MOOE requires PyTorch. Install it with: pip install torch") from exc


@dataclass
class MOOEConfig:
    """Configuration for the native FatTummy MOOE model."""

    hidden_size: int = 256
    intermediate_size: int = 1024
    num_experts: int = 4
    num_layers: int = 4
    vocab_size: int = 32000
    top_k: int = 2
    max_position_embeddings: int = 512

    @property
    def num_hidden_layers(self) -> int:
        """Compatibility alias used by some model tooling."""
        return self.num_layers

    @property
    def num_experts_per_tok(self) -> int:
        """Compatibility alias for top-k routing."""
        return self.top_k

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive.")
        if self.num_experts <= 0 or self.num_layers <= 0:
            raise ValueError("num_experts and num_layers must be positive.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")


class Expert(nn.Module):
    """Two-layer feed-forward expert used inside an MOOE layer."""

    def __init__(self, config: MOOEConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the expert transformation."""
        return self.down(F.gelu(self.up(hidden_states)))


class MOOELayer(nn.Module):
    """Top-k routed mixture-of-experts block with residual-friendly output."""

    def __init__(self, config: MOOEConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(Expert(config) for _ in range(config.num_experts))
        self.top_k = config.top_k
        self.input_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route each token to top-k experts and aggregate weighted outputs.
        Returns a tuple of (output_states, load_balancing_loss)."""
        residual_shape = hidden_states.shape
        normalized = self.input_norm(hidden_states)
        flat = normalized.reshape(-1, residual_shape[-1])

        gate_logits = self.gate(flat)
        routing_probs = torch.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Load balancing auxiliary loss
        num_tokens = flat.size(0)
        num_experts = len(self.experts)
        me = routing_probs.mean(dim=0)
        one_hot_indices = F.one_hot(top_indices, num_classes=num_experts).float()
        ce = one_hot_indices.mean(dim=(0, 1))
        aux_loss = num_experts * torch.sum(me * ce)

        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_positions, route_positions = torch.where(top_indices == expert_index)
            if token_positions.numel() == 0:
                continue
            expert_input = flat[token_positions]
            expert_output = expert(expert_input)
            weights = top_weights[token_positions, route_positions].unsqueeze(-1)
            output.index_add_(0, token_positions, expert_output * weights)

        return output.reshape(residual_shape), aux_loss


class MOOE(nn.Module):
    """CPU-compatible causal language model using MOOE feed-forward layers."""

    def __init__(self, config: Optional[MOOEConfig] = None) -> None:
        super().__init__()
        self.config = config or MOOEConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(MOOELayer(self.config) for _ in range(self.config.num_layers))
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass and optionally compute next-token loss along with load-balancing loss."""
        hidden_states = self.embed_tokens(input_ids)
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        for layer in self.layers:
            layer_output, aux_loss = layer(hidden_states)
            hidden_states = hidden_states + layer_output
            total_aux_loss = total_aux_loss + aux_loss
            
        logits = self.lm_head(self.norm(hidden_states))

        result = {"logits": logits, "aux_loss": total_aux_loss}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = ce_loss + 0.01 * total_aux_loss
        return result

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        """Greedy token generation for native experimentation."""
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated)["logits"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated
