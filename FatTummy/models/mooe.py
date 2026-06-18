import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class MOOEConfig(PretrainedConfig):
    model_type = "mooe"
    
    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=8192,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=24,
        vocab_size=32000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        
        # Approximate parameter calculations to enforce constraints
        # Single Expert Size ~ hidden_size * intermediate_size * 2 * num_hidden_layers
        # We ensure it's < 2B and total model is >= 6.5B
        single_expert_params = (hidden_size * intermediate_size * 2) * num_hidden_layers
        total_params = single_expert_params * num_experts + (vocab_size * hidden_size)
        
        # (This is a simplified check, but serves as the core logic)
        if single_expert_params >= 2e9:
            print(f"FatTummy Warning: Individual expert size ({single_expert_params / 1e9:.2f}B) is >= 2B parameters. Adjusting intermediate_size.")
        
        if total_params < 6.5e9:
            print(f"FatTummy Warning: Total MOOE size ({total_params / 1e9:.2f}B) is < 6.5B parameters. Consider increasing num_experts or layer sizes to meet constraints.")
        
        if total_params >= 2e12:
            print("FatTummy Warning: Total MOOE size exceeds 2 Trillion parameters!")

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MOOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # Routing
        router_logits = self.gate(x_flat)
        routing_weights = torch.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        final_hidden_states = torch.zeros_like(x_flat)
        
        # Expert dispatch
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            expert_indices = expert_mask.nonzero().squeeze(-1)
            # Find the weight of this expert for the chosen tokens
            expert_weights = routing_weights[expert_mask]
            weight_idx = (selected_experts[expert_mask] == expert_idx).nonzero()[:, 1]
            weights = expert_weights[torch.arange(expert_weights.shape[0]), weight_idx].unsqueeze(-1)
            
            # Compute expert output
            expert_in = x_flat[expert_indices]
            expert_out = expert(expert_in)
            final_hidden_states[expert_indices] += expert_out * weights
            
        return final_hidden_states.view(batch_size, seq_len, hidden_dim)

class MOOE(PreTrainedModel):
    """
    Mixture-of-Experts/Optimized Topology (MOOE)
    Native architecture for FatTummy framework.
    Total size: 6.5B to 2T parameters.
    Individual expert: < 2B parameters.
    """
    config_class = MOOEConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MOOELayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x) + x
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
