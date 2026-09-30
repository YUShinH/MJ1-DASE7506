"""Final model: RoPE + squared ReLU + strictly causal, within-window gated cache.

Baseline GPT remains available for the required reference measurement.
Historical experimental architectures are intentionally excluded from this package.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import GPT

class SquaredReLU(nn.Module):
    """Primer-inspired activation; arxiv:2109.08668."""
    def forward(self, x):
        return F.relu(x).square()



class RotaryBlock(nn.Module):
    """Adjacent-pair Q/K RoPE, Su et al. (2021), arxiv:2104.09864.

    Deterministic position tables, no learned or cross-window memory.
    Shared starter parameters keep their initialization and checkpoint names.
    """
    def __init__(self, original, width, context, theta=10000.):
        super().__init__()
        self.heads = original.heads
        self.norm1, self.norm2 = original.norm1, original.norm2
        self.qkv, self.proj, self.mlp = original.qkv, original.proj, original.mlp
        head_dim = width // self.heads
        if head_dim % 2 or theta <= 0:
            raise ValueError('RoPE needs an even head dimension and positive theta.')
        frequency = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angle = torch.arange(context, dtype=torch.float32)[:, None] * frequency[None, :]
        self.register_buffer('rope_cos', angle.cos()[None, None], persistent=False)
        self.register_buffer('rope_sin', angle.sin()[None, None], persistent=False)

    def rotate(self, x):
        length = x.shape[-2]
        even, odd = x.float()[..., ::2], x.float()[..., 1::2]
        c, s = self.rope_cos[:, :, :length], self.rope_sin[:, :, :length]
        return torch.stack((even*c-odd*s, even*s+odd*c), dim=-1).flatten(-2).to(x.dtype)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))



class RegularizedGPT(GPT):
    """Final trainable backbone; initializer order matches the frozen experiment."""
    def __init__(self, config):
        super().__init__(config)
        if config.get('backbone') != 'relu2' or config.get('position_encoding') != 'rope':
            raise ValueError('This package supports the final RoPE/ReLU2 architecture only.')
        if config.get('norm_type', 'layernorm') != 'layernorm':
            raise ValueError('The final model uses LayerNorm.')
        probability = float(config.get('dropout', 0.0))
        if not 0 <= probability < 1:
            raise ValueError('Invalid dropout.')
        for block in self.blocks:
            block.proj = nn.Sequential(block.proj, nn.Dropout(probability))
            block.mlp.append(nn.Dropout(probability))
            block.mlp[1] = SquaredReLU()
        self.blocks = nn.ModuleList([RotaryBlock(block, config['width'], self.context,
                                   float(config.get('rope_theta', 10000.))) for block in self.blocks])
        del self.pos
        self.cache_theta = float(config.get('cache_theta', 10.0))
        self.cache_window = int(config.get('cache_window', 256))
        if self.cache_theta <= 0 or not 1 <= self.cache_window <= self.context:
            raise ValueError('Invalid cache settings.')
        if float(config.get('cache_lambda', 0.)) != 0:
            raise ValueError('Fixed-lambda cache is not used by the final model.')

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def predict_log_probs(self, ids):
        return F.log_softmax(self.head(self.features(ids)).float(), dim=-1)

    def cache_distribution(self, hidden, ids, theta=None):
        """Cosine neural cache, reconstructed independently for each example/call.

        Adaptation of Grave et al. (2017), arxiv:1612.04426; see also
        salesforce/awd-lstm-lm/pointer.py for the original dot-product mixture.
        This implementation uses cosine similarity and NEVER cross-window state.
        At query t, pair (h_i, ids[i+1]) is visible only when i < t.
        """
        batch, length = ids.shape
        if length < 2:
            return hidden.new_zeros((batch, length, self.config['vocab']), dtype=torch.float32)
        unit = F.normalize(hidden.float(), dim=-1)
        similarities = unit @ unit[:, :-1].transpose(-1, -2)
        query = torch.arange(length, device=ids.device)[:, None]
        key = torch.arange(length-1, device=ids.device)[None, :]
        allowed = (key < query) & (key >= query-self.cache_window)
        scores = similarities * (self.cache_theta if theta is None else theta)
        scores = scores.masked_fill(~allowed, float('-inf'))
        # No history at t=0: avoid all-masked softmax, then zero its weights.
        scores[:, 0, :] = 0
        weights = scores.softmax(-1).masked_fill(~allowed, 0)
        distribution = hidden.new_zeros((batch, length, self.config['vocab']), dtype=torch.float32)
        destinations = ids[:, None, 1:].expand(batch, length, length-1)
        return distribution.scatter_add(-1, destinations, weights)



class GatedCacheGPT(RegularizedGPT):
    """Input-only adaptive mixture, inspired by Pointer Sentinel / efficient kNN-LM.

    The GPT and cosine cache are unchanged. Train this gate with train_gate.py,
    whose loss uses mixed probabilities (the legacy forward returns base logits).
    """
    def __init__(self, config):
        super().__init__(config)
        self.gate_cap = float(config.get('gate_cap', 0.5))
        initial = float(config.get('gate_initial', 0.05))
        if not 0 < initial < self.gate_cap < 1:
            raise ValueError('Require 0 < gate_initial < gate_cap < 1.')
        self.register_buffer('gate_mean', torch.zeros(8))
        self.register_buffer('gate_std', torch.ones(8))
        self.gate_kind = config.get('gate_kind', 'mlp')
        if self.gate_kind == 'constant':
            self.gate = nn.Linear(1, 1, bias=False)
            nn.init.constant_(self.gate.weight, math.log(initial/(self.gate_cap-initial)))
        elif self.gate_kind == 'mlp':
            self.gate = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, math.log(initial/(self.gate_cap-initial)))
        else:
            raise ValueError('Unknown gate_kind.')

    def gate_features(self, logp, cached):
        """Eight causal scalars, independent of targets and other batch examples."""
        probability = logp.exp()
        scale = math.log(self.config['vocab'])
        confidence, predicted = probability.max(-1)
        entropy = -(probability * logp).sum(-1) / scale
        cache_entropy = -(cached * cached.clamp_min(1e-30).log()).sum(-1) / scale
        agreement = (probability * cached).sum(-1)
        support_mass = (probability * (cached > 0)).sum(-1)
        cache_at_prediction = cached.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
        position = torch.arange(logp.shape[1], device=logp.device, dtype=logp.dtype)
        position = (position / max(1, self.context-1)).expand(logp.shape[:2])
        return torch.stack((entropy, confidence, cache_entropy, cached.max(-1).values,
                            agreement, support_mass, cache_at_prediction, position), -1)

    def gate_log_weights(self, features):
        if self.gate_kind == 'constant':
            logits = self.gate(torch.ones_like(features[..., :1])).squeeze(-1)
        else:
            logits = self.gate((features-self.gate_mean)/self.gate_std).squeeze(-1)
        log_cache = math.log(self.gate_cap) + F.logsigmoid(logits)
        return torch.log1p(-log_cache.exp()), log_cache

    def predict_log_probs(self, ids):
        hidden = self.features(ids)
        logp = F.log_softmax(self.head(hidden).float(), dim=-1)
        if ids.shape[1] < 2:
            return logp
        cached = self.cache_distribution(hidden, ids)
        base_weight, cache_weight = self.gate_log_weights(self.gate_features(logp, cached))
        mixed = torch.logaddexp(logp + base_weight.unsqueeze(-1),
                                cached.log() + cache_weight.unsqueeze(-1))
        return torch.cat((logp[:, :1], mixed[:, 1:]), dim=1)



def build_model(config):
    variant = config.get('variant', 'baseline')
    if variant == 'baseline':
        # Existing student checkpoints without a variant still load unchanged.
        return GPT(config)
    if variant == 'residual_dropout':
        return RegularizedGPT(config)
    if variant == 'gated_cache':
        return GatedCacheGPT(config)
    raise ValueError(f'Unknown student variant: {variant}')

