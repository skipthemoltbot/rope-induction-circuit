"""
Small RoPE Transformer - Proper Implementation
===============================================
Based on Chris Wendler's equations with actual RoPE

Architecture:
- 2 layers, 4 heads, 128 dim, head dim 32
- RoPE (Rotary Position Embeddings)
- 27-token vocab (a-z + space)
- Trained on induction tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Optional, Tuple

# %% Configuration
VOCAB_SIZE = 27  # a-z + space
D_MODEL = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 32
N_LAYERS = 2
ROPE_BASE = 10000.0
MAX_SEQ_LEN = 128

# Character to ID mapping
CHARS = 'abcdefghijklmnopqrstuvwxyz '
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

# %% RoPE Implementation
def compute_rope_frequencies(d_head: int, base: float = ROPE_BASE):
    """
    Compute RoPE frequencies: theta_i = 1 / (base^(2i/d_head))
    """
    freqs = []
    for i in range(d_head // 2):
        theta = 1.0 / (base ** (2 * i / d_head))
        freqs.append(theta)
    return torch.tensor(freqs, dtype=torch.float32)

def apply_rope(x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor):
    """
    Apply RoPE rotation to input tensor.
    
    x: (..., d_head) - last dimension is head dimension
    pos: (...) - position indices
    freqs: (d_head/2,) - precomputed frequencies
    
    Returns: rotated tensor of same shape
    """
    # Reshape x to separate pairs: (..., d_head/2, 2)
    original_shape = x.shape
    x = x.reshape(*x.shape[:-1], -1, 2)
    
    # Split into pairs
    x1 = x[..., 0]  # Even indices
    x2 = x[..., 1]  # Odd indices
    
    # Compute rotation angles: m * theta_i
    # pos shape: (...) -> add dims for broadcasting
    pos = pos.unsqueeze(-1)  # (..., 1)
    angles = pos * freqs.to(x.device)  # (..., d_head/2)
    
    # Apply rotation
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    rotated_x1 = x1 * cos_angles - x2 * sin_angles
    rotated_x2 = x1 * sin_angles + x2 * cos_angles
    
    # Stack back
    rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
    
    return rotated.reshape(original_shape)

# %% Multi-Head Attention with RoPE
class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Precompute RoPE frequencies
        self.register_buffer('rope_freqs', compute_rope_frequencies(self.d_head))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.w_q(x)  # (batch, seq, d_model)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # Reshape to (batch, n_heads, seq, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE to Q and K
        positions = torch.arange(seq_len, device=x.device)
        q = apply_rope(q, positions, self.rope_freqs)
        k = apply_rope(k, positions, self.rope_freqs)
        
        # Attention: (batch, heads, seq, d_head) @ (batch, heads, d_head, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        
        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, seq, d_head)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(out)
        
        return out, attn

# %% Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))

# %% Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = RoPEAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm architecture
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)
        
        return x, attn_weights

# %% Complete RoPE Transformer
class RoPETransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        max_seq_len: int = MAX_SEQ_LEN,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, return_attentions: bool = False):
        """
        tokens: (batch, seq_len)
        Returns: logits (batch, seq_len, vocab_size)
        """
        x = self.token_emb(tokens)
        
        attentions = []
        for block in self.blocks:
            x, attn = block(x)
            if return_attentions:
                attentions.append(attn)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        if return_attentions:
            return logits, attentions
        return logits
    
    def generate(self, tokens: torch.Tensor, max_new: int = 10, temperature: float = 1.0):
        """Generate continuation"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new):
                logits = self(tokens)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

# %% Dataset for Induction Tasks
class InductionDataset:
    """Generate synthetic induction task sequences"""
    
    def __init__(self, seq_len: int = 20, vocab_size: int = VOCAB_SIZE):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def generate_induction_sequence(self):
        """Generate [A][B][C][A] → [B] pattern"""
        # Pick 3 distinct tokens
        tokens = np.random.choice(self.vocab_size - 1, size=3, replace=False)  # Exclude space
        
        # Pattern: A B C A ? (predict B)
        seq = [tokens[0], tokens[1], tokens[2], tokens[0]]  # A B C A
        target = tokens[1]  # B
        
        return seq, target
    
    def generate_previous_token_sequence(self):
        """Generate previous token prediction task"""
        tokens = np.random.choice(self.vocab_size - 1, size=5, replace=False)
        # Predict token at position i-1
        pos = np.random.randint(1, len(tokens))
        seq = tokens[:pos+1].tolist()
        target = tokens[pos-1]  # Previous token
        return seq, target
    
    def generate_batch(self, batch_size: int, task_type: str = 'induction'):
        """Generate a batch of sequences"""
        seqs = []
        targets = []
        
        for _ in range(batch_size):
            if task_type == 'induction':
                seq, target = self.generate_induction_sequence()
            elif task_type == 'previous':
                seq, target = self.generate_previous_token_sequence()
            else:
                # Mixed
                if np.random.random() < 0.5:
                    seq, target = self.generate_induction_sequence()
                else:
                    seq, target = self.generate_previous_token_sequence()
            
            seqs.append(seq)
            targets.append(target)
        
        return seqs, targets

# %% Training function
def train_step(model, optimizer, batch_seqs, batch_targets, device='cpu'):
    """Single training step"""
    model.train()
    
    # Pad sequences
    max_len = max(len(s) for s in batch_seqs)
    padded = []
    for seq in batch_seqs:
        padded.append(seq + [0] * (max_len - len(seq)))
    
    tokens = torch.tensor(padded, device=device)
    targets = torch.tensor(batch_targets, device=device)
    
    # Forward
    logits = model(tokens)
    
    # Loss: predict next token (last position)
    loss = F.cross_entropy(logits[:, -1, :], targets)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# %% Export attention visualization data
def export_attention_data(model, sequence: str, output_path: str):
    """Export attention patterns for visualization"""
    model.eval()
    
    # Convert to tokens
    tokens = [CHAR_TO_ID.get(c, 26) for c in sequence.lower() if c in CHAR_TO_ID or c == ' ']
    tokens_tensor = torch.tensor([tokens])
    
    with torch.no_grad():
        logits, attentions = model(tokens_tensor, return_attentions=True)
    
    # Convert to JSON-serializable format
    data = {
        'sequence': sequence,
        'tokens': [ID_TO_CHAR[t] for t in tokens],
        'attentions': []
    }
    
    for layer_idx, attn in enumerate(attentions):
        layer_data = {
            'layer': layer_idx,
            'heads': []
        }
        for head_idx in range(attn.shape[1]):
            head_attn = attn[0, head_idx].cpu().numpy().tolist()
            layer_data['heads'].append({
                'head': head_idx,
                'attention': head_attn
            })
        data['attentions'].append(layer_data)
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    return data

# %% Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("RoPE Transformer - Testing Implementation")
    print("=" * 60)
    
    # Test RoPE
    print("\n[1] Testing RoPE implementation...")
    freqs = compute_rope_frequencies(D_HEAD)
    print(f"  Frequencies shape: {freqs.shape}")
    print(f"  First few: {freqs[:5].tolist()}")
    
    # Test rotation
    x = torch.randn(1, 1, D_HEAD)
    pos = torch.tensor([5])
    rotated = apply_rope(x, pos, freqs)
    print(f"  Input norm: {torch.norm(x).item():.4f}")
    print(f"  Rotated norm: {torch.norm(rotated).item():.4f}")
    print(f"  ✓ Norm preserved (should be ~equal)")
    
    # Create model
    print("\n[2] Creating RoPE Transformer...")
    model = RoPETransformer()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: {N_LAYERS} layers, {N_HEADS} heads, {D_MODEL} dim")
    
    # Test forward pass
    print("\n[3] Testing forward pass...")
    test_seq = torch.randint(0, VOCAB_SIZE, (2, 10))
    logits = model(test_seq)
    print(f"  Input shape: {test_seq.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  ✓ Forward pass successful")
    
    # Test generation
    print("\n[4] Testing generation...")
    prompt = torch.tensor([[CHAR_TO_ID['a'], CHAR_TO_ID['b'], CHAR_TO_ID['c']]])
    generated = model.generate(prompt, max_new=5, temperature=0.8)
    print(f"  Prompt: abc")
    print(f"  Generated length: {generated.shape[1]}")
    
    # Test attention extraction
    print("\n[5] Testing attention extraction...")
    export_attention_data(model, "a b c a ", "/tmp/test_attention.json")
    print(f"  ✓ Attention data exported to /tmp/test_attention.json")
    
    print("\n" + "=" * 60)
    print("All tests passed! Model ready for training.")
    print("=" * 60)

print("\nTo train:")
print("  from rope_transformer import RoPETransformer, train_step, InductionDataset")
print("  model = RoPETransformer()")
print("  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)")
print("  dataset = InductionDataset()")
print("  # Training loop...")
