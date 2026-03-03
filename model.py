"""
RoPE Transformer for Induction Head Discovery
==============================================

A small transformer (2 layers, 4 heads, 128 dim) with Rotary Position Embeddings
trained to demonstrate induction behavior on a 27-token vocabulary (a-z + space).

RoPE Math:
----------
Given a position m and dimension pair (2i, 2i+1):

  theta_i = 1 / (10000 ^ (2i / d))

For query/key vector x at position m:
  RoPE(x, m)[2i]   = x[2i]   * cos(m * theta_i) - x[2i+1] * sin(m * theta_i)
  RoPE(x, m)[2i+1] = x[2i]   * sin(m * theta_i) + x[2i+1] * cos(m * theta_i)

This is equivalent to multiplying each 2D pair by a rotation matrix:
  [cos(m*theta_i)  -sin(m*theta_i)] [x[2i]  ]
  [sin(m*theta_i)   cos(m*theta_i)] [x[2i+1]]

Key insight: dot(RoPE(q, m), RoPE(k, n)) depends only on (m - n),
making attention naturally position-relative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from typing import Optional


# Vocabulary: 26 lowercase letters + space = 27 tokens
VOCAB = list("abcdefghijklmnopqrstuvwxyz ")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}

# Model hyperparameters
D_MODEL = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 32
N_LAYERS = 2
MAX_SEQ_LEN = 128
ROPE_BASE = 10000.0


def precompute_rope_frequencies(d_head: int, max_seq_len: int, base: float = ROPE_BASE):
    """
    Precompute the rotation frequencies for RoPE.

    theta_i = 1 / (base ^ (2i / d_head))  for i = 0, 1, ..., d_head/2 - 1

    Returns:
        freqs: (max_seq_len, d_head/2) - the angles m * theta_i
        cos_cache: (max_seq_len, d_head/2) - cos(m * theta_i)
        sin_cache: (max_seq_len, d_head/2) - sin(m * theta_i)
    """
    # Dimension indices: i = 0, 1, ..., d_head/2 - 1
    dim_indices = torch.arange(0, d_head, 2, dtype=torch.float32)  # [0, 2, 4, ...]

    # theta_i = 1 / (base ^ (2i / d_head))
    thetas = 1.0 / (base ** (dim_indices / d_head))  # (d_head/2,)

    # Position indices: m = 0, 1, ..., max_seq_len - 1
    positions = torch.arange(0, max_seq_len, dtype=torch.float32)  # (max_seq_len,)

    # Outer product: freqs[m, i] = m * theta_i
    freqs = torch.outer(positions, thetas)  # (max_seq_len, d_head/2)

    cos_cache = torch.cos(freqs)  # (max_seq_len, d_head/2)
    sin_cache = torch.sin(freqs)  # (max_seq_len, d_head/2)

    return freqs, cos_cache, sin_cache


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply rotary position embeddings to input tensor.

    For each pair (x[2i], x[2i+1]) at position m:
      x_rot[2i]   = x[2i]   * cos(m * theta_i) - x[2i+1] * sin(m * theta_i)
      x_rot[2i+1] = x[2i]   * sin(m * theta_i) + x[2i+1] * cos(m * theta_i)

    Args:
        x: (..., seq_len, d_head) - query or key vectors
        cos: (seq_len, d_head/2) - precomputed cosines
        sin: (seq_len, d_head/2) - precomputed sines

    Returns:
        x_rotated: same shape as x
    """
    # Split into even and odd dimensions
    x_even = x[..., 0::2]  # (..., seq_len, d_head/2)
    x_odd = x[..., 1::2]   # (..., seq_len, d_head/2)

    # Apply rotation
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    # Interleave back
    x_rotated = torch.stack([x_rot_even, x_rot_odd], dim=-1)
    x_rotated = x_rotated.flatten(-2)  # (..., seq_len, d_head)

    return x_rotated


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Precompute RoPE frequencies
        freqs, cos_cache, sin_cache = precompute_rope_frequencies(self.d_head, MAX_SEQ_LEN)
        self.register_buffer('rope_freqs', freqs)
        self.register_buffer('rope_cos', cos_cache)
        self.register_buffer('rope_sin', sin_cache)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_intermediates: bool = False):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional causal mask
            return_intermediates: if True, return all intermediate tensors
        """
        B, T, D = x.shape

        # Project to Q, K, V
        Q = self.W_Q(x)  # (B, T, D)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head: (B, n_heads, T, d_head)
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Store pre-RoPE Q, K for visualization
        Q_pre_rope = Q.clone()
        K_pre_rope = K.clone()

        # Apply RoPE to Q and K
        cos = self.rope_cos[:T]  # (T, d_head/2)
        sin = self.rope_sin[:T]

        Q_rot = apply_rope(Q, cos, sin)
        K_rot = apply_rope(K, cos, sin)

        # Attention scores: (Q_rot @ K_rot^T) / sqrt(d_head)
        scale = math.sqrt(self.d_head)
        attn_scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Apply causal mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, d_head)

        # Reshape back: (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        output = self.W_O(attn_output)

        if return_intermediates:
            intermediates = {
                'Q_pre_rope': Q_pre_rope.detach(),
                'K_pre_rope': K_pre_rope.detach(),
                'V': V.detach(),
                'Q_rotated': Q_rot.detach(),
                'K_rotated': K_rot.detach(),
                'rope_cos': cos.detach(),
                'rope_sin': sin.detach(),
                'rope_freqs': self.rope_freqs[:T].detach(),
                'attn_scores_raw': attn_scores.detach(),
                'attn_weights': attn_weights.detach(),
                'attn_output': attn_output.detach(),
            }
            return output, intermediates

        return output


class TransformerBlock(nn.Module):
    """Transformer block: attention + FFN with residual connections and layer norm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, mask=None, return_intermediates=False):
        # Pre-norm attention
        x_normed = self.ln1(x)

        if return_intermediates:
            attn_out, intermediates = self.attn(x_normed, mask, return_intermediates=True)
            intermediates['input_to_attn'] = x_normed.detach()
        else:
            attn_out = self.attn(x_normed, mask)
            intermediates = {}

        x = x + attn_out  # residual

        # Pre-norm FFN
        x_normed2 = self.ln2(x)
        ffn_out = self.ffn(x_normed2)
        x = x + ffn_out  # residual

        if return_intermediates:
            intermediates['post_attn_residual'] = (x - ffn_out).detach()
            intermediates['ffn_output'] = ffn_out.detach()
            intermediates['block_output'] = x.detach()

        return x, intermediates


class RoPETransformer(nn.Module):
    """
    Small RoPE transformer for induction head demonstration.

    Architecture:
      - Token embedding (27 -> 128)
      - 2 transformer blocks (each with 4-head attention + FFN)
      - Output projection (128 -> 27)

    No learned positional embeddings - RoPE provides position info!
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Token embedding - NO positional embedding (RoPE handles position)
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # Final layer norm + output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor, return_intermediates: bool = False):
        """
        Args:
            token_ids: (batch, seq_len) - integer token IDs
            return_intermediates: if True, return all intermediate tensors
        """
        B, T = token_ids.shape

        # Token embedding
        x = self.token_emb(token_ids)  # (B, T, D)

        # Causal mask: each position can only attend to previous positions
        mask = torch.tril(torch.ones(T, T, device=token_ids.device)).unsqueeze(0).unsqueeze(0)

        all_intermediates = {}
        if return_intermediates:
            all_intermediates['token_embeddings'] = x.detach()

        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            x, block_intermediates = block(x, mask, return_intermediates=return_intermediates)
            if return_intermediates:
                all_intermediates[f'layer_{i}'] = block_intermediates

        # Final layer norm
        x = self.ln_final(x)

        # Output logits
        logits = self.output_proj(x)  # (B, T, vocab_size)

        if return_intermediates:
            all_intermediates['final_hidden'] = x.detach()
            all_intermediates['logits'] = logits.detach()
            all_intermediates['probs'] = F.softmax(logits, dim=-1).detach()

        return logits, all_intermediates

    def predict(self, text: str):
        """Run inference on a text string, returning all intermediates."""
        token_ids = torch.tensor([[CHAR_TO_ID.get(c, CHAR_TO_ID[' ']) for c in text]])
        logits, intermediates = self.forward(token_ids, return_intermediates=True)
        return logits, intermediates


def create_induction_training_data(n_samples: int = 10000, seq_len: int = 64):
    """
    Generate training data designed to teach induction heads.

    Induction pattern: if "AB" appeared before, and we see "A" again,
    predict "B" next. We create sequences with repeated bigrams.

    Format: random prefix + pattern + random middle + pattern again
    """
    data = []
    for _ in range(n_samples):
        seq = torch.randint(0, VOCAB_SIZE, (seq_len,))

        # Create repeated patterns at random positions
        # Pick a pattern length between 2-6 characters
        pattern_len = torch.randint(2, 7, (1,)).item()
        pattern = torch.randint(0, VOCAB_SIZE, (pattern_len,))

        # Place pattern at a random early position
        pos1 = torch.randint(0, seq_len // 3, (1,)).item()
        end1 = min(pos1 + pattern_len, seq_len)
        seq[pos1:end1] = pattern[:end1 - pos1]

        # Place same pattern again later (this is what induction heads learn)
        pos2 = torch.randint(seq_len // 3, 2 * seq_len // 3, (1,)).item()
        end2 = min(pos2 + pattern_len, seq_len)
        seq[pos2:end2] = pattern[:end2 - pos2]

        # Optionally place it a third time
        if torch.rand(1).item() > 0.5:
            pos3 = torch.randint(2 * seq_len // 3, seq_len - pattern_len, (1,)).item()
            end3 = min(pos3 + pattern_len, seq_len)
            seq[pos3:end3] = pattern[:end3 - pos3]

        data.append(seq)

    return torch.stack(data)


def train_model(model: RoPETransformer, epochs: int = 30, batch_size: int = 64,
                lr: float = 3e-4, seq_len: int = 64, device: str = 'cpu',
                verbose: bool = True):
    """Train the model to learn induction heads."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if verbose:
        print("Generating training data...")
    train_data = create_induction_training_data(n_samples=10000, seq_len=seq_len).to(device)

    if verbose:
        print(f"Training on {len(train_data)} sequences of length {seq_len}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        # Shuffle data
        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            # Input is all tokens except last, target is all tokens except first
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    if verbose:
        print(f"Training complete. Best loss: {best_loss:.4f}")

    return model


def test_induction(model: RoPETransformer, verbose: bool = True):
    """Test if the model has learned induction behavior."""
    model.eval()

    test_cases = [
        "abc abc",           # Should predict: after second 'a', predict 'b'; after second 'b', predict 'c'
        "hello hello",       # Should complete the repeated word
        "xyz xyz",           # Another repeated pattern
        "the cat the cat",   # Longer pattern
        "ab cd ab",          # Should predict 'cd' after second 'ab'
    ]

    results = []
    with torch.no_grad():
        for text in test_cases:
            logits, intermediates = model.predict(text)
            probs = F.softmax(logits[0], dim=-1)

            result = {'input': text, 'predictions': []}
            for pos in range(len(text)):
                top_probs, top_ids = probs[pos].topk(3)
                preds = [(ID_TO_CHAR[idx.item()], prob.item()) for idx, prob in zip(top_ids, top_probs)]
                result['predictions'].append({
                    'position': pos,
                    'char': text[pos],
                    'top_predictions': preds
                })

                if verbose and pos > 0:
                    actual_next = text[pos] if pos < len(text) else '?'
                    pred_char = preds[0][0]
                    correct = "✓" if pred_char == actual_next else " "
                    # Show what the model predicts AFTER seeing this character

            if verbose:
                print(f"\nInput: '{text}'")
                print(f"{'Pos':>3} {'Char':>4} → {'Pred':>4} ({'Prob':>5}) | Top 3")
                print("-" * 60)
                for pos in range(len(text) - 1):
                    pred_info = result['predictions'][pos]
                    next_char = text[pos + 1]
                    top = pred_info['top_predictions']
                    correct = "✓" if top[0][0] == next_char else " "
                    top_str = ", ".join(f"'{c}':{p:.3f}" for c, p in top)
                    disp_char = '⎵' if text[pos] == ' ' else text[pos]
                    disp_next = '⎵' if next_char == ' ' else next_char
                    print(f"{pos:>3} '{disp_char}' → '{top[0][0]}' ({top[0][1]:.3f}) {correct} [actual: '{disp_next}'] | {top_str}")

            results.append(result)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("RoPE Transformer - Induction Head Training")
    print("=" * 60)
    print(f"Vocab: {VOCAB_SIZE} tokens (a-z + space)")
    print(f"Model: {N_LAYERS} layers, {N_HEADS} heads, d={D_MODEL}")
    print(f"Head dim: {D_HEAD}, RoPE base: {ROPE_BASE}")
    print()

    model = RoPETransformer()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    model = train_model(model, epochs=30, batch_size=64, seq_len=64)

    print("\n" + "=" * 60)
    print("Testing Induction Behavior")
    print("=" * 60)
    test_induction(model)

    # Save model
    save_path = "weights/rope_transformer.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
