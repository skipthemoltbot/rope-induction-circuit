# RoPE Transformer Explorer

An educational, interactive visualization of a small transformer with **Rotary Position Embeddings (RoPE)** trained to demonstrate **induction head** behavior.

## Architecture

| Parameter | Value |
|-----------|-------|
| Vocabulary | 27 tokens (a-z + space) |
| Embedding dim | 128 |
| Layers | 2 |
| Attention heads | 4 per layer |
| Head dimension | 32 |
| FFN hidden dim | 512 |
| Position encoding | RoPE (no learned positional embeddings) |
| Total parameters | ~330K |

## RoPE Mathematics

### The Problem with Absolute Position Embeddings

Traditional transformers add a fixed positional embedding to each token:
```
input = token_embedding + position_embedding
```

This means attention patterns are tied to absolute positions, not relative distances between tokens.

### RoPE: Encoding Position Through Rotation

RoPE applies position-dependent rotations to query and key vectors *inside* the attention mechanism, making the dot product naturally depend on relative position.

### Step 1: Define Rotation Frequencies

For a head dimension `d`, we define frequencies for each dimension pair `i`:

```
θ_i = 1 / (10000^(2i/d))    for i = 0, 1, ..., d/2 - 1
```

Low-index pairs rotate fast (capturing local patterns), high-index pairs rotate slowly (capturing global structure).

### Step 2: Apply Rotation to Q and K

For a query or key vector `x` at position `m`, we rotate each consecutive pair of dimensions:

```
RoPE(x, m)[2i]   = x[2i]   · cos(m·θ_i) − x[2i+1] · sin(m·θ_i)
RoPE(x, m)[2i+1] = x[2i]   · sin(m·θ_i) + x[2i+1] · cos(m·θ_i)
```

This is equivalent to a 2D rotation matrix applied to each pair:

```
┌ cos(m·θ_i)  −sin(m·θ_i) ┐ ┌ x[2i]   ┐
│                           │ │          │
└ sin(m·θ_i)   cos(m·θ_i)  ┘ └ x[2i+1] ┘
```

### Step 3: Position-Relative Attention

The key mathematical property:

```
dot(RoPE(q, m), RoPE(k, n)) = dot(R(m)·q, R(n)·k) = q^T · R(m−n) · k
```

The attention score between positions `m` and `n` depends only on `(m−n)`, the relative distance. This emerges naturally from the rotation algebra — no explicit relative position mechanism needed.

### Step 4: Attention Computation

```
Attention = softmax(Q_rot · K_rot^T / √d_k) · V
```

Note: RoPE is applied to Q and K only, **not** to V. Values carry content information, while Q/K interactions encode both content and position.

## Induction Heads

An **induction head** is a circuit across two attention layers that copies patterns:

1. **Layer 0 (Previous Token Head)**: Learns to attend to the previous token. When processing token B, it copies information about token A (which precedes B).

2. **Layer 1 (Induction Head)**: Uses the "previous token" information from Layer 0 to find where the same pattern occurred before. When it sees token A again, it attends to the position where A was followed by B, and predicts B.

### How RoPE Helps Induction

RoPE makes attention inherently position-relative, which is exactly what induction heads need. The induction head doesn't need to learn "attend to position 3" — it learns "attend to the position whose content matches, regardless of absolute position."

### Testing Induction

The model is tested on patterns like:
- `"abc abc"` — after seeing `a` the second time, predict `b`
- `"hello hello"` — complete repeated words
- `"the cat the cat"` — longer repeated patterns

## Running

```bash
pip install torch fastapi uvicorn
cd projects/rope-transformer
python model.py          # Train and save weights
uvicorn server:app       # Start web server
# Open http://localhost:8000
```

## Interactive Demo

The web interface shows every intermediate computation:

1. **Token Embeddings** — The raw embedding vectors (no positional encoding added)
2. **RoPE Rotations** — Rotation frequencies, circle visualizations, cos/sin heatmaps
3. **Layer 0 & 1** — For each attention head:
   - Q, K vectors before and after RoPE rotation
   - Attention weight heatmaps (the key visualization for induction)
   - Value vectors
4. **Predictions** — Top-5 next-token predictions with probabilities

Look for the characteristic induction pattern in Layer 1 attention: off-diagonal stripes where the model attends to previous occurrences of the current token's context.
