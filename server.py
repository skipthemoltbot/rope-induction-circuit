"""
FastAPI server for RoPE Transformer inference.
Exposes all intermediate tensors for educational visualization.
"""

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import math
import os

from model import (
    RoPETransformer, VOCAB, VOCAB_SIZE, CHAR_TO_ID, ID_TO_CHAR,
    D_MODEL, N_HEADS, D_HEAD, N_LAYERS, ROPE_BASE, MAX_SEQ_LEN,
    precompute_rope_frequencies, train_model, test_induction
)

app = FastAPI(title="RoPE Transformer Explorer")

# Global model
model: RoPETransformer = None


def tensor_to_list(t: torch.Tensor, max_dims: int = 4):
    """Convert tensor to nested list, rounding for JSON sanity."""
    if t.dim() == 0:
        return round(t.item(), 6)
    return [[round(v, 6) for v in row] for row in t.reshape(-1, t.shape[-1]).tolist()] \
        if t.dim() <= 2 else t.tolist()


def format_matrix(t: torch.Tensor, precision: int = 4):
    """Convert a 2D tensor to a list of lists with controlled precision."""
    if t.dim() == 1:
        return [round(v, precision) for v in t.tolist()]
    result = []
    for row in t.tolist():
        if isinstance(row, list):
            result.append([round(v, precision) for v in row])
        else:
            result.append(round(row, precision))
    return result


class InferenceRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup():
    global model
    model = RoPETransformer()

    weight_path = os.path.join(os.path.dirname(__file__), "weights", "rope_transformer.pt")
    if os.path.exists(weight_path):
        print(f"Loading trained weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    else:
        print("No trained weights found. Training model...")
        model = train_model(model, epochs=30, batch_size=64, seq_len=64, verbose=True)
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save(model.state_dict(), weight_path)
        print(f"Saved trained weights to {weight_path}")

    model.eval()
    print("Model ready!")
    test_induction(model)


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/api/model-info")
async def model_info():
    """Return model architecture information."""
    return {
        "vocab": VOCAB,
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "d_head": D_HEAD,
        "n_layers": N_LAYERS,
        "rope_base": ROPE_BASE,
        "max_seq_len": MAX_SEQ_LEN,
        "total_params": sum(p.numel() for p in model.parameters()),
    }


@app.post("/api/inference")
async def inference(req: InferenceRequest):
    """
    Run inference on input text and return ALL intermediate tensors.
    This is the main educational endpoint.
    """
    text = req.text.lower()
    if not text:
        return JSONResponse({"error": "Empty input"}, status_code=400)

    # Clamp length
    text = text[:MAX_SEQ_LEN]

    # Filter to valid chars
    text = ''.join(c if c in CHAR_TO_ID else ' ' for c in text)

    tokens = [CHAR_TO_ID[c] for c in text]
    token_ids = torch.tensor([tokens])

    with torch.no_grad():
        logits, intermediates = model(token_ids, return_intermediates=True)

    seq_len = len(text)
    response = {
        "input": {
            "text": text,
            "tokens": tokens,
            "token_names": list(text),
            "seq_len": seq_len,
        },
        "token_embeddings": format_matrix(intermediates['token_embeddings'][0]),
        "rope_info": {},
        "layers": [],
        "final_hidden": format_matrix(intermediates['final_hidden'][0]),
        "predictions": [],
    }

    # RoPE frequency information
    freqs, cos_cache, sin_cache = precompute_rope_frequencies(D_HEAD, seq_len)
    response["rope_info"] = {
        "thetas": format_matrix(1.0 / (ROPE_BASE ** (torch.arange(0, D_HEAD, 2, dtype=torch.float32) / D_HEAD))),
        "frequencies": format_matrix(freqs),   # (seq_len, d_head/2) - the angles m*theta_i
        "cos_values": format_matrix(cos_cache), # cos(m*theta_i)
        "sin_values": format_matrix(sin_cache), # sin(m*theta_i)
        "d_head": D_HEAD,
        "n_pairs": D_HEAD // 2,
        "base": ROPE_BASE,
    }

    # Layer-by-layer intermediates
    for layer_idx in range(N_LAYERS):
        layer_key = f'layer_{layer_idx}'
        layer_data = intermediates[layer_key]

        layer_info = {
            "layer_idx": layer_idx,
            "heads": [],
        }

        for head_idx in range(N_HEADS):
            head_info = {
                "head_idx": head_idx,
                # Q, K, V before RoPE (for this head)
                "Q_pre_rope": format_matrix(layer_data['Q_pre_rope'][0, head_idx]),
                "K_pre_rope": format_matrix(layer_data['K_pre_rope'][0, head_idx]),
                "V": format_matrix(layer_data['V'][0, head_idx]),
                # Q, K after RoPE
                "Q_rotated": format_matrix(layer_data['Q_rotated'][0, head_idx]),
                "K_rotated": format_matrix(layer_data['K_rotated'][0, head_idx]),
                # Attention
                "attn_scores": format_matrix(layer_data['attn_scores_raw'][0, head_idx]),
                "attn_weights": format_matrix(layer_data['attn_weights'][0, head_idx]),
            }
            layer_info["heads"].append(head_info)

        layer_info["block_output"] = format_matrix(layer_data['block_output'][0])
        response["layers"].append(layer_info)

    # Final predictions for each position
    probs = F.softmax(logits[0], dim=-1)
    for pos in range(seq_len):
        top_k = 5
        top_probs, top_ids = probs[pos].topk(top_k)
        preds = []
        for j in range(top_k):
            char = ID_TO_CHAR[top_ids[j].item()]
            preds.append({
                "token": char,
                "token_id": top_ids[j].item(),
                "probability": round(top_probs[j].item(), 6),
            })
        response["predictions"].append({
            "position": pos,
            "input_char": text[pos],
            "top_predictions": preds,
            "all_probs": format_matrix(probs[pos]),
        })

    return response


# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
