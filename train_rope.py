#!/usr/bin/env python3
"""
Train RoPE Transformer on Induction Tasks
=========================================
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path

from rope_transformer import RoPETransformer, InductionDataset, train_step, export_attention_data
from rope_transformer import CHAR_TO_ID, ID_TO_CHAR

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
STEPS_PER_EPOCH = 50
DEVICE = 'cpu'  # Change to 'cuda' if GPU available

def main():
    print("=" * 60)
    print("Training RoPE Transformer on Induction Tasks")
    print("=" * 60)
    
    # Create model
    model = RoPETransformer()
    model = model.to(DEVICE)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {DEVICE}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Dataset
    dataset = InductionDataset()
    
    # Training loop
    losses = []
    print("\nStarting training...")
    
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        
        for step in range(STEPS_PER_EPOCH):
            # Generate batch
            batch_seqs, batch_targets = dataset.generate_batch(BATCH_SIZE, task_type='mixed')
            
            # Train step
            loss = train_step(model, optimizer, batch_seqs, batch_targets, device=DEVICE)
            epoch_losses.append(loss)
            losses.append(loss)
        
        # Print progress
        avg_loss = np.mean(epoch_losses)
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Save model
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'losses': losses,
        'config': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'vocab_size': 27
        }
    }
    
    save_path = '/data/workspace/projects/rope-transformer/checkpoints/rope_transformer_trained.pt'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Test on specific examples
    print("\n" + "=" * 60)
    print("Testing on examples:")
    print("=" * 60)
    
    test_cases = [
        "a b c a ",      # Should predict 'b'
        "x y z x ",      # Should predict 'y'
        "m n o m ",      # Should predict 'n'
    ]
    
    model.eval()
    for seq_str in test_cases:
        tokens = [CHAR_TO_ID.get(c, 26) for c in seq_str.lower() if c in CHAR_TO_ID or c == ' ']
        tokens_tensor = torch.tensor([tokens]).to(DEVICE)
        
        with torch.no_grad():
            logits = model(tokens_tensor)
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=0)
            predicted_id = torch.argmax(probs).item()
            predicted_char = ID_TO_CHAR[predicted_id]
            
            # Get top 3 predictions
            top3 = torch.topk(probs, 3)
            top3_chars = [(ID_TO_CHAR[i.item()], p.item()) for i, p in zip(top3.indices, top3.values)]
        
        print(f"\n  Input: '{seq_str}'")
        print(f"  Predicted next: '{predicted_char}' (prob: {probs[predicted_id]:.3f})")
        print(f"  Top 3: {top3_chars}")
        
        # Export attention for first example
        if seq_str == test_cases[0]:
            export_attention_data(model, seq_str, 
                '/data/workspace/projects/rope-transformer/attention_viz.json')
            print(f"  ✓ Attention data exported")
    
    print("\n" + "=" * 60)
    print("Training complete! Files saved:")
    print(f"  - Model: {save_path}")
    print(f"  - Attention viz: /data/workspace/projects/rope-transformer/attention_viz.json")
    print("=" * 60)

if __name__ == '__main__':
    main()
