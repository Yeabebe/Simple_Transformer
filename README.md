# Transformer Decoder-Only Model for Text Generation (PyTorch)

This repository contains a decoder-only Transformer architecture (GPT-style) built from scratch in PyTorch.
It includes data loading from the Penn Treebank dataset, a full training loop with checkpointing, and a simple text generation function.

---

## Features

- Decoder-only Transformer model (GPT-style)
- Positional encoding for sequence order
- Causal (autoregressive) masking
- Multi-head self-attention
- Penn Treebank dataset integration
- Training loop with validation
- Checkpoint saving & loading
- Greedy decoding for text generation
- Modular, readable code in a single Jupyter notebook

---

## 📁 Structure
```
.
├── dataset
├── transformerDecoder.ipynb # Jupyter notebook with full model code
├── README.md # This file
```
---

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- Jupyter Notebook (or JupyterLab)
- TorchText (for dataset handling)

Install PyTorch via [https://pytorch.org](https://pytorch.org) or:

```
pip install torch
```
## How to Use

1. Prepare Dataset
Download the Penn Treebank dataset
and place it inside the dataset/ folder.

2. Train the Model
Inside the notebook:
```
# Train for 3 epochs (example)
for epoch in range(3):
    train(...)
    validate(...)
    save_checkpoint("checkpoint.pth")
```
3. Resume Training
```
model, optimizer, epoch = load_checkpoint("checkpoint.pth")
```
5. Generate Text
```
# Provide starting input tokens
start = torch.tensor([[1, 5, 7]]).to(device)

# Generate new sequence
output = generate_text(model, start, max_new_tokens=20, device=device)

print(output.tolist())
```
## Transformer Architecture 
<img width="1024" height="1536" alt="ChatGPT Image Aug 16, 2025, 04_27_32 PM" src="https://github.com/user-attachments/assets/fa5dc4c0-437b-45ec-a8e0-94ff5402eb06" />


## Notes
- Model starts with random weights; training is required for meaningful text generation.
- You can extend the notebook with better sampling methods (top-k, nucleus sampling).
- Checkpoints let you resume training without starting from scratch.

## Learning Goals
This project is ideal for:
- Understanding Transformer internals step by step
- Experimenting with dataset-driven training
- Learning autoregressive language modeling
- Extending towards GPT-style models

## License
MIT License – feel free to use, modify, and share.

## Acknowledgments
Inspired by the original "Attention Is All You Need" paper and the GPT architecture. Designed for learning and experimentation.
