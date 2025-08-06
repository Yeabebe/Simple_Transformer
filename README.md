# Transformer Decoder-Only Model for Text Generation (PyTorch)

This repository contains a clean and minimal implementation of a **decoder-only Transformer architecture** built from scratch using PyTorch. It includes core components such as multi-head self-attention, positional encoding, feedforward networks, and autoregressive text generation.

---

## Features

- Decoder-only Transformer model (GPT-style)
- Positional encoding for sequence order
- Causal (autoregressive) masking
- Multi-head self-attention
- Greedy decoding for text generation
- Modular, readable code in a single Jupyter notebook
- Token-level inference with optional EOS handling

---

## 📁 Structure
```
.
├── transformer_decoder.ipynb # Jupyter notebook with full model code
├── README.md # This file
```
---

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- Jupyter Notebook (or JupyterLab)

Install PyTorch via [https://pytorch.org](https://pytorch.org) or:

```
pip install torch
```
How to Use
Inside the notebook:
```
# Initialize model
model = TransformerDecoder(vocab_size=100)

# Provide starting input tokens
start = torch.tensor([[1, 5, 7]])

# Generate new sequence
output = generate_text(model, start, max_new_tokens=10)

# Print generated token IDs
print(output.tolist())
```
## Notes
This model uses randomly initialized weights — it will generate nonsense tokens unless trained.
You can integrate your own tokenizer and training loop to turn this into a fully working language model.
Supports batch inference and trimming based on max_len.

## Learning Goals
This project is ideal for:
- Understanding the internal mechanics of Transformers
- Educational use in deep learning courses
- Building a foundation for autoregressive models like GPT

## License
MIT License – feel free to use, modify, and share.

## Acknowledgments
Inspired by the original "Attention Is All You Need" paper and the GPT architecture. Designed for learning and experimentation.
