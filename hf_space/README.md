---
title: AlphaGalerkin Go
emoji: ⚫
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# AlphaGalerkin Go Demo

This Space demonstrates **AlphaGalerkin**, a resolution-independent Neural Operator model for the game of Go.

## About the Model

AlphaGalerkin uses a continuous operator learning approach (Galerkin Transformers & FNet) to learn the game of Go in a resolution-independent manner. This allows the model to be trained on small boards (e.g., 9x9) and potentially transfer to larger boards.

## How to Play

1. The board is displayed on the left.
2. Enter your move in the text box as `row,col` (e.g., `3,3` for the 4-4 point, since it's 0-indexed) or input `PASS`.
3. Click "Submit Move".
4. The AI (White) will respond.

## Local Installation

To run this locally:

```bash
git clone https://huggingface.co/spaces/[your-username]/alphagalerkin
cd alphagalerkin
pip install -r requirements.txt
python app.py
```
