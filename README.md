# MNIST JAX MLP Project

## Overview

This project implements a Multi-Layer Perceptron (MLP) **from scratch** using JAX, featuring:

* Manual forward & backward pass with JAX's `grad` and custom `compute_loss`
* He initialization for weights (`generate_params`)
* Dropout regularization integrated into the feedforward pass
* Exponential learning rate scheduler and early stopping
* Batch-wise evaluation of loss and accuracy
* A Streamlit web app for drawing digits and real-time inference

All implemented without high-level frameworks like PyTorch, Keras, or Flax—just JAX and NumPy.

---

## Repository Structure

```
project_root/
├── mlp_model.py           # MLP definition: feedforward, loss, dropout, parameter updates
├── train_mnist_model.py   # Training script: scheduler, early stopping, evaluation
├── main_mnist_predict.py  # Streamlit app for interactive digit drawing and prediction
├── mlp_mnist_model.pkl    # Saved model parameters (pickle file)
└── README.md              # This documentation
```

---

## Installation

Make sure you have Python 3.8+ installed. Then install dependencies:

```bash
pip install jax jaxlib numpy datasets streamlit streamlit-drawable-canvas opencv-python
```

---

## Training the Model

1. Run the training script:

   ```bash
   python train_mnist_model.py
   ```
2. Model parameters and activation names will be saved to `mlp_mnist_model.pkl`.
3. Configuration options:

   * Batch size, number of epochs, learning rate schedule, regularization type (`l1`, `l2`, `l1_l2`), dropout rate, early stopping patience, and `min_delta` can be modified in `train_mnist_model.py`.

---

## Running the Streamlit App

1. Start the app:

   ```bash
   streamlit run main_mnist_predict.py
   ```
2. Open your browser to `http://localhost:8501`.
3. Draw a digit (0–9) on the canvas, then click **Predict**.
4. The app will display:

   * The original canvas image
   * The processed 28×28 grayscale input
   * Logits and predicted digit with confidence score

---

## Key Concepts

* **Manual MLP**: Constructs parameter lists and activation pipeline without high-level abstractions.
* **Loss & Grad**: Custom `compute_loss` (cross-entropy, MSE, etc.) paired with `jax.grad` for backpropagation.
* **Dropout**: Active only during training; disabled during inference.
* **Scheduler & Early Stopping**: Exponential decay learning rate and stop when no improvement for `patience` epochs.
* **Streamlit UI**: Provides a web interface for handwriting capture and live model inference.

---

## Pulling Model from Hugging Face Hub

The model is hosted on Hugging Face Hub (e.g. at `mystic/mnist-mlp`), you can download and load it in Python as follows:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import pickle

# Download the pickle file from the hub
model_path = hf_hub_download(repo_id="mysticdan/mnist-mlp", filename="mlp_mnist_model.pkl")

# Load model parameters and activation names
with open(model_path, "rb") as f:
    data = pickle.load(f)

# Reconstruct your MLP instance
mlp = MLP.__new__(MLP)
mlp.params = data["params"]
mlp.activation = tuple(
    activation_map["identity"] if name == "<lambda>" else activation_map[name]
    for name in data["activation_names"]
)
mlp.key = jax.random.key(0)

# Now you can call mlp.predict(...) on your input
```

---

*Developed by MysticDan*

