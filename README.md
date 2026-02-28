# Med-GPT: Autoregressive Transformer for Medication Trajectories

This repository contains a PyTorch implementation of a GPT-style (decoder-only) Transformer designed to model longitudinal patient medication histories. 

The model treats medical histories as a "language," where drug codes and time intervals form a sequence. By training the model to predict the next event in a patient's history, we can extract high-dimensional embeddings that summarize complex clinical trajectories.

## 💡 The Concept

Standard clinical models often treat medical codes as a static "bag-of-words." This project captures the **temporal dynamics** of healthcare by representing a patient's history as an interleaved sequence:

`[START] -> [Δt_0] -> [ATC_1] -> [Δt_1] -> [ATC_2] ... [END]`

* **ATC Tokens:** Represent specific medication classes (Anatomical Therapeutic Chemical Classification System).
* **Time Tokens ($\Delta t$):** Represent discretized time bins between prescriptions (e.g., "0-7 days", "1 month").
* **Self-Supervised Learning:** By predicting the next token, the model learns the underlying patterns of polypharmacy and disease progression without needing manual labels.



---

## 🏗️ Model Architecture

The core of the system is a **SimpleGPT** model consisting of:

* **Causal Self-Attention:** Uses a triangular mask to ensure that at any position $t$, the model can only attend to past events ($<t$), preventing information leakage from the "future."
* **Transformer Decoder Blocks:** Multiple layers featuring:
    * **Multi-Head Attention:** To find correlations between medications separated by long time gaps.
    * **Pre-Normalization:** `LayerNorm` is applied before attention and MLP blocks for more stable training.
    * **Position-wise Feed-Forward Networks (MLP):** Two linear layers with a ReLU activation.
* **Contextual Embeddings:** A dedicated `extract_hidden()` method allows you to pull the final hidden states (the "thought vectors") for downstream tasks like patient clustering, mortality prediction, or risk stratification.



---

## 🚀 Key Features

* **Sliding Window Processing:** Automatically chunks long patient histories into segments of `max_seq_len` with a configurable overlap to maintain longitudinal context.
* **Causal Masking:** Implements a boolean mask to maintain the autoregressive property.
* **Early Stopping:** Automatically halts training when validation loss stops improving to prevent overfitting.
* **Flexible Configuration:** Hyperparameters (embedding size, number of heads, dropout, etc.) are managed via a single `YAML` file.
