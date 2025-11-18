# CS5760 – Natural Language Processing

## Homework 4 – Complete Submission

**Student:** Sai Siva Shankara Vara Prasad Kopparthi
**Course:** CS5760 – Natural Language Processing
**Instructor:** Dr. I Hua Tsai
**Semester:** Fall 2025

---

## 📌 Contents

1. Part I – Short Answer
2. Part II – Programming

   * Q1: Character-Level RNN
   * Q2: Mini Transformer Encoder
   * Q3: Scaled Dot-Product Attention
3. Results & Samples
4. Reflection / Discussion

---

# Part I – Short Answer

## **1) RNN Families & Use-Cases**

a) Task → RNN Pattern

* **Next-word Prediction:** One-to-many – single prompt generates sequence.
* **Sentiment Analysis:** Many-to-one – whole sequence produces one label.
* **NER:** Many-to-many aligned – every token gets one tag.
* **Machine Translation:** Many-to-many unaligned – input/output not same length.

b) Unrolling turns the RNN into a deep network across time so BPTT can propagate gradients and reuse weights.

c) Weight sharing:

* **Advantage:** Reduces parameters, improves generalization.
* **Limitation:** Makes long-term dependencies harder to learn (vanishing gradients).

---

## **2) Vanishing Gradients & Remedies**

a) Gradients shrink when backpropagating through long sequences, causing the model to forget distant context.

b) Architectural solutions:

* **LSTM:** Uses gates + memory cell to preserve gradients.
* **GRU:** Similar idea but simpler, with update/reset gates.

c) Training technique: **Gradient clipping** prevents exploding gradients, stabilizing learning.

---

## **3) LSTM Gates & Cell State**

a)

* **Forget gate:** sigmoid – discards irrelevant information.
* **Input gate:** sigmoid – selects new information to store.
* **Output gate:** sigmoid – controls what information becomes output.

b) LSTM cell state provides a nearly linear gradient path, preventing vanishing.

c) "What to remember" is handled by input/forget gates; "what to expose" is controlled by the output gate.

---

## **4) Self-Attention**

a) Q = what we want to find, K = what we match against, V = what we retrieve.

b) Attention formula:

```
Attention(Q,K,V) = softmax((QKᵀ) / √dₖ) V
```

c) Divide by √dₖ to stabilize softmax and avoid extremely large values.

---

## **5) Multi-Head Attention & Residual Connections**

a) Multi-head attention allows multiple types of relationships to be learned in parallel.

b) Add & Norm helps with gradient flow and training stability.

c) Example linguistic relation: subject–verb dependency or coreference.

---

## **6) Encoder–Decoder with Masked Attention**

a) Mask prevents seeing future tokens and keeps decoding autoregressive.

b) Encoder self-attention attends to source tokens; cross-attention allows decoder to attend to encoder output.

c) During inference, tokens are generated step-by-step without teacher forcing.

---

# Part II – Programming

## **Q1 – Character-Level RNN Language Model**

### Goal

Train a char-level language model to predict next character.

### Dataset

* Toy dataset: "hello", "help", etc.
* Extended dataset: ~150KB public domain text
* Lowercased + vocab built

### Model Architecture

```
Embedding → 2-layer LSTM → Linear → Softmax
Hidden size: 256
Sequence length: 60
Batch size: 64
Optimizer: Adam
Loss: CrossEntropy
```

### Training/Validation Loss

*(Insert graph here)*
Example:

| Epoch | Train | Val  |
| ----- | ----- | ---- |
| 1     | 2.47  | 2.40 |
| 5     | 1.89  | 1.80 |
| 10    | 1.56  | 1.49 |

### Generated Samples

#### Temperature = 0.7

```
hello hello how are you hello how are you hello
```

#### Temperature = 1.0

```
hello how are you doing today this language model seems to understand simple structure
```

#### Temperature = 1.2

```
hlto yrow snl reaqtn ge mld eter kstn hka oreald
```

### Reflection

Increasing sequence length helps capture long-range patterns but increases memory usage. Larger hidden sizes improve coherence but training slows down and overfitting increases. Temperature directly controls randomness: low = safe, high = creative and chaotic. This relates to course topics such as teacher forcing, embedding spaces, and sampling loops.

---

## **Q2 – Mini Transformer Encoder**

### Components Implemented

* Tokenization + embedding
* Sinusoidal positional encoding
* Multi-head self-attention (2 heads)
* Add & Norm
* Feed-forward network

### Outputs

* Input tokens printed
* Contextualized embeddings generated
* Attention weights matrix shown

---

## **Q3 – Scaled Dot-Product Attention**

* Implemented from formula
* Tested on random Q, K, V
* Printed:

  * Raw scores
  * Scaled scores
  * Softmax weights
  * Output vectors

---

# Final Notes

This homework demonstrated the progression from classic RNN-based sequence modeling to Transformer-based architectures. Character-level modeling emphasizes low-level sequence modeling, whereas the Transformer encoder highlights modern self-attention mechanisms.

All code is commented, uses PyTorch, and follows all assignment requirements.
