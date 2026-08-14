# Architecture & Math Notes

This document explains the core building blocks of the framework and the
mathematics used during training. Implementation lives in `tensor.py`,
`layer.py`, `model.py`, and `mathematics.py`.

---

## Vector

A **Vector** is a 1-D list of numbers with length `n`.

**Responsibilities**
- Element-wise `+`, `-`, `*`
- Dot product and matrix multiply helpers (`@`)
- Utilities: `copy`, `bias` / `unbias`, `apply`, `softmax`, `argmax`

**Bias trick**  
Appending a constant `1` to an activation vector lets the next weight matrix
absorb a bias term as an extra row, so \(z = x_b W\) includes bias without a
separate bias vector.

**Softmax** (numerically stable)

\[
p_i = \frac{e^{z_i - \max z}}{\sum_j e^{z_j - \max z}}
\]

---

## Matrix

A **Matrix** is a list of row `Vector`s with shape `(n, m)`.

**Responsibilities**
- Element-wise `+`, `-`, `*`
- Matrix multiplication `@`
- `transpose`, `flatten`, `copy`, `bias` / `unbias`

In this codebase, a dense layer with bias stores \(W \in \mathbb{R}^{(d_{\mathrm{in}}+1)\times d_{\mathrm{out}}}\)
and multiplies as \(z = x_b W\) (row-vector convention).

---

## Layer

A **Layer** wraps a `Vector` of activations for one forward step.

**Responsibilities**
- Hold current activations (`neurons`)
- Apply element-wise activations
- Run `forward(weight)`: bias-augment \(\rightarrow\) matmul \(\rightarrow\) activation

For layer \(\ell\):

\[
x_b = [x;\;1],\qquad
z = x_b W,\qquad
y = f(z)
\]

where \(f\) is ReLU or identity (linear). Intermediate values \(x\), \(z\), \(y\)
are cached on the corresponding `Weight` for backpropagation.

---

## Weight

A **Weight** owns one dense connection: matrix \(W\), activation \(f\), and bias flag.

**Forward cache**
- \(x\): input activations
- \(x_b\): input with bias
- \(z\): pre-activation
- \(y\): post-activation

**Backward (one layer)**

\[
\delta z = f'(z)\odot \delta y,\qquad
\frac{\partial L}{\partial W} = x_b^\top \delta z,\qquad
\delta x = (\delta z\, W^\top)_{\text{drop bias}},\qquad
W \leftarrow W - \eta\,\frac{\partial L}{\partial W}
\]

---

## Model

`Model` stacks three dense layers:

\[
4 \xrightarrow{\mathrm{ReLU}} 8 \xrightarrow{\mathrm{ReLU}} 8 \xrightarrow{\mathrm{linear}} 3
\]

then applies **softmax** on the logits for classification.

**Loss (cross-entropy)** with one-hot targets \(y\) and probabilities \(p\):

\[
L = -\sum_i y_i \log(\max(p_i, \varepsilon))
\]

**Gradient of softmax + CE w.r.t. logits**

\[
\frac{\partial L}{\partial z} = p - y
\]

Training uses **stochastic / online SGD**: one sample update at a time.

---

## Data pipeline (Iris)

1. Map species names to class indices  
2. One-hot encode labels  
3. Standardize features **per column** (zero mean, unit variance)  
4. Train/test split  

Iris (Fisher’s flower dataset) is used as the demo classification task.

---

## Comparison

`PyTorch.ipynb` mirrors the same architecture in PyTorch for a Colab-side
comparison on Iris.
