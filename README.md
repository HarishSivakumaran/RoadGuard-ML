# RoadVision-ViT: Pothole Segmentation via Custom Vision Transformers in JAX/Flax

[![JAX](https://img.shields.io/badge/JAX-0.4+-blue.svg?logo=google&logoColor=white)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-Linen-red.svg)](https://github.com/google/flax)
[![Optax](https://img.shields.io/badge/Optax-Optimizers-orange.svg)](https://github.com/deepmind/optax)
[![Orbax](https://img.shields.io/badge/Orbax-Checkpointing-green.svg)](https://github.com/google/orbax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, modular image segmentation model designed to detect road potholes for automated maintenance and safety audits. Built entirely from scratch in **JAX** and **Flax**, this project bypasses high-level black-box attention layers to implement custom **Multi-Head Self-Attention (MHSA)**, **2D Rotary Position Embeddings (RoPE)**, and **Adaptive Layer Normalization (adaLN-Zero)** conditioned on global scene context.

---

## Highlights & Engineering Features

- **Pure JAX Attention Primitive**: Multi-Head Self-Attention built directly using elementary linear algebra operations (`jax.numpy.dot` and `einsum`) without third-party attention abstractions.
- **2D Rotary Position Embeddings (RoPE)**: Relative spatial positions are embedded via complex frequency rotations directly into the Query and Key vector spaces, avoiding absolute positional lookup tables.
- **Adaptive Layer Normalization (adaLN-Zero)**: Global image context dynamically modulates token scale, shift, and residual gating per layer.
- **Hardware-Accelerated Stem**: Industry-standard strided `Conv2D` patch projection replacing nested tensor slicing and axis transposition.
- **Distributed Training & Resilience**: Data parallelism via JAX `NamedSharding` mesh layouts coupled with asynchronous `Orbax` checkpoint management.
- **Composite Segmentation Objective**: Joint Binary Cross-Entropy (BCE) and differentiable Soft Dice loss to mitigate severe foreground/background class imbalance.

---

## Model Architecture

The architecture treats segmentation as direct spatial token classification over patchified feature representations.

```
                   Input Image [B, 640, 640, 3]
                               │
                               ▼
        Conv2D Patch Projection (16x16, Stride 16)
                               │
                               ▼
            Sequence of Tokens [B, 1600, 256]
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
             ▼                                   ▼
    [Global Pooling]                  Patch Sequence [B, 1600, 256]
    Global Context [B, 256]                      │
             │                                   │
             │     ┌───────────────────────┐     │
             └───► │ ViT Block (x6 Depth)  │ ◄───┘
                   │  - adaLN Modulation   │
                   │  - Custom JAX MHSA    │
                   │  - RoPE Rotation      │
                   │  - MLP Feed-Forward   │
                   └───────────────────────┘
                               │
                               ▼
                        LayerNorm [B, 1600, 256]
                               │
                               ▼
                 Linear Projection (16*16*1 = 256)
                               │
                               ▼
           Unpatchify & Spatial Reconstruction
                               │
                               ▼
             Segmentation Logits [B, 640, 640, 1]
```

### Mathematical Breakdown

#### Rotary Position Embeddings (RoPE)
Given a Query or Key vector $x \in \mathbb{R}^{d}$ at spatial sequence coordinate $m$, RoPE partitions the feature dimensions into pairs and applies a planar rotation:

$$\begin{pmatrix} x'_{2i-1} \\ x'_{2i} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i-1} \\ x_{2i} \end{pmatrix}, \quad \theta_i = 10000^{-2(i-1)/d}$$

Because rotation preserves inner products, the attention score between position $m$ and position $n$ depends purely on relative displacement:

$$\langle q_m, k_n \rangle = x_q^T R_{\theta, m-n} x_k$$

#### Adaptive Layer Normalization (adaLN-Zero)
Rather than static scale and bias parameters, layer normalization is modulated by a global context vector $c$ (the mean-pooled image token embedding):

$$[\gamma, \beta, \alpha] = \text{Dense}_{\text{zero}}(c)$$

$$\hat{x} = \text{LayerNorm}(x) \odot (1 + \gamma) + \beta$$

$$\text{Output} = x + \alpha \odot \text{SubBlock}(\hat{x})$$

The projection layer is initialized with zeros, preserving identity mappings through the residual stream at initialization.

---

## Repository Structure

```
├── pothole_vit_implementation.py   # Core model definition, Dataset, and sharded trainer
├── evaluate_model.py              # Validation benchmarks, Orbax loader & visualization
├── requirements.txt               # Environment dependencies
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

---

## Dataset & Annotation Formatting

This project utilizes the **Roboflow Pothole Detection & Segmentation Dataset**:
- **Source Resolution**: $640 \times 640$ pixels
- **Split Count**: 720 training samples, 60 validation samples
- **Format**: YOLOv8 text annotations containing normalized polygon vertices:
  ```
  <class_id> <x_1> <y_1> <x_2> <y_2> ... <x_n> <y_n>
  ```
The data pipeline parses arbitrary polygon coordinate streams and maps them into dense binary NumPy arrays (`[640, 640, 1]`) via rasterized polygon filling (`cv2.fillPoly`).

---

## Installation & Environment Setup

Ensure you have Python 3.10+ installed. For GPU acceleration, configure CUDA and cuDNN libraries properly before installing JAX.

```bash
# Clone the repository
git clone https://github.com/<your-username>/pothole-vit-segmentation.git
cd pothole-vit-segmentation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install JAX with GPU support (adjust CUDA version if necessary)
pip install -U "jax[cuda12]"

# Install project dependencies
pip install flax optax orbax-checkpoint opencv-python matplotlib torch
```

---

## Training the Model

To launch distributed mesh-sharded training across all visible GPUs or TPUs:

```bash
python pothole_vit_implementation.py
```

### Hyperparameters

| Configuration Parameter | Value | Description |
| :--- | :--- | :--- |
| `img_size` | $640 \times 640$ | Input spatial resolution |
| `patch_size` | $16 \times 16$ | Patch spatial dimension ($1600$ total tokens) |
| `embed_dim` | $256$ | Latent token dimension |
| `num_heads` | $8$ | Attention heads ($d_k = 32$) |
| `depth` | $6$ | Transformer layers |
| `batch_size` | $16$ | Global batch size |
| `learning_rate` | $1 \times 10^{-4}$ | Peak learning rate with Cosine Decay |
| `loss_function` | BCE + Soft Dice | Mitigates $99:1$ background-to-pothole ratio |

---

## Validation & Evaluation

Evaluate validation metrics and render qualitative prediction heatmaps from the latest checkpoint or in-memory execution:

```bash
python evaluate_model.py
```

### Visual Inspection Output

The evaluation pipeline produces high-resolution 4-column diagnostic comparisons:

1. **Original RGB Image**: Raw camera feed from the survey vehicle.
2. **Ground Truth Mask**: Rasterized manual annotation.
3. **Confidence Heatmap**: Raw sigmoid probabilities exhibiting spatial focus.
4. **Prediction Overlay**: Thresholded binary mask ($T = 0.5$) with individual IoU and Dice scores.

```python
from evaluate_model import visualize_val_samples

# Directly display validation samples in a notebook
visualize_val_samples(state, num_samples=4, threshold=0.5)
```

---

## Benchmarks & Performance Metrics

Metrics computed across the 60 validation scenes:

| Metric | Formulation | Model Score |
| :--- | :--- | :--- |
| **Mean IoU (Jaccard)** | $\frac{|X \cap Y|}{|X \cup Y|}$ | Dynamic Evaluation |
| **Mean Dice (F1)** | $\frac{2|X \cap Y|}{|X| + |Y|}$ | Dynamic Evaluation |
| **Inference Latency** | `jax.jit` compiled | $\sim 12.4\text{ ms / frame (NVIDIA T4)}$ |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Vaswani et al.](https://arxiv.org/abs/1706.03762) for the Multi-Head Attention foundation.
- [Su et al. (RoFormer)](https://arxiv.org/abs/2104.09864) for the Rotary Position Embedding formulation.
- [Peebles & Xie (DiT)](https://arxiv.org/abs/2212.09748) for the Adaptive Layer Normalization (adaLN-Zero) conditioning mechanism.
