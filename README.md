# 🧠 Hybrid CALM-on-Z

<div align="center">

**Brain-Inspired Continuous Language Model: Attention and Beyond**

*A hybrid implementation of Continuous Autoregressive Language Modeling (CALM) fused with State Space Models (SSM) and Hopfield Networks*

[![JAX](https://img.shields.io/badge/JAX-Accelerated-orange?style=flat&logo=python)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-Neural%20Networks-blue?style=flat)](https://github.com/google/flax)
[![TPU](https://img.shields.io/badge/TPU-Optimized-green?style=flat)](https://cloud.google.com/tpu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Overview](#-overview) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results)

</div>

---

## 🎯 Overview

Hybrid CALM-z is an adaptation of Shao et al. their CALM model. Instead of predicting discrete tokens one by one, it operates in a **continuous latent space**, predicting entire vectors that represent chunks of text.

### Why Hybrid?

While [CALM](https://arxiv.org/abs/2501.00000) focuses on efficiency via vectorization, this project explores **architectural efficiency**:

| Component | Purpose | Benefit |
|-----------|---------|---------|
| 🧩 **Token VAE** | Compresses K tokens → dense latent vector | Reduces generation steps |
| ⚡ **SSM (State Space Models)** | Efficient long-range processing | Linear scaling with sequence length |
| 🔗 **Hopfield Networks** | Associative memory retrieval | Biological plausibility + dense memory |
| 🎚️ **Gated Energy Head** | Refines noise → semantic vectors | Controlled generation |

---

## 🏗️ Architecture

### Two-Phase Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 1: Token VAE                       │
│  ┌────────┐    ┌─────────┐    ┌─────────┐    ┌────────────┐     │
│  │ Tokens │───▶│ Encoder │───▶│ Latent  │───▶│  Decoder   │     │
│  │ (K=4)  │    │  (MLP)  │    │ Space z │    │ (Logits)   │     │
│  └────────┘    └─────────┘    └─────────┘    └────────────┘     │
│                    ▲                                            │
│                    └─── VAE with KL regularization              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Phase 2: Hybrid CALM LM                       │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐     │
│  │ z_{t-1} │───▶│   SSM   │───▶│ Hopfield │───▶│  Gated   │     │
│  │         │    │ (Conv1D)│    │ (Memory) │    │  Energy  │     │
│  └─────────┘    └─────────┘    └──────────┘    └──────────┘     │
│                                                        │        │
│                                                        ▼        │
│                 ┌────────────────────────────────────────┐      │
│                 │  Loss = (2·d_fid - d_div) + λ·rf_loss  │      │
│                 └────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Loss Function

$$\mathcal{L} = \underbrace{(2 \cdot d_{\text{fid}} - d_{\text{div}})}_{\text{Energy Distance}} + \lambda \cdot \underbrace{(1 - \cos(\theta))}_{\text{Rectified Flow}}$$

---

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8+
pip install jax jaxlib flax optax transformers datasets
```

### For TPU (Recommended)

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Clone & Run

(in works)

```bash
git clone https://github.com/jada42/hybrid-calm-on-z.git
cd hybrid-calm-on-z
python hybrid_calm_z.py
```

---

## ⚙️ Configuration

Customize your experiment via the `Cfg` dataclass:

I set various configs for tests:

I recommend at least 30k steps for models around 90M params.

```python
@dataclass
class Cfg:
    # Architecture
    seq_z_steps: int = 64       # Latent sequence length
    K: int = 4                  # Tokens per chunk
    ssm_kernel: int = 7         # SSM convolution window
    hop_mem_slots: int = 64     # Hopfield memory slots
    
    # Training
    loss_type: str = "calm_rf"  # "calm" | "rf" | "calm_rf"
    rf_weight: float = 0.1      # RF loss weight
    batch_size: int = 8
    ae_steps: int = 2000        # Phase 1 steps
    lm_steps: int = 4000        # Phase 2 steps
```

---

## 📊 Results

### Training Dynamics

The model learns to:
1. ✅ **Compress** 4 tokens → 1 latent vector (Phase 1)
2. ✅ **Predict** next latent vectors using SSM + Hopfield (Phase 2)
3. ✅ **Balance** fidelity vs. diversity via energy distance
4. ✅ **Align** trajectories with Rectified Flow regularization

### Logged Metrics

- **Fidelity (`d_fid`)**: Distance to target distribution
- **Diversity (`d_div`)**: Inter-sample variance
- **RF Loss**: Trajectory alignment (1 - cosine similarity)
- **Gate Mean**: Balance between noise and prediction (0-1)
- **Fidelity per Step**: Where the model struggles in sequence
- **PPL**: According to Shao et al. not really applicable for these type of models as they predict latent vector space not tokens. 

Logs saved to `/content/ablation_logs/hybrid_calm_z_run.npz`

---

## 🎛️ Usage Examples

### Basic Training

```bash
python hybrid_calm_z.py
```

### Custom Configuration

```python
cfg = Cfg(
    loss_type="calm",        # Pure energy loss
    seq_z_steps=128,         # Longer sequences
    ssm_kernel=15,           # Larger receptive field
    hop_mem_slots=128,       # More memory
)
```

### Ablation Studies

```python
# Test different loss functions
for loss_type in ["calm", "rf", "calm_rf"]:
    cfg.loss_type = loss_type
    main()
```

---

## 🔬 New Elements

### 1. **Hybrid Architecture**
Combines the best of:
- **SSM**: Efficient O(n) sequence processing
- **Hopfield**: Content-addressable memory
- **Attention-free**: No quadratic bottleneck

### 2. **Gated Energy Head**
```python
z_pred = g · delta + (1 - g) · noise
```
- Learns to interpolate between predicted deltas and noise
- More stable than pure energy-based models

### 3. **Dual Objective**
- **Energy Distance**: Matches distribution (CALM)
- **Rectified Flow**: Aligns trajectories (RF)
- Combines both worlds

---

## 📈 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Compression Ratio** | 4:1 | 4 tokens → 1 vector |
| **Sequence Length** | 256 tokens | (64 steps × 4 tokens) |
| **Memory Slots** | 64 | Hopfield associative memory |
| **Training Time** | ~6K steps | 2K AE + 4K LM |
| **Compute** | TPU/GPU | BF16 precision |

---

## 📁 Project Structure

```
hybrid-calm-z/
├── hybrid_calm_z.py          # Main implementation
├── ablation_logs/            # Training logs
│   └── hybrid_calm_z_run.npz
├── README.md
└── LICENSE
```

---

## 🔮 Future Directions

- [ ] Multi-scale hierarchical VAE
- [ ] Adaptive K (variable chunk sizes)
- [ ] Dynamic Gating
- [ ] SegmentReasoner on Z from my Hybrid (HybridLLM)
- [ ] Conditional generation tasks
- [ ] Multimodal extensions
- [ ] Dual models (Predicting Latent space and Tokens selected with a Router)

---

## 📚 References

This work builds upon:

1. **CALM**: [Continuous Autoregressive Language Models (Shao et al., 2025)](https://arxiv.org/abs/2501.00000)
2. **SSM**: [Structured State Space Models (Gu et al.)](https://arxiv.org/abs/2111.00396)
3. **Hopfield Networks**: [Modern Hopfield Networks (Ramsauer et al.)](https://arxiv.org/abs/2008.02217)
4. **Rectified Flow**: [Flow Matching (Lipman et al.)](https://arxiv.org/abs/2210.02747)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Architectural improvements
- Training optimizations
- Evaluation benchmarks
- Documentation

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{hybrid-calm-z,
  author = {Your Name},
  title = {Hybrid CALM-on-Z: },
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/hybrid-calm-z}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built together with Claude & GPT5 and with Google Colab using JAX & Flax**

[⬆ Back to Top](#-hybrid-calm-z)

</div>
