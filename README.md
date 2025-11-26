# Hybrid CALM-on-Z

<div align="center">

**Brain-Inspired Continuous Language Model: Attention and Beyond**

*A hybrid implementation of Continuous Autoregressive Language Modeling (CALM) fused with State Space Models (SSM), Hopfield Networks, and Inference-Time Alignment.*

<p align="center">
  <a href="https://github.com/google/jax">
    <img src="https://img.shields.io/badge/JAX-Accelerated-orange?style=flat&logo=python&logoColor=white" alt="JAX">
  </a>
  <a href="https://github.com/google/flax">
    <img src="https://img.shields.io/badge/Flax-Neural%20Networks-blue?style=flat&logo=google&logoColor=white" alt="Flax">
  </a>
  <a href="https://cloud.google.com/tpu">
    <img src="https://img.shields.io/badge/TPU-Optimized-green?style=flat&logo=google-cloud&logoColor=white" alt="TPU">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Status-Experimental-red?style=flat" alt="Status">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
</p>

[Overview](#-overview) • [Architecture](#-architecture) • [Experimental: Latent-Alignment TTT](#-experimental-latent-ttt) • [Installation](#-installation) • [Results](#-results)


</div>

## 🎯 Overview
Hybrid CALM-z is an adaptation of the CALM model (Shao et al., 2025). Instead of predicting discrete tokens one by one, it operates in a continuous latent space, predicting entire vectors that represent chunks of text.

Why Hybrid?

While standard CALM focuses on efficiency via vectorization, this project explores architectural efficiency and inference-time plasticity:

Component	Purpose	Benefit
```
-  Token VAE	Compresses K tokens → dense latent vector	Reduces generation steps
- ⚡ SSM (State Space Models)	Efficient long-range processing	Linear scaling with sequence length
-  Hopfield Networks	Associative memory retrieval	Biological plausibility + dense memory
-  Gated Energy Head	Refines noise → semantic vectors	Controlled generation (Diffusion-like)
-  SEAL Alignment	New: Inference-Time Weight Updates	Align thought process via gradient descent
```

## 🏗️ Architecture
Three-Phase Pipeline

Phase 1: Token VAE (Compression)
```
┌─────────────────────────────────────────────────────────────────┐
│  ┌────────┐    ┌─────────┐    ┌─────────┐    ┌────────────┐     │
│  │ Tokens │───▶│ Encoder │───▶│ Latent  │───▶│  Decoder   │     │
│  │ (K=4)  │    │  (MLP)  │    │ Space z │    │ (Logits)   │     │
│  └────────┘    └─────────┘    └─────────┘    └────────────┘     │
│                    ▲                                            │
│                    └─── VAE with KL regularization              │
└─────────────────────────────────────────────────────────────────┘
Phase 2: Hybrid CALM LM (Trajectory Learning)

┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐     │
│  │ z_{t-1} │───▶│   SSM   │───▶│ Hopfield │───▶│  Gated   │     │
│  │         │    │ (Conv1D)│    │ (Memory) │    │  Energy  │     │
│  └─────────┘    └─────────┘    └──────────┘    └──────────┘     │
│                                                        │        │
│                 ┌────────────────────────────────────────┐      │
│                 │  Loss = (2·d_fid - d_div) + λ·rf_loss  │      │
│                 └────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
Phase 3: SEAL Inference (Latent Alignment)

┌─────────────────────────────────────────────────────────────────┐
│               SEAL: Self-Editing Alignment Layer                │
│                                                                 │
│  1. Hypothesize: Generate N latent trajectories (Thoughts)      │
│  2. Critique: Score w/ Differentiable Reward Model (Critic)     │
│  3. Rewire: Calculate ∇_θ and update weights (SGD)              │
│  4. Act: Regenerate output with updated brain                   │
└─────────────────────────────────────────────────────────────────┘
Hybrid Loss Function
```

```
Loss = Energy Distance (2⋅d_fid−d_div)+λ⋅ Rectified Flow (1−cos(θ))

​
```
 
​	
 
## Experimental: Latent TTT
Latent Test-Time Training (Latent-TTT) via the SEAL method.

Unlike "Pondering" (which reuses static weights), SEAL exploits the differentiability of the latent space to perform Thinking Process Optimization.

The model generates potential futures.

-> A (Simulated) Reward Model evaluates the vector trajectory.

--> The model runs Backpropagation on itself during inference.

---> It temporarily "learns" the concept needed for the specific prompt.

----> Result: In testing, this shifted the model from generating generic stop-words to concrete, concept-aligned entities (e.g., "Iowa", "Population") by optimizing against a target concept vector.

## 🚀 Installation
Prerequisites

Bash
# Python 3.8+
pip install jax jaxlib flax optax transformers datasets
For TPU (Recommended)

Bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
Clone & Run

Bash
git clone https://github.com/jada42/hybrid-calm-on-z.git
cd hybrid-calm-on-z
python hybrid_calm_z.py
⚙️ Configuration
Customize your experiment via the Cfg dataclass. Note: For TTT experiments, ensure compute_bf16=True to save memory.

Python
@dataclass
class Cfg:
    # Architecture
    seq_z_steps: int = 128      # Latent sequence length
    K: int = 4                  # Tokens per chunk
    
    # Training
    loss_type: str = "calm_rf"  # "calm" | "rf" | "calm_rf"
    rf_weight: float = 2.0      # RF loss weight
    
    # Latent TTT (SEAL)
    inner_steps: int = 3        # Gradient steps during inference
    temp_lr: float = 1e-2       # Learning rate for self-updates
## 📊 Results
Training Dynamics (1 Hour TPU Run)

The model self-discovers a diffusion-like generation strategy:

Gate Mean Convergence: The Gating mechanism starts at 1.0 (pure prediction) and converges to ~0.35. This proves the model learns to mix Autoregression with Noise (Diffusion) naturally.

Trajectory Alignment: The Rectified Flow loss (rf_loss) successfully aligns the vector field, allowing the SEAL inference step to optimize trajectories via gradient descent.

Logged Metrics

Fidelity (d_fid): Distance to target distribution.

Gate Mean: Balance between noise and prediction.

High (>0.9): Deterministic / Autoregressive

Low (<0.4): Stochastic / Diffusion-based

SEAL Delta: Semantic distance between Standard Output and Aligned Output.

Logs saved to /content/ablation_logs/hybrid_calm_z_run.npz

## 🎛️ Usage Examples
Basic Training

Bash
python hybrid_calm_z.py
Running SEAL Inference (Code Snippet)

Python
# Inside the inference loop:
 1. Generate Hypotheses & Score them
z_hypotheses = generate_candidates(...)
scores = reward_model(z_hypotheses)

 2. Self-Correction Loop
grads = jax.grad(loss_fn)(lm_state.params)
fast_params = p - TEMP_LR * grads

3. Final Prediction
final_z = predict(fast_params, ...)
🔬 Key Innovations
1. Hybrid Architecture

Combines the best of:

SSM: Efficient O(n) sequence processing.

Hopfield: Content-addressable memory.

2. Gated Energy Head

Python
z_pred = g · delta + (1 - g) · noise
Learns to interpolate between predicted deltas and noise, stabilizing generation in continuous space.

3. SEAL (Self-Editing Alignment Layer)

Enables the model to "rewire" its weights on the fly to satisfy alignment constraints that were not present during pre-training.

# 📈 Performance Characteristics
Metric	Value	Notes
Compression Ratio	4:1	4 tokens → 1 vector
Inference Mode	Hybrid	Auto-Regressive + Diffusion
Alignment	Dynamic	Weights update per prompt (TTT)
Compute	TPU/GPU	JAX/Flax Optimized

## 🔮 Future Directions
[ ] Scale to 1B+ parameters

[ ] Train a real differentiable Reward Model (Safety/Coherence)

[ ] Multi-scale hierarchical VAE

[ ] SegmentReasoner on Z from my Hybrid (HybridLLM)

[ ] Dual models (Router selecting between Latent prediction vs Token prediction)

## 📚 References
This work builds upon:

CALM: Continuous Autoregressive Language Models (Shao et al., 2025)

SSM: Structured State Space Models (Gu et al.)

Hopfield Networks: Modern Hopfield Networks (Ramsauer et al.)

Rectified Flow: Flow Matching (Lipman et al.)

TTT: Test-Time Training (Sun et al.)

## 🤝 Contributing
Contributions welcome! Areas of interest:

Architectural improvements

Training optimizations

Evaluation benchmarks (Latent Space PPL)

### 📝 Citation
If you use this code in your research, please cite:

Code-Snippet
@misc{hybrid-calm-z,
  author = {Jada42},
  title = {Hybrid CALM-on-Z: Latent Alignment via Test-Time Training},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jada42/hybrid-calm-on-z}
}
#### 📄 License
MIT License - see LICENSE file for details.

<div align="center">

Built together with Claude & GPT5 and with Google Colab using JAX & Flax

⬆ Back to Top

</div>
