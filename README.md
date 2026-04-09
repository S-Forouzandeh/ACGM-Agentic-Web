# ACGM: Adaptive Cross-Modal Graph Memory

**Paper:** *Task-Adaptive Retrieval over Agentic Multi-Modal Web Histories via Learned Graph Memory*
**Venue:** The 49th International ACM SIGIR Conference on Research and Development in Information Retrieval - SIGIR 2026 

---

## Overview

ACGM is a learned graph-memory retriever that constructs **task-adaptive relevance graphs** over multi-modal agent histories using policy-gradient optimization from downstream task success.

| Component | Paper Reference | File |
|---|---|---|
| RelevancePredictor `g_φ` (2-layer MLP) | Eq. 2 | `ACGM_Model.py` |
| Policy-gradient training + EMA baseline | Eq. 3 | `ACGMTrainer.stage2_train` |
| Modality-specific temporal decay `λ_m` | Eq. 4–5 | `ModalityTemporalDecay` |
| Decay regularization `L_decay` | Eq. 6 | `decay_regularization_loss` |
| Full objective `L = L_ret + 0.1·L_edge + 0.05·L_decay` | Eq. 7 | `ACGMTrainer.stage2_train` |
| Two-tier hierarchical retrieval O(log T) | Sec. 2.3 | `HierarchicalRetriever` |
| IR metrics (nDCG@10, MAP@10, MRR, Rec@10, Prec@10) | Sec. 3 | `IRMetrics` |

---

## Results (WebShop)

| Method | nDCG@10 | MAP@10 | MRR | Recall@10 | Prec@10 |
|---|---|---|---|---|---|
| GPT-4o | 73.4 | 64.9 | 80 | 83.8 | 81.5 |
| MAHA | 73.6 | 64.2 | 79 | 83.1 | 80.4 |
| **ACGM (ours)** | **82.7*** | **74.9*** | **88*** | **91.3*** | **89.2*** |

*p < 0.001 vs. second-best (Bonferroni-corrected)

---

## Installation

```bash
git clone https://anonymous.4open.science/r/ACGM_SIGIR-CB1B
cd ACGM_SIGIR
pip install -r requirements.txt
```

---

## Data

### WebShop
Download from the official repository:
```
https://github.com/princeton-nlp/WebShop
```
Place the file at the path set in `WEBSHOP_FILE` at the top of `ACGM_Model.py`:
```python
WEBSHOP_FILE = r"C:\Webshob_data\webshop_sft.txt"   # update this
```

### Human Annotation Data
The 500-pair temporal relevance annotation study (Fleiss' κ = 0.74) used to
initialize `λ_m` is included in:
```
annotations/decay_annotations_500pairs.json
```
Format:
```json
[
  {
    "obs_i": "click[Buy Now] Product: Blue T-Shirt ...",
    "obs_j": "click[Add to Cart] Product: Red T-Shirt ...",
    "modality": "visual",
    "delta_t": 3,
    "relevance_weight": 0.82,
    "annotator_agreement": 0.74
  },
  ...
]
```

---

## Usage

```bash
# Full training + evaluation
python ACGM_Model.py

# Outputs:
#   acgm_model.json       — saved model weights and procedures
#   acgm_results.json     — evaluation metrics (IR + task success)
```

---

## Architecture

```
ACGM
├── RelevancePredictor   g_φ         P(relevant(i,j)) = σ(g_φ(ẽ_i, ẽ_j, f_ij))
├── ModalityTemporalDecay            α_i^m ∝ exp(s_i^m/τ) · exp(-λ_m · Δt)
├── HierarchicalRetriever            Two-tier: flat (recent) + 4-ary tree (older)
├── ACGMTrainer
│   ├── stage1_train()               Pre-train g_φ with L_edge  (supervised)
│   └── stage2_train()               Fine-tune with L = L_ret + 0.1·L_edge + 0.05·L_decay
├── ProceduralMemorySystem           Bayesian Beta-distribution posteriors
├── BayesianProcedureSelector        EU = ρ·R_max − risk·(1−ρ)·C_fail + info
└── SemanticContextExtractor         Domain-agnostic ontology
```

**Learned decay rates** (initialized from human annotations, fine-tuned in Stage 2):

| Modality | GT (annotated) | Learned |
|---|---|---|
| Visual `λ_v` | 0.47 | ~0.45 |
| Knowledge `λ_k` | 0.23 | ~0.23 |
| Text `λ_x` | 0.11 | ~0.12 |

Visual decays **4.3× faster** than text, consistent with cognitive science literature.

---

## Training Protocol

| Stage | Steps | LR | Objective |
|---|---|---|---|
| Stage 1 | 50K | 1e-4 | `L_edge` only (supervised, ~18h on 8×A100) |
| Stage 2 | 50K | 1e-5 | Full `L = L_ret + 0.1·L_edge + 0.05·L_decay` (~22h) |

The EMA baseline `b_t ← 0.99·b_{t-1} + 0.01·R(τ)` reduces policy-gradient variance
by 38% compared to a zero baseline (measured as std of gradient norms, 1K episodes).

---

## Reproducibility

- Three-fold cross-validation with 95% confidence intervals
- Paired t-tests, Bonferroni-corrected
- Seeds: `torch.manual_seed(42)`, `np.random.seed(42)`
- Hardware: 8× NVIDIA A100 80GB

---

## Citation

```bibtex
@inproceedings{acgm2026sigir,
  title     = {Task-Adaptive Retrieval over Multi-Modal Web Histories
               via Learned Graph Memory},
  author    = {Anonymous},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference
               on Research and Development in Information Retrieval},
  year      = {2026}
}
```

---

## License

MIT License. See `LICENSE`.
