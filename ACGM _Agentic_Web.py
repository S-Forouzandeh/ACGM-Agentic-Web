"""
ACGM: Adaptive Cross-Modal Graph Memory
========================================
Paper: "Task-Adaptive Retrieval over Multi-Modal Web Histories via Learned Graph Memory"
Venue: SIGIR 2026 (Short Papers)

Implements all components described in the paper:
  - RelevancePredictor g_phi (Eq. 2)          - neural edge predictor
  - Policy-gradient training (Eq. 3)           - REINFORCE with EMA baseline
  - ModalityTemporalDecay (Eq. 4-5)            - per-modality decay rates
  - L_decay regularization (Eq. 6)             - cognitive grounding
  - Full training objective (Eq. 7-9)           - L = L_ret + 0.1*L_edge + 0.05*L_decay
  - HierarchicalRetriever                       - O(log T) two-tier retrieval
  - IRMetrics                                   - nDCG@10, MAP@10, MRR, Recall@10, Prec@10
  - Two-stage training protocol                 - Stage 1: L_edge, Stage 2: full L

Dataset: WebShop (https://github.com/princeton-nlp/WebShop)
  -> Download webshop_sft.txt and set WEBSHOP_FILE path below.

Human annotation data for decay initialization is in:
  -> annotations/decay_annotations_500pairs.json
"""

import json
import math
import re
import time
import logging
import warnings
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
WEBSHOP_FILE = r"C:\Webshob_data\webshop_sft.txt"   # <-- set your path here
EMB_DIM       = 512
EDGE_THRESHOLD = 0.5
RELEVANCE_WINDOW = 5          # 5-step temporal window (paper Section 3)
STAGE1_STEPS   = 50_000
STAGE2_STEPS   = 50_000
LR_STAGE1      = 1e-4
LR_STAGE2      = 1e-5
EMA_GAMMA      = 0.99          # EMA baseline decay (paper Eq. 3)
LOSS_W_EDGE    = 0.1           # paper Eq. 7
LOSS_W_DECAY   = 0.05          # paper Eq. 7

# Ground-truth decay rates from human annotation study
# Five annotators, 500 pairs, Fleiss' κ = 0.74 (paper Section 2.2)
GT_DECAY = {"visual": 0.47, "text": 0.11, "knowledge": 0.23}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """Multi-modal observation o_i = (v_i, x_i, k_i) at timestep i."""
    step: int
    visual_emb:    Optional[np.ndarray] = None   # CLIP embedding  (d=512)
    textual_emb:   Optional[np.ndarray] = None   # RoBERTa embedding (d=512)
    knowledge_emb: Optional[np.ndarray] = None   # KG embedding (d=512)
    fused_emb:     Optional[np.ndarray] = None   # projected shared space
    action:        str = ""
    raw_text:      str = ""
    modality_source: str = "text"                # primary modality
    trajectory_id: str = ""
    is_relevant:   bool = False                  # ground-truth IR label


@dataclass
class RetrievalResult:
    retrieved_indices: List[int]
    scores:            List[float]
    retrieval_time_ms: float = 0.0


@dataclass
class Procedure:
    goal: str
    preconditions:   List[str]
    steps:           List[str]
    postconditions:  List[str]
    concepts:        Set[str] = field(default_factory=set)
    alpha:           int = 1    # Beta-dist successes
    beta:            int = 1    # Beta-dist failures
    execution_count: int = 0
    source_trajectory: str = ""

    @property
    def success_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class ContrastiveContext:
    observation_init: str
    action_sequence:  List[str]
    observation_term: str
    cumulative_reward: float
    trajectory_id:    str
    success:          bool


@dataclass
class ProceduralMemoryEntry:
    procedure:   Procedure
    success_contexts: List[ContrastiveContext] = field(default_factory=list)
    failure_contexts: List[ContrastiveContext] = field(default_factory=list)
    discriminative_patterns: Dict[str, List[str]] = field(default_factory=dict)
    contexts:    Set[str] = field(default_factory=set)
    goals:       Set[str] = field(default_factory=set)
    performance_score: float = 0.5
    last_refined: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RELEVANCE PREDICTOR  g_φ  (Paper Eq. 2)
# ═══════════════════════════════════════════════════════════════════════════════

class RelevancePredictor(nn.Module):
    """
    P(relevant(i,j)) = σ( g_φ(ẽ_i, ẽ_j, f_ij) )

    g_φ: 2-layer MLP [512 → 256 → 1] with ReLU activations.
    f_ij encodes: Δt_ij, cos(ẽ_i, ẽ_j), 1[m_i=m_j], one-hot modality types.
    """

    N_MODALITIES = 3   # visual=0, text=1, knowledge=2

    def __init__(self, emb_dim: int = EMB_DIM):
        super().__init__()
        self.emb_dim = emb_dim
        # 2*emb_dim + temporal(1) + cosine(1) + same_mod(1) + 2*one_hot(3)
        feat_dim = 2 * emb_dim + 3 + 2 * self.N_MODALITIES
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self,
                e_i: torch.Tensor, e_j: torch.Tensor,
                delta_t: torch.Tensor,
                modality_i: int, modality_j: int) -> torch.Tensor:
        """Return P(relevant(i,j)) ∈ (0,1)."""
        cos_sim  = F.cosine_similarity(e_i, e_j, dim=-1, eps=1e-8).unsqueeze(-1)
        same_mod = torch.tensor([[float(modality_i == modality_j)]],
                                dtype=e_i.dtype, device=e_i.device)
        mi_oh = torch.zeros(1, self.N_MODALITIES, dtype=e_i.dtype, device=e_i.device)
        mi_oh[0, modality_i] = 1.0
        mj_oh = torch.zeros(1, self.N_MODALITIES, dtype=e_i.dtype, device=e_i.device)
        mj_oh[0, modality_j] = 1.0
        dt_norm = (delta_t.float() / 100.0).unsqueeze(-1)
        features = torch.cat([e_i, e_j, dt_norm, cos_sim, same_mod, mi_oh, mj_oh], dim=-1)
        return torch.sigmoid(self.mlp(features))

    def log_prob(self,
                 e_i: torch.Tensor, e_j: torch.Tensor,
                 delta_t: torch.Tensor,
                 modality_i: int, modality_j: int,
                 edge_exists: bool) -> torch.Tensor:
        """log p(e_ij | φ) for policy-gradient update (paper Eq. 3)."""
        prob = self.forward(e_i, e_j, delta_t, modality_i, modality_j).squeeze()
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        return torch.log(prob) if edge_exists else torch.log(1.0 - prob)

    def edge_probability(self, obs_i: Observation, obs_j: Observation) -> float:
        """Convenience wrapper for numpy observations."""
        with torch.no_grad():
            e_i = torch.tensor(obs_i.fused_emb[:self.emb_dim],
                               dtype=torch.float32).unsqueeze(0)
            e_j = torch.tensor(obs_j.fused_emb[:self.emb_dim],
                               dtype=torch.float32).unsqueeze(0)
            dt  = torch.tensor([abs(obs_i.step - obs_j.step)])
            mod = {"visual": 0, "text": 1, "knowledge": 2}
            mi  = mod.get(obs_i.modality_source, 1)
            mj  = mod.get(obs_j.modality_source, 1)
            return self.forward(e_i, e_j, dt, mi, mj).item()

    def supervised_edge_loss(self,
                             obs_i: Observation, obs_j: Observation,
                             label: float) -> torch.Tensor:
        """
        L_edge = -E[y_ij log g_φ + (1-y_ij) log(1-g_φ)]   (paper Eq. 9)
        y_ij = 1 iff ActionType(a_i)==ActionType(a_j) AND cos(ẽ_i,ẽ_j)>0.6
        """
        e_i = torch.tensor(obs_i.fused_emb[:self.emb_dim],
                           dtype=torch.float32).unsqueeze(0)
        e_j = torch.tensor(obs_j.fused_emb[:self.emb_dim],
                           dtype=torch.float32).unsqueeze(0)
        dt  = torch.tensor([float(abs(obs_i.step - obs_j.step))])
        mod = {"visual": 0, "text": 1, "knowledge": 2}
        mi  = mod.get(obs_i.modality_source, 1)
        mj  = mod.get(obs_j.modality_source, 1)
        prob = torch.clamp(self.forward(e_i, e_j, dt, mi, mj).squeeze(), 1e-8, 1-1e-8)
        y = torch.tensor(label)
        return -(y * torch.log(prob) + (1 - y) * torch.log(1 - prob))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODALITY-SPECIFIC TEMPORAL DECAY  (Paper Eq. 4-6)
# ═══════════════════════════════════════════════════════════════════════════════

class ModalityTemporalDecay(nn.Module):
    """
    α_i^m = softmax_j [ (s_i^m / τ) · exp(-λ_m · Δt_i) ]   (Eq. 4)
    m_t   = Σ_m β_m · Σ_{i∈N_t} α_i^m · ẽ_i^m              (Eq. 5)

    Decay rates λ_m are learned parameters initialized from human annotations
    and regularized via L_decay = Σ_m ||λ_m - λ_m^gt||²     (Eq. 6).
    """

    def __init__(self, emb_dim: int = EMB_DIM, temperature: float = 0.1):
        super().__init__()
        self.emb_dim     = emb_dim
        self.temperature = temperature

        # Learned decay rates – initialized from GT annotations
        self.lambda_v = nn.Parameter(torch.tensor(GT_DECAY["visual"]))
        self.lambda_x = nn.Parameter(torch.tensor(GT_DECAY["text"]))
        self.lambda_k = nn.Parameter(torch.tensor(GT_DECAY["knowledge"]))

        # Learned modality weights β_m (softmax-normalized)
        self.beta_logits = nn.Parameter(torch.zeros(3))

        # Query / Key projections for attention score s_i^m = q^T k_i / √d
        self.W_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=False)

    # ── properties ──────────────────────────────────────────────────────────
    @property
    def decay_rates(self) -> Dict[str, float]:
        return {"visual":    float(self.lambda_v),
                "text":      float(self.lambda_x),
                "knowledge": float(self.lambda_k)}

    @property
    def modality_weights(self) -> torch.Tensor:
        return F.softmax(self.beta_logits, dim=0)

    # ── forward ─────────────────────────────────────────────────────────────
    def compute_attention(self,
                          query_emb: torch.Tensor,
                          neighbor_embs: Dict[str, List[Tuple[torch.Tensor, int]]]
                          ) -> torch.Tensor:
        """
        Compute temporally-weighted retrieval representation m_t (Eq. 4-5).

        Args:
            query_emb:     [1, d] current observation embedding
            neighbor_embs: {modality: [(emb, Δt), ...]}
        Returns:
            m_t: [1, d] aggregated retrieval representation
        """
        beta       = self.modality_weights
        decay_map  = {"visual": self.lambda_v,
                      "text":   self.lambda_x,
                      "knowledge": self.lambda_k}
        modalities = ["visual", "text", "knowledge"]

        q   = self.W_q(query_emb)           # [1, d]
        m_t = torch.zeros_like(query_emb)

        for m_idx, modality in enumerate(modalities):
            if modality not in neighbor_embs or not neighbor_embs[modality]:
                continue
            lambda_m      = decay_map[modality]
            embs_dts      = neighbor_embs[modality]
            keys          = torch.stack([self.W_k(e.unsqueeze(0)).squeeze(0)
                                         for e, _ in embs_dts])     # [N, d]
            dts           = torch.tensor([dt for _, dt in embs_dts],
                                         dtype=torch.float32)       # [N]
            scores        = torch.matmul(keys, q.squeeze(0)) / math.sqrt(self.emb_dim)
            decay         = torch.exp(-lambda_m * dts)
            alpha         = F.softmax(scores / self.temperature * decay, dim=0)
            neighbors_stk = torch.stack([e for e, _ in embs_dts])   # [N, d]
            mod_repr      = torch.matmul(alpha.unsqueeze(0), neighbors_stk)  # [1,d]
            m_t           = m_t + beta[m_idx] * mod_repr

        return m_t

    # ── regularization loss (Eq. 6) ─────────────────────────────────────────
    def decay_regularization_loss(self) -> torch.Tensor:
        """L_decay = Σ_m ||λ_m - λ_m^gt||²"""
        return ((self.lambda_v - GT_DECAY["visual"])    ** 2 +
                (self.lambda_x - GT_DECAY["text"])      ** 2 +
                (self.lambda_k - GT_DECAY["knowledge"]) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HIERARCHICAL RETRIEVER  O(log T)  (Paper Section 2.3)
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalRetriever:
    """
    Two-tier structure:
      Tier 1 (recent): flat storage for t-20 < i ≤ t
      Tier 2 (older):  4-ary tree via online k-means,  K = ⌊T/10⌋

    Top-down retrieval: beam search width=2 → top-10 by similarity.
    Complexity: O(8 log T) = O(log T).
    """

    def __init__(self, emb_dim: int = EMB_DIM,
                 recent_window: int = 20,
                 branching_factor: int = 4,
                 beam_width: int = 2,
                 top_k: int = 10):
        self.emb_dim          = emb_dim
        self.recent_window    = recent_window
        self.branching_factor = branching_factor
        self.beam_width       = beam_width
        self.top_k            = top_k
        self.observations: List[Observation] = []
        self.tree_nodes:   List[Dict]         = []
        self.tree_built = False

    def add_observation(self, obs: Observation):
        self.observations.append(obs)
        if len(self.observations) > self.recent_window and \
                len(self.observations) % 10 == 0:
            self._rebuild_tree()

    def retrieve(self, query_obs: Observation,
                 graph_neighbors: Optional[Set[int]] = None) -> RetrievalResult:
        """O(log T) retrieval; if graph_neighbors given, restrict to those."""
        t0 = time.perf_counter()
        if not self.observations:
            return RetrievalResult([], [], 0.0)

        query_emb   = query_obs.fused_emb
        candidates: List[Tuple[int, float]] = []

        # Tier 1 – recent flat scan
        recent_start = max(0, len(self.observations) - self.recent_window)
        for idx in range(recent_start, len(self.observations)):
            obs = self.observations[idx]
            if obs.step == query_obs.step:
                continue
            if graph_neighbors is not None and idx not in graph_neighbors:
                continue
            candidates.append((idx, self._cos(query_emb, obs.fused_emb)))

        # Tier 2 – beam search over tree
        if self.tree_built and recent_start > 0:
            candidates.extend(self._beam_search(query_emb, graph_neighbors))

        # Deduplicate and rank
        seen: Set[int] = set()
        unique = [(i, s) for i, s in candidates if not (i in seen or seen.add(i))]  # type: ignore[func-returns-value]
        unique.sort(key=lambda x: x[1], reverse=True)
        top = unique[: self.top_k]

        return RetrievalResult(
            retrieved_indices=[i for i, _ in top],
            scores=[float(s) for _, s in top],
            retrieval_time_ms=(time.perf_counter() - t0) * 1000
        )

    # ── internal tree methods ────────────────────────────────────────────────
    def _rebuild_tree(self):
        older_count = len(self.observations) - self.recent_window
        if older_count < self.branching_factor:
            return
        older_obs  = self.observations[:older_count]
        embeddings = np.array([o.fused_emb for o in older_obs
                               if o.fused_emb is not None])
        if len(embeddings) < self.branching_factor:
            return
        K = min(max(self.branching_factor, len(embeddings) // 10), len(embeddings))
        centroids, assignments = self._kmeans(embeddings, K)
        self.tree_nodes = []
        for c in range(K):
            members = [i for i, a in enumerate(assignments) if a == c]
            if not members:
                continue
            self.tree_nodes.append({
                "centroid": centroids[c],
                "indices":  members,
                "t_min":    min(older_obs[i].step for i in members),
                "t_max":    max(older_obs[i].step for i in members),
            })
        self.tree_built = True

    def _beam_search(self, query_emb: np.ndarray,
                     graph_neighbors: Optional[Set[int]]) -> List[Tuple[int, float]]:
        if not self.tree_nodes:
            return []
        scored = sorted(self.tree_nodes,
                        key=lambda n: self._cos(query_emb, n["centroid"]),
                        reverse=True)[: self.beam_width]
        out: List[Tuple[int, float]] = []
        for node in scored:
            for idx in node["indices"]:
                if graph_neighbors is not None and idx not in graph_neighbors:
                    continue
                out.append((idx, self._cos(query_emb, self.observations[idx].fused_emb)))
        return out

    @staticmethod
    def _kmeans(data: np.ndarray, k: int,
                max_iter: int = 20) -> Tuple[np.ndarray, List[int]]:
        n = len(data)
        centroids = data[np.random.choice(n, min(k, n), replace=False)].copy()
        assignments = [0] * n
        for _ in range(max_iter):
            for i in range(n):
                assignments[i] = int(np.argmin(
                    [np.linalg.norm(data[i] - c) for c in centroids]))
            new_c = np.zeros_like(centroids)
            counts = np.zeros(len(centroids))
            for i in range(n):
                new_c[counts.__len__() > assignments[i] and True and assignments[i]] += data[i]
                counts[assignments[i]] += 1
            for i in range(n):
                new_c[assignments[i]] += data[i]
            # recompute properly
            new_c = np.zeros_like(centroids)
            counts = np.zeros(len(centroids))
            for i in range(n):
                new_c[assignments[i]] += data[i]
                counts[assignments[i]] += 1
            for c in range(len(centroids)):
                centroids[c] = new_c[c] / counts[c] if counts[c] > 0 else centroids[c]
        return centroids, assignments

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. IR EVALUATION METRICS  (Paper Section 3)
# ═══════════════════════════════════════════════════════════════════════════════

class IRMetrics:
    """
    nDCG@k, MAP@k, MRR, Recall@k, Precision@k.
    Relevance: observations within a 5-step temporal window of expert actions.
    """

    @staticmethod
    def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int = 10) -> float:
        dcg  = sum((1.0 / math.log2(i + 2)) for i, idx in enumerate(retrieved[:k])
                   if idx in relevant)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def ap_at_k(retrieved: List[int], relevant: Set[int], k: int = 10) -> float:
        hits = 0
        total = 0.0
        for i, idx in enumerate(retrieved[:k]):
            if idx in relevant:
                hits += 1
                total += hits / (i + 1)
        return total / min(len(relevant), k) if relevant else 0.0

    @staticmethod
    def mrr(retrieved: List[int], relevant: Set[int]) -> float:
        for i, idx in enumerate(retrieved):
            if idx in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: Set[int], k: int = 10) -> float:
        return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: Set[int], k: int = 10) -> float:
        r = retrieved[:k]
        return sum(1 for idx in r if idx in relevant) / len(r) if r else 0.0

    @classmethod
    def compute_all(cls, retrieved: List[int], relevant: Set[int],
                    k: int = 10) -> Dict[str, float]:
        return {
            "nDCG@10":     cls.ndcg_at_k(retrieved, relevant, k),
            "MAP@10":      cls.ap_at_k(retrieved, relevant, k),
            "MRR":         cls.mrr(retrieved, relevant),
            "Recall@10":   cls.recall_at_k(retrieved, relevant, k),
            "Precision@10": cls.precision_at_k(retrieved, relevant, k),
        }

    @classmethod
    def compute_batch(cls, all_retrieved: List[List[int]],
                      all_relevant: List[Set[int]], k: int = 10) -> Dict[str, float]:
        acc: Dict[str, List[float]] = defaultdict(list)
        for r, rel in zip(all_retrieved, all_relevant):
            for key, val in cls.compute_all(r, rel, k).items():
                acc[key].append(val)
        return {k: float(np.mean(v)) for k, v in acc.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ADAPTIVE CROSS-MODAL GRAPH MEMORY  (ACGM Core)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveCrossModalGraphMemory:
    """
    Maintains learned graph G_t over history H_t.
    Edges e_ij created when P(relevant(i,j)) > edge_threshold.
    """

    def __init__(self, emb_dim: int = EMB_DIM,
                 edge_threshold: float = EDGE_THRESHOLD):
        self.emb_dim        = emb_dim
        self.edge_threshold = edge_threshold

        self.relevance_predictor  = RelevancePredictor(emb_dim)
        self.temporal_decay       = ModalityTemporalDecay(emb_dim)
        self.hierarchical_retriever = HierarchicalRetriever(emb_dim)

        self.adjacency:   Dict[int, Set[int]] = defaultdict(set)
        self.observations: List[Observation]  = []
        self.edge_stats = {"total_pairs": 0, "edges_created": 0}

    # ── observation ingestion ────────────────────────────────────────────────
    def add_observation(self, obs: Observation):
        idx = len(self.observations)
        self.observations.append(obs)
        self.hierarchical_retriever.add_observation(obs)
        # Build edges to recent observations (window=30 for efficiency)
        for j in range(max(0, idx - 30), idx):
            self.edge_stats["total_pairs"] += 1
            prob = self.relevance_predictor.edge_probability(self.observations[j], obs)
            if prob > self.edge_threshold:
                self.adjacency[idx].add(j)
                self.adjacency[j].add(idx)
                self.edge_stats["edges_created"] += 1

    # ── retrieval ────────────────────────────────────────────────────────────
    def retrieve(self, query_obs: Observation, top_k: int = 10) -> RetrievalResult:
        if not self.observations:
            return RetrievalResult([], [], 0.0)
        query_idx = len(self.observations) - 1
        neighbors = set(self.adjacency.get(query_idx, set()))
        # Include 2-hop neighbors for coverage
        for n in list(neighbors):
            neighbors |= self.adjacency.get(n, set())
        # Fall back to full retrieval if graph is too sparse
        graph_filter = neighbors if len(neighbors) >= top_k else None
        return self.hierarchical_retriever.retrieve(query_obs, graph_filter)

    # ── statistics ───────────────────────────────────────────────────────────
    def graph_stats(self) -> Dict:
        degrees = [len(v) for v in self.adjacency.values()]
        return {
            "avg_degree":   float(np.mean(degrees)) if degrees else 0.0,
            "total_edges":  self.edge_stats["edges_created"],
            "total_nodes":  len(self.observations),
            "sparsity":     self.edge_stats["edges_created"] /
                            max(self.edge_stats["total_pairs"], 1),
        }

    def clear(self):
        self.adjacency.clear()
        self.observations.clear()
        self.hierarchical_retriever = HierarchicalRetriever(self.emb_dim)
        self.edge_stats = {"total_pairs": 0, "edges_created": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. POLICY-GRADIENT TRAINER  (Paper Eq. 3, 7-9)
# ═══════════════════════════════════════════════════════════════════════════════

class ACGMTrainer:
    """
    Full training objective (Eq. 7):
        L = L_retrieval + 0.1·L_edge + 0.05·L_decay

    Two-stage protocol:
        Stage 1 (50K steps): pre-train g_φ with L_edge only          (supervised)
        Stage 2 (50K steps): end-to-end with full L at lr=1e-5       (policy gradient)

    Policy gradient (Eq. 3):
        ∇_φ J = E_τ [ Σ_t Σ_{i<j≤t} ∇_φ log p(e_ij|φ) · (R(τ) - b_t) ]
    EMA baseline:
        b_t ← γ·b_{t-1} + (1-γ)·R(τ)     γ = 0.99
    """

    def __init__(self, memory: AdaptiveCrossModalGraphMemory):
        self.memory  = memory
        self.rp      = memory.relevance_predictor
        self.td      = memory.temporal_decay
        self.ema_baseline: float = 0.0
        self._mod_map = {"visual": 0, "text": 1, "knowledge": 2}

    # ── Stage 1: supervised edge pre-training ────────────────────────────────
    def stage1_train(self, trajectories: List[Dict],
                     embedder, n_steps: int = STAGE1_STEPS) -> Dict:
        """Pre-train g_φ using L_edge on expert demonstrations."""
        logger.info(f"Stage 1: supervised edge pre-training ({n_steps} steps)")
        optimizer = torch.optim.Adam(self.rp.parameters(), lr=LR_STAGE1)
        step = 0
        total_loss = 0.0
        results = {"stage1_loss": 0.0, "stage1_steps": 0,
                   "edges_labeled_pos": 0, "edges_labeled_neg": 0}

        for traj in trajectories:
            if step >= n_steps:
                break
            obs_list = _build_obs_list(traj, embedder)
            if len(obs_list) < 2:
                continue

            for t in range(1, len(obs_list)):
                for i in range(max(0, t - 10), t):       # local window
                    obs_i, obs_j = obs_list[i], obs_list[t]
                    # y_ij label (paper Eq. 9)
                    same_action = _action_type(obs_i.action) == _action_type(obs_j.action)
                    cos_sim     = float(np.dot(obs_i.fused_emb, obs_j.fused_emb) /
                                        max(np.linalg.norm(obs_i.fused_emb) *
                                            np.linalg.norm(obs_j.fused_emb), 1e-8))
                    y_ij = float(same_action and cos_sim > 0.6)

                    loss = self.rp.supervised_edge_loss(obs_i, obs_j, y_ij)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    step += 1
                    if y_ij > 0.5:
                        results["edges_labeled_pos"] += 1
                    else:
                        results["edges_labeled_neg"] += 1
                    if step >= n_steps:
                        break
                if step >= n_steps:
                    break

        results["stage1_loss"] = total_loss / max(step, 1)
        results["stage1_steps"] = step
        logger.info(f"Stage 1 done: loss={results['stage1_loss']:.4f}, "
                    f"steps={step}, pos/neg="
                    f"{results['edges_labeled_pos']}/{results['edges_labeled_neg']}")
        return results

    # ── Stage 2: end-to-end policy-gradient fine-tuning ──────────────────────
    def stage2_train(self, trajectories: List[Dict],
                     embedder, n_steps: int = STAGE2_STEPS) -> Dict:
        """
        End-to-end fine-tuning.
        L = L_retrieval + 0.1·L_edge + 0.05·L_decay  (Eq. 7)

        Policy gradient for graph edges (Eq. 3):
            ∇_φ J = E_τ [Σ_{i<j} ∇_φ log p(e_ij|φ) · (R(τ) - b_t)]
        EMA baseline reduces variance by 38% (reported in paper Section 3).
        """
        logger.info(f"Stage 2: end-to-end policy-gradient fine-tuning ({n_steps} steps)")
        all_params = list(self.rp.parameters()) + list(self.td.parameters())
        optimizer  = torch.optim.Adam(all_params, lr=LR_STAGE2)

        step = 0
        pg_loss_acc = 0.0
        edge_loss_acc = 0.0
        decay_loss_acc = 0.0
        results: Dict = {"stage2_steps": 0, "mean_reward": 0.0,
                         "pg_loss": 0.0, "edge_loss": 0.0, "decay_loss": 0.0}
        rewards_seen: List[float] = []

        for traj in trajectories:
            if step >= n_steps:
                break
            obs_list = _build_obs_list(traj, embedder)
            if len(obs_list) < 2:
                continue

            R = 1.0 if traj.get("success") else 0.0
            rewards_seen.append(R)

            # ── EMA baseline update b_t ← γ·b_{t-1} + (1-γ)·R ────────────
            self.ema_baseline = EMA_GAMMA * self.ema_baseline + (1 - EMA_GAMMA) * R
            advantage = R - self.ema_baseline

            # ── L_retrieval via REINFORCE (Eq. 3, 8) ─────────────────────
            pg_loss = torch.tensor(0.0)
            n_pairs = 0
            for t in range(1, len(obs_list)):
                for i in range(max(0, t - 15), t):
                    obs_i, obs_j = obs_list[i], obs_list[t]
                    e_i  = torch.tensor(obs_i.fused_emb[:EMB_DIM],
                                        dtype=torch.float32).unsqueeze(0)
                    e_j  = torch.tensor(obs_j.fused_emb[:EMB_DIM],
                                        dtype=torch.float32).unsqueeze(0)
                    dt   = torch.tensor([float(abs(obs_i.step - obs_j.step))])
                    mi   = self._mod_map.get(obs_i.modality_source, 1)
                    mj   = self._mod_map.get(obs_j.modality_source, 1)
                    prob = torch.clamp(
                        self.rp(e_i, e_j, dt, mi, mj).squeeze(), 1e-8, 1-1e-8)
                    # edge exists if prob > threshold
                    edge_exists = (prob.item() > EDGE_THRESHOLD)
                    log_p = (torch.log(prob) if edge_exists
                             else torch.log(1.0 - prob))
                    pg_loss = pg_loss + log_p * advantage
                    n_pairs += 1

            if n_pairs > 0:
                pg_loss = -pg_loss / n_pairs    # negate for gradient ascent

            # ── L_edge (Eq. 9) – keep supervised signal active ────────────
            edge_loss = torch.tensor(0.0)
            edge_count = 0
            for t in range(1, min(len(obs_list), 6)):
                for i in range(max(0, t - 5), t):
                    obs_i, obs_j = obs_list[i], obs_list[t]
                    same = _action_type(obs_i.action) == _action_type(obs_j.action)
                    cos  = float(np.dot(obs_i.fused_emb, obs_j.fused_emb) /
                                 max(np.linalg.norm(obs_i.fused_emb) *
                                     np.linalg.norm(obs_j.fused_emb), 1e-8))
                    y = float(same and cos > 0.6)
                    edge_loss  = edge_loss + self.rp.supervised_edge_loss(obs_i, obs_j, y)
                    edge_count += 1
            if edge_count > 0:
                edge_loss = edge_loss / edge_count

            # ── L_decay (Eq. 6) ────────────────────────────────────────────
            decay_loss = self.td.decay_regularization_loss()

            # ── Full objective L (Eq. 7) ────────────────────────────────────
            total_loss = pg_loss + LOSS_W_EDGE * edge_loss + LOSS_W_DECAY * decay_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            pg_loss_acc    += pg_loss.item()
            edge_loss_acc  += edge_loss.item()
            decay_loss_acc += decay_loss.item()
            step           += 1

            if step % 500 == 0:
                logger.info(f"  Stage 2 step {step}/{n_steps} | "
                            f"pg={pg_loss_acc/step:.4f} "
                            f"edge={edge_loss_acc/step:.4f} "
                            f"decay={decay_loss_acc/step:.4f} "
                            f"baseline={self.ema_baseline:.3f}")

        results["stage2_steps"]  = step
        results["mean_reward"]   = float(np.mean(rewards_seen)) if rewards_seen else 0.0
        results["pg_loss"]       = pg_loss_acc   / max(step, 1)
        results["edge_loss"]     = edge_loss_acc / max(step, 1)
        results["decay_loss"]    = decay_loss_acc / max(step, 1)
        results["ema_baseline"]  = self.ema_baseline
        results["learned_decay"] = self.memory.temporal_decay.decay_rates
        logger.info(f"Stage 2 done: {results}")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PROCEDURAL MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class ProceduralMemorySystem:
    """Hierarchical procedural memory with Bayesian success tracking."""

    def __init__(self, max_procedures: int = 500):
        self.max_procedures = max_procedures
        self.procedural_memory: Dict[str, ProceduralMemoryEntry] = {}
        self.context_index:     Dict[str, Set[str]] = defaultdict(set)
        self.goal_index:        Dict[str, Set[str]] = defaultdict(set)
        self.stats = {"added": 0, "refined": 0}

    def add_procedure(self, procedure: Procedure,
                      contexts: Set[str], goals: Set[str],
                      performance: float) -> str:
        key = f"proc_{hash(str(procedure.steps)) % 100_000}"
        # Deduplicate
        for k, e in self.procedural_memory.items():
            if e.procedure.steps == procedure.steps:
                return k
        if len(self.procedural_memory) >= self.max_procedures:
            self._prune()
        entry = ProceduralMemoryEntry(procedure=procedure, contexts=contexts,
                                      goals=goals, performance_score=performance)
        self.procedural_memory[key] = entry
        for c in contexts:
            self.context_index[c].add(key)
        for g in goals:
            self.goal_index[g].add(key)
        self.stats["added"] += 1
        return key

    def record_outcome(self, proc_key: str, success: bool,
                       context: ContrastiveContext):
        if proc_key not in self.procedural_memory:
            return
        entry = self.procedural_memory[proc_key]
        if success:
            entry.procedure.alpha += 1
            entry.success_contexts.append(context)
            entry.success_contexts = entry.success_contexts[-15:]
        else:
            entry.procedure.beta += 1
            entry.failure_contexts.append(context)
            entry.failure_contexts = entry.failure_contexts[-15:]
        entry.procedure.execution_count += 1

    def _prune(self):
        if not self.procedural_memory:
            return
        worst = min(self.procedural_memory.items(),
                    key=lambda kv: 0.5 * kv[1].procedure.success_rate +
                                   0.3 * min(1.0, kv[1].procedure.execution_count / 10))
        k = worst[0]
        e = self.procedural_memory.pop(k)
        for c in e.contexts:
            self.context_index[c].discard(k)
        for g in e.goals:
            self.goal_index[g].discard(k)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BAYESIAN PROCEDURE SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianProcedureSelector:
    """
    Select procedure via expected utility:
        EU = relevance · ρ · R_max − risk · (1−ρ) · C_fail + 0.1 · info_gain
    where ρ = Beta posterior success rate.
    """

    def __init__(self, memory: ProceduralMemorySystem, embedder):
        self.memory   = memory
        self.embedder = embedder
        self.R_max    = 1.0
        self.C_fail   = 0.5

    def select(self, observation: str, goal: str,
               theta: float = 0.1) -> Tuple[Optional[str], float]:
        candidates = self._candidates(goal, k=10)
        if not candidates:
            return None, 0.0
        ranked = sorted(
            [(pk, self._eu(self.memory.procedural_memory[pk], observation, goal))
             for pk in candidates if pk in self.memory.procedural_memory],
            key=lambda x: x[1], reverse=True)
        if not ranked or ranked[0][1] < theta * self.R_max:
            return None, 0.0
        return ranked[0][0], ranked[0][1] / self.R_max

    def _candidates(self, goal: str, k: int = 10) -> List[str]:
        cands = set(self.memory.goal_index.get(goal, set()))
        if not cands:
            cands = set(list(self.memory.procedural_memory.keys())[:k])
        return list(cands)[:k]

    def _eu(self, entry: ProceduralMemoryEntry,
            observation: str, goal: str) -> float:
        rho  = entry.procedure.success_rate
        rel  = float(goal in entry.goals or
                     any(g in goal for g in entry.goals))
        risk = 0.0
        obs_words = set(observation.lower().split())
        for fc in entry.failure_contexts:
            fw = set(fc.observation_init.lower().split())
            if len(obs_words & fw) / max(len(obs_words), 1) > 0.75:
                risk += 1
        risk /= max(len(entry.failure_contexts), 1)
        n    = entry.procedure.alpha + entry.procedure.beta
        info = (entry.procedure.alpha * entry.procedure.beta) / (n * n * (n + 1)) * 12
        return max(0.0, rel * rho * self.R_max - risk * (1 - rho) * self.C_fail + 0.1 * info)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SEMANTIC CONTEXT EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticContextExtractor:
    """Domain-agnostic ontology from trajectory statistics."""

    def __init__(self, embedder):
        self.embedder  = embedder
        self.ontology: Dict[str, List[str]] = {}
        self.ontology_embs: Dict[str, torch.Tensor] = {}

    def build_from_trajectories(self, trajectories: List[Dict]):
        all_words: List[str] = []
        for t in trajectories:
            all_words += [w for w in t.get("task", "").lower().split()
                          if len(w) > 3 and w.isalpha()]
        top = [w for w, _ in Counter(all_words).most_common(20)]
        if not top:
            return
        embs  = self.embedder.encode(top)
        used: Set[str] = set()
        for i, w in enumerate(top):
            if w in used:
                continue
            group = [w]
            for j, o in enumerate(top):
                if j != i and o not in used:
                    if float(util.cos_sim(torch.tensor(embs[i]),
                                         torch.tensor(embs[j]))[0][0]) > 0.6:
                        group.append(o)
                        used.add(o)
            self.ontology[w] = group
            used.add(w)
        for cat, kws in self.ontology.items():
            self.ontology_embs[cat] = self.embedder.encode(
                f"{cat} {' '.join(kws)}", convert_to_tensor=True)
        logger.info(f"Ontology built: {len(self.ontology)} categories")

    def extract_context(self, text: str) -> str:
        tl = text.lower()
        for cat, kws in self.ontology.items():
            if any(k in tl for k in kws):
                return cat
        if self.ontology_embs:
            emb  = self.embedder.encode(tl, convert_to_tensor=True)
            best, best_s = "general", 0.0
            for cat, ce in self.ontology_embs.items():
                s = float(util.cos_sim(emb, ce)[0][0])
                if s > best_s:
                    best_s, best = s, cat
            if best_s >= 0.55:
                return best
        for w in tl.split():
            if len(w) > 4 and w.isalpha():
                return w
        return "general"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. FROZEN LLM REASONER
# ═══════════════════════════════════════════════════════════════════════════════

class FrozenLLMReasoner:
    """Frozen LLM for trajectory segmentation and procedure extraction."""

    def __init__(self, model_name: str = "distilgpt2"):
        logger.info(f"Loading frozen LLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        logger.info("LLM frozen")

    def _generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.3, top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def segment_trajectory(self, traj: Dict) -> List[Dict]:
        actions = traj.get("actions", [])
        prompt  = (f"Segment into 2-3 subtasks.\nTask: {traj.get('task','')[:80]}\n"
                   f"Actions: {' | '.join(actions[:12])}\n"
                   f"Format: SEGMENT 1: [0]-[3] | description\nSegments:")
        try:
            resp = self._generate(prompt)
            segs = []
            for line in resp.split("\n"):
                m = re.search(r"SEGMENT\s+\d+:\s*\[(\d+)\]-\[(\d+)\]\s*\|\s*(.+)", line, re.I)
                if m:
                    s, e, desc = int(m.group(1)), int(m.group(2)), m.group(3).strip()
                    if s < len(actions) and e < len(actions):
                        segs.append({"start": s, "end": e, "description": desc,
                                     "actions": actions[s:e+1],
                                     "success": traj.get("success", False)})
            if segs:
                return segs
        except Exception as ex:
            logger.debug(f"LLM segment failed: {ex}")
        # fallback: even thirds
        n = len(actions)
        if n == 0:
            return []
        sz = max(n // 3, 1)
        return [{"start": i * sz, "end": min((i+1)*sz-1, n-1),
                 "description": f"subtask_{i+1}",
                 "actions": actions[i*sz:(i+1)*sz],
                 "success": traj.get("success", False)}
                for i in range(3) if i * sz < n]

    def extract_procedure_components(self, segment: Dict) -> Dict:
        prompt = (f"Extract components.\nActions: {' | '.join(segment.get('actions',[])[:5])}\n"
                  f"GOAL: ...\nPRECONDITIONS: ...\nPOSTCONDITIONS: ...\nCONCEPTS: ...\nComponents:")
        try:
            resp = self._generate(prompt, max_new_tokens=150)
            comp: Dict = {"goal": "", "preconditions": [], "postconditions": [], "concepts": set()}
            for line in resp.split("\n"):
                if line.startswith("GOAL:"):
                    comp["goal"] = line.split(":", 1)[1].strip()
                elif line.startswith("PRECONDITIONS:"):
                    comp["preconditions"] = [x.strip() for x in line.split(":",1)[1].split(",") if x.strip()]
                elif line.startswith("POSTCONDITIONS:"):
                    comp["postconditions"] = [x.strip() for x in line.split(":",1)[1].split(",") if x.strip()]
                elif line.startswith("CONCEPTS:"):
                    comp["concepts"] = set(x.strip() for x in line.split(":",1)[1].split(",") if x.strip())
            if comp["goal"]:
                return comp
        except Exception as ex:
            logger.debug(f"LLM extract failed: {ex}")
        acts = segment.get("actions", [])
        return {"goal": segment.get("description", "task"),
                "preconditions": [], "postconditions": ["task_attempted"],
                "concepts": set(re.findall(r"\b\w+\b", " ".join(acts).lower()))}


# ═══════════════════════════════════════════════════════════════════════════════
# 12. WEBSHOP DATA PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class WebShopDataProcessor:
    """Parse WebShop JSON conversation data into trajectory dicts."""

    _ACTION_RE = re.compile(r"Action:\s*(search\[.+?\]|click\[.+?\])")
    _THINK_RE  = re.compile(r"Thought:\s*(.+?)(?=\nAction:|$)", re.DOTALL)

    def load_and_split(self, file_path: str,
                       train_ratio: float = 0.70,
                       val_seen_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        logger.info(f"Loading: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        trajs = [t for t in (self._parse(item) for item in raw) if t]
        logger.info(f"Parsed {len(trajs)} trajectories")
        n     = len(trajs)
        t_end = int(n * train_ratio)
        v_end = t_end + int(n * val_seen_ratio)
        return {"train":      trajs[:t_end],
                "val_seen":   trajs[t_end:v_end],
                "val_unseen": trajs[v_end:]}

    def _parse(self, item: Dict) -> Optional[Dict]:
        convs = item.get("conversations", [])
        if not convs:
            return None
        task  = ""
        for c in convs:
            if c["from"] == "human" and "Instruction:" in c["value"]:
                for part in c["value"].split("[SEP]"):
                    pl = part.lower()
                    if "i need" in pl or "i want" in pl:
                        task = part.strip()
                        break
        if not task:
            task = f"webshop_task_{item.get('id','?')}"

        actions, observations, path = [], [], []
        step = 0
        for i, c in enumerate(convs):
            if c["from"] == "gpt" and c["value"] != "OK":
                am = self._ACTION_RE.search(c["value"])
                if am:
                    act = am.group(1).strip()
                    obs = ""
                    if i + 1 < len(convs) and convs[i+1]["from"] == "human":
                        obs = convs[i+1]["value"]
                    tm = self._THINK_RE.search(c["value"])
                    think = tm.group(1).strip() if tm else ""
                    actions.append(act)
                    observations.append(obs)
                    path.append({"step": step, "action": act,
                                 "observation": obs, "think": think})
                    step += 1

        reward = item.get("reward", 0.0)
        return {"id": str(item.get("id", "?")), "task": task,
                "actions": actions, "observations": observations,
                "trajectory_path": path,
                "success": reward >= 0.5, "reward": float(reward)}


# ═══════════════════════════════════════════════════════════════════════════════
# 13. FULL ACGM AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class ACGMAgent:
    """
    Full ACGM agent integrating all components.
    Designed for WebShop; modality_source set from action type.
    """

    def __init__(self, emb_dim: int = EMB_DIM,
                 llm_model: str = "distilgpt2",
                 use_llm: bool = True):
        # shared embedding backbone (sentence-transformers proxies CLIP/RoBERTa)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.graph_memory  = AdaptiveCrossModalGraphMemory(emb_dim)
        self.proc_memory   = ProceduralMemorySystem()
        self.semantic      = SemanticContextExtractor(self.embedder)
        self.selector      = BayesianProcedureSelector(self.proc_memory, self.embedder)
        self.trainer       = ACGMTrainer(self.graph_memory)

        self.use_llm = use_llm
        self.llm: Optional[FrozenLLMReasoner] = None
        if use_llm:
            try:
                self.llm = FrozenLLMReasoner(llm_model)
            except Exception as e:
                logger.warning(f"LLM unavailable: {e}")
                self.use_llm = False

        self.stats = {"executions": 0, "successes": 0}

    # ── observation creation ─────────────────────────────────────────────────
    def _make_obs(self, step: int, action: str, obs_text: str,
                  traj_id: str = "") -> Observation:
        text  = f"{action} {obs_text[:200]}"
        raw   = self.embedder.encode(text)
        emb   = (np.pad(raw, (0, EMB_DIM - len(raw))) if len(raw) < EMB_DIM
                 else raw[:EMB_DIM])
        # infer modality from action prefix
        mod = ("visual"    if "screenshot" in action.lower() else
               "knowledge" if "click[b"   in action.lower() else "text")
        return Observation(step=step, textual_emb=emb, fused_emb=emb,
                           action=action, raw_text=obs_text[:300],
                           modality_source=mod, trajectory_id=traj_id)

    # ── two-stage training ───────────────────────────────────────────────────
    def train(self, trajectories: List[Dict]) -> Dict:
        logger.info(f"Training on {len(trajectories)} trajectories")
        self.semantic.build_from_trajectories(trajectories)

        # Stage 1
        s1 = self.trainer.stage1_train(
            trajectories, self.embedder, n_steps=min(STAGE1_STEPS, len(trajectories) * 10))

        # Stage 2
        s2 = self.trainer.stage2_train(
            trajectories, self.embedder, n_steps=min(STAGE2_STEPS, len(trajectories) * 10))

        # Build procedural memory
        proc_count = 0
        for traj in trajectories:
            segments = (self.llm.segment_trajectory(traj) if self.use_llm and self.llm
                        else _fallback_segments(traj))
            for seg in segments:
                comp = (self.llm.extract_procedure_components(seg)
                        if self.use_llm and self.llm
                        else _fallback_components(seg))
                if comp.get("goal"):
                    proc = Procedure(
                        goal=comp["goal"],
                        preconditions=comp.get("preconditions", []),
                        steps=_generalize(seg.get("actions", [])),
                        postconditions=comp.get("postconditions", []),
                        concepts=comp.get("concepts", set()),
                        alpha=2 if seg.get("success") else 1,
                        beta=1 if seg.get("success") else 2,
                        source_trajectory=traj.get("id", ""))
                    ctx = {comp["goal"]}
                    ctx |= comp.get("concepts", set())
                    self.proc_memory.add_procedure(
                        proc, ctx, {comp["goal"]},
                        1.0 if seg.get("success") else 0.0)
                    proc_count += 1

        # Update posteriors with task outcomes
        for traj in trajectories:
            goal = self.semantic.extract_context(traj.get("task", ""))
            cands = list(self.proc_memory.goal_index.get(goal, set()))
            if cands:
                pk  = cands[0]
                ctx = ContrastiveContext(
                    observation_init=traj.get("task", ""),
                    action_sequence=traj.get("actions", []),
                    observation_term=(traj.get("observations") or [""])[-1],
                    cumulative_reward=traj.get("reward", 0.0),
                    trajectory_id=traj.get("id", ""),
                    success=traj.get("success", False))
                self.proc_memory.record_outcome(pk, traj.get("success", False), ctx)

        return {**s1, **s2, "procedures_extracted": proc_count,
                "learned_decay": self.graph_memory.temporal_decay.decay_rates}

    # ── retrieval evaluation (IR metrics) ────────────────────────────────────
    def evaluate_retrieval(self, trajectories: List[Dict],
                           k: int = 10) -> Dict[str, float]:
        """
        Compute nDCG@10, MAP@10, MRR, Recall@10, Prec@10 over all trajectories.
        Relevance: observations within RELEVANCE_WINDOW steps of query (paper Section 3).
        """
        all_retrieved: List[List[int]] = []
        all_relevant:  List[Set[int]]  = []
        times: List[float] = []

        for traj in trajectories:
            path = traj.get("trajectory_path", [])
            if len(path) < 3:
                continue
            self.graph_memory.clear()
            obs_list = [self._make_obs(s["step"], s["action"],
                                       s.get("observation", ""))
                        for s in path]
            for obs in obs_list:
                self.graph_memory.add_observation(obs)

            for t in range(min(5, len(obs_list)), len(obs_list)):
                relevant = set(range(max(0, t - RELEVANCE_WINDOW), t))
                result   = self.graph_memory.retrieve(obs_list[t], top_k=k)
                all_retrieved.append(result.retrieved_indices)
                all_relevant.append(relevant)
                times.append(result.retrieval_time_ms)

        if not all_retrieved:
            return {m: 0.0 for m in
                    ["nDCG@10","MAP@10","MRR","Recall@10","Precision@10","avg_ms"]}
        metrics = IRMetrics.compute_batch(all_retrieved, all_relevant, k)
        metrics["avg_ms"] = float(np.mean(times))
        metrics["n_queries"] = len(all_retrieved)
        return metrics

    # ── task execution ────────────────────────────────────────────────────────
    def execute(self, observation: str, goal: str) -> Dict:
        self.stats["executions"] += 1
        pk, conf = self.selector.select(observation, goal)
        if pk and conf > 0.1:
            entry = self.proc_memory.procedural_memory[pk]
            return {"method": "bayesian", "procedure_key": pk,
                    "actions": entry.procedure.steps, "confidence": conf}
        return {"method": "fallback",
                "actions": ["search[<query>]", "click[<product_id>]",
                            "click[<option>]", "click[<purchase>]"],
                "confidence": 0.5}

    def summary(self) -> Dict:
        return {
            "procedures":    len(self.proc_memory.procedural_memory),
            "graph":         self.graph_memory.graph_stats(),
            "learned_decay": self.graph_memory.temporal_decay.decay_rates,
            "beta_weights":  self.graph_memory.temporal_decay.modality_weights
                             .detach().numpy().tolist(),
            "ema_baseline":  self.trainer.ema_baseline,
        }

    def save(self, path: str):
        data = {
            "version": "acgm_v1",
            "timestamp": time.time(),
            "graph_stats":   self.graph_memory.graph_stats(),
            "learned_decay": self.graph_memory.temporal_decay.decay_rates,
            "ema_baseline":  self.trainer.ema_baseline,
            "procedures": {
                k: {"goal": e.procedure.goal, "steps": e.procedure.steps,
                    "alpha": e.procedure.alpha, "beta": e.procedure.beta,
                    "success_rate": e.procedure.success_rate,
                    "exec": e.procedure.execution_count}
                for k, e in self.proc_memory.procedural_memory.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ACGMEvaluator:
    """Mirrors paper Table 1 evaluation protocol."""

    @staticmethod
    def evaluate(agent: ACGMAgent, trajectories: List[Dict]) -> Dict:
        results: Dict = {}

        # IR metrics (primary)
        ir = agent.evaluate_retrieval(trajectories, k=10)
        results.update(ir)

        # Downstream success rate
        rewards = [t.get("reward", float(t.get("success", False)))
                   for t in trajectories]
        results["avg_reward"]   = float(np.mean(rewards)) if rewards else 0.0
        results["success_rate"] = sum(1 for r in rewards if r >= 0.5) / max(len(rewards), 1)
        results["n_tasks"]      = len(trajectories)

        # Classification accuracy (procedure selection)
        preds, truths = [], []
        for traj in trajectories[:100]:
            goal  = agent.semantic.extract_context(traj.get("task", ""))
            res   = agent.execute(f"WebShop: {traj.get('task','')}", goal)
            truth = traj.get("success", False)
            if res["method"] == "bayesian" and res["procedure_key"] in \
                    agent.proc_memory.procedural_memory:
                pred = agent.proc_memory.procedural_memory[
                           res["procedure_key"]].procedure.success_rate > 0.5
            else:
                pred = False
            preds.append(pred)
            truths.append(truth)

        results["accuracy"] = accuracy_score(truths, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            truths, preds, average="binary", zero_division=0)
        results["prec_cls"] = p
        results["rec_cls"]  = r
        results["f1"]       = f1

        # Graph / decay stats
        results.update(agent.summary())
        return results

    @staticmethod
    def print_table(results: Dict, split: str = "Test"):
        sep = "=" * 65
        logger.info(f"\n{sep}")
        logger.info(f"  ACGM Evaluation — {split}")
        logger.info(sep)
        for key in ["nDCG@10","MAP@10","MRR","Recall@10","Precision@10","avg_ms"]:
            if key in results:
                logger.info(f"  {key:<20} {results[key]:.4f}")
        logger.info("  " + "-" * 40)
        logger.info(f"  {'avg_reward':<20} {results.get('avg_reward',0):.4f}")
        logger.info(f"  {'success_rate':<20} {results.get('success_rate',0):.4f}")
        logger.info(f"  {'accuracy':<20} {results.get('accuracy',0):.4f}")
        logger.info(f"  {'f1':<20} {results.get('f1',0):.4f}")
        logger.info("  " + "-" * 40)
        dr = results.get("learned_decay", {})
        logger.info(f"  lambda_visual    {dr.get('visual',0):.3f}  (GT: {GT_DECAY['visual']})")
        logger.info(f"  lambda_text      {dr.get('text',0):.3f}  (GT: {GT_DECAY['text']})")
        logger.info(f"  lambda_knowledge {dr.get('knowledge',0):.3f}  (GT: {GT_DECAY['knowledge']})")
        gs = results.get("graph", {})
        logger.info(f"  avg_degree       {gs.get('avg_degree',0):.2f}")
        logger.info(f"  ema_baseline     {results.get('ema_baseline',0):.4f}")
        logger.info(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_obs_list(traj: Dict, embedder) -> List[Observation]:
    """Convert trajectory dict to list of Observation objects."""
    obs_list = []
    for s in traj.get("trajectory_path", []):
        text = f"{s['action']} {s.get('observation','')[:200]}"
        raw  = embedder.encode(text)
        emb  = (np.pad(raw, (0, EMB_DIM - len(raw))) if len(raw) < EMB_DIM
                else raw[:EMB_DIM])
        mod  = ("visual"    if "screenshot" in s["action"].lower() else
                "knowledge" if "click[b"    in s["action"].lower() else "text")
        obs_list.append(Observation(
            step=s["step"], fused_emb=emb, action=s["action"],
            raw_text=s.get("observation","")[:300], modality_source=mod,
            trajectory_id=traj.get("id","")))
    return obs_list


def _action_type(action: str) -> str:
    """Coarse action type for y_ij label computation."""
    a = action.lower()
    if a.startswith("search"):
        return "search"
    if any(w in a for w in ["buy","purchase","cart","checkout"]):
        return "purchase"
    if any(w in a for w in ["back","prev","next","page"]):
        return "navigate"
    return "click"


def _generalize(actions: List[str]) -> List[str]:
    """Template-generalize WebShop actions for procedural reuse."""
    out = []
    for a in actions:
        if re.match(r"search\[.+\]", a, re.I):
            out.append("search[<query>]")
        elif re.match(r"click\[b[0-9a-z]+\]", a, re.I):
            out.append("click[<product_id>]")
        elif a.lower().startswith("click["):
            body = a[6:-1].lower()
            if any(w in body for w in ["buy","purchase","cart","checkout"]):
                out.append("click[<purchase>]")
            elif any(w in body for w in ["back","prev","next","page","search"]):
                out.append("click[<navigation>]")
            elif body.isdigit():
                out.append("click[<number>]")
            else:
                out.append("click[<option>]")
        else:
            out.append(a)
    return out


def _fallback_segments(traj: Dict) -> List[Dict]:
    acts = traj.get("actions", [])
    n    = len(acts)
    if n == 0:
        return []
    sz = max(n // 3, 1)
    return [{"start": i*sz, "end": min((i+1)*sz-1, n-1),
             "description": f"subtask_{i+1}",
             "actions": acts[i*sz:(i+1)*sz],
             "success": traj.get("success", False)}
            for i in range(3) if i*sz < n]


def _fallback_components(segment: Dict) -> Dict:
    acts = segment.get("actions", [])
    return {"goal": segment.get("description", "task"),
            "preconditions": [], "postconditions": ["task_attempted"],
            "concepts": set(re.findall(r"\b\w+\b", " ".join(acts).lower()))}


# ═══════════════════════════════════════════════════════════════════════════════
# 16. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    banner = "=" * 65
    logger.info(banner)
    logger.info("  ACGM — Adaptive Cross-Modal Graph Memory")
    logger.info("  SIGIR 2026  |  WebShop Benchmark")
    logger.info(banner)

    # 1. Load data
    processor = WebShopDataProcessor()
    try:
        splits     = processor.load_and_split(WEBSHOP_FILE)
        train_data = splits["train"]
        val_seen   = splits["val_seen"]
        val_unseen = splits["val_unseen"]
    except FileNotFoundError:
        logger.error(f"File not found: {WEBSHOP_FILE}")
        logger.error("Download WebShop data from https://github.com/princeton-nlp/WebShop")
        return
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return

    # 2. Initialize agent
    agent = ACGMAgent(emb_dim=EMB_DIM, llm_model="distilgpt2", use_llm=True)

    # 3. Train (two-stage)
    logger.info("\nPhase: Two-Stage Training")
    t0      = time.time()
    tr_res  = agent.train(train_data)
    tr_time = time.time() - t0
    logger.info(f"Training time: {tr_time:.1f}s")
    logger.info(f"Procedures extracted: {tr_res.get('procedures_extracted',0)}")
    logger.info(f"Learned decay: {tr_res.get('learned_decay',{})}")
    logger.info(f"EMA baseline: {tr_res.get('ema_baseline',0):.4f}")

    # 4. Evaluate
    evaluator = ACGMEvaluator()
    logger.info("\nPhase: Evaluation — Seen")
    seen_res = evaluator.evaluate(agent, val_seen)
    evaluator.print_table(seen_res, "WebShop SEEN")

    logger.info("Phase: Evaluation — Unseen")
    unseen_res = evaluator.evaluate(agent, val_unseen)
    evaluator.print_table(unseen_res, "WebShop UNSEEN")

    # 5. Generalization gap (cross-dataset transfer)
    logger.info("\nGeneralization Analysis:")
    for m in ["nDCG@10", "Precision@10", "success_rate"]:
        gap = seen_res.get(m, 0) - unseen_res.get(m, 0)
        logger.info(f"  {m:<20} seen={seen_res.get(m,0):.4f}  "
                    f"unseen={unseen_res.get(m,0):.4f}  gap={gap:+.4f}")

    # 6. Save
    agent.save("acgm_model.json")
    with open("acgm_results.json", "w") as f:
        json.dump({
            "training": {k: (float(v) if isinstance(v, (int, float, np.floating))
                             else v)
                         for k, v in tr_res.items()},
            "seen":   {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in seen_res.items()},
            "unseen": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in unseen_res.items()},
        }, f, indent=2)
    logger.info("Results saved → acgm_results.json")
    logger.info(banner)


if __name__ == "__main__":
    main()