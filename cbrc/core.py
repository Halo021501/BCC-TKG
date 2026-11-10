from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

FEATURE_NAMES = (
    "confidence", "mapping_score", "template_agreement", "temporal_score", "rank_prior",
    "graph_gap", "graph_rank", "graph_entropy", "past_count", "past_freshness",
)
VARIANTS = ("cbrc", "graph_control", "positive", "no_semantic", "no_graph", "no_kl")
SCALES = (0.0, 0.10, 0.20, 0.35)


@dataclass(frozen=True)
class PublicQuery:
    known_entity_id: int
    oriented_relation_id: int
    timestamp: int

    @classmethod
    def from_mapping(cls, value: Mapping):
        required = {"known_entity_id", "oriented_relation_id", "timestamp"}
        if set(value) != required:
            raise ValueError("feature queries permit only known entity, oriented relation, timestamp")
        return cls(**{k: int(v) for k, v in value.items()})


class CausalCandidateHistory:
    """Exact oriented-triple counts; a timestamp barrier controls observed feedback."""

    def __init__(self, train_facts: Sequence[tuple[int, int, int, int]]):
        self.times = defaultdict(list)
        rel_times = defaultdict(set)
        for s, r, v, t in train_facts:
            self.times[s, r, v].append(t)
            rel_times[r].add(t)
        for times in self.times.values():
            times.sort()
        gaps_by_relation = {}
        all_gaps = []
        for r, values in rel_times.items():
            ordered = sorted(values)
            gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
            gaps_by_relation[r] = gaps
            all_gaps.extend(gaps)
        def median(values):
            x = sorted(values)
            n = len(x)
            return (x[(n - 1) // 2] + x[n // 2]) / 2 if n else 1.0
        fallback = median(all_gaps)
        self.tau = {r: max(1.0, median(g) if g else fallback) for r, g in gaps_by_relation.items()}
        self.fallback_tau = max(1.0, fallback)
        self.pending_time = None
        self.last_committed = None

    def begin_timestamp(self, timestamp: int):
        if self.pending_time is not None:
            raise ValueError("previous timestamp was not committed")
        if self.last_committed is not None and timestamp <= self.last_committed:
            raise ValueError("timestamps must advance strictly")
        self.pending_time = timestamp

    def features(self, query: PublicQuery, candidates: Sequence[int]):
        if self.pending_time != query.timestamp:
            raise ValueError("queries must be scored within a frozen timestamp barrier")
        result = []
        for v in candidates:
            times = self.times.get((query.known_entity_id, query.oriented_relation_id, v), [])
            n = bisect.bisect_left(times, query.timestamp)
            count = min(1.0, math.log1p(n) / math.log(11.0))
            fresh = math.exp(-(query.timestamp - times[n-1]) /
                             self.tau.get(query.oriented_relation_id, self.fallback_tau)) if n else 0.0
            result.append((count, fresh))
        return result

    def commit_timestamp(self, facts):
        if self.pending_time is None or any(f[3] != self.pending_time for f in facts):
            raise ValueError("commit only the fully scored current timestamp")
        for s, r, v, t in facts:
            bisect.insort(self.times[s, r, v], t)
        self.last_committed=self.pending_time
        self.pending_time = None


def validate_candidates(ids, mask, entities):
    if ids.ndim != 2 or mask.shape != ids.shape or mask.dtype != torch.bool:
        raise ValueError("candidate ids and Boolean mask must be [batch, candidates]")
    if mask.any() and ((ids[mask] < 0).any() or (ids[mask] >= entities).any()):
        raise ValueError("mapped candidate outside declared vocabulary")
    for row, valid in zip(ids, mask):
        if row[valid].unique().numel() != int(valid.sum()):
            raise ValueError("deduplicate mapped candidates before correction")


def candidate_features(base_logp, queries, ids, mask, semantic, history):
    """Inputs contain public queries only. No gold labels or filtered ranks."""
    validate_candidates(ids, mask, base_logp.shape[1])
    if len(queries) != len(base_logp) or semantic.shape != (*ids.shape, 5):
        raise ValueError("feature shapes disagree")
    if not torch.isfinite(base_logp).all() or not torch.isfinite(semantic).all():
        raise ValueError("nonfinite scores/features are not admissible")
    n, e = base_logp.shape
    safe = ids.clamp(0, e-1)
    scores = base_logp.gather(1, safe)
    # Sorted unfiltered scores avoid an N x E x candidate temporary tensor.
    sorted_scores = base_logp.sort(dim=1).values.contiguous()
    left = torch.searchsorted(sorted_scores, scores.contiguous(), right=False)
    right = torch.searchsorted(sorted_scores, scores.contiguous(), right=True)
    ranks = 1.0 + (e-right).float() + (right-left-1).float()/2.0
    gap = (scores-base_logp.amax(dim=1, keepdim=True)).clamp(-5.0, 0.0)/5.0
    rank_feature = torch.log1p(ranks)/math.log1p(e)
    normalized = base_logp - base_logp.logsumexp(dim=1, keepdim=True)
    entropy = -(normalized.exp()*normalized).sum(dim=1)/math.log(max(2, e))
    hist = [history.features(q, row.tolist()) for q, row in zip(queries, safe.cpu())]
    temporal = base_logp.new_tensor(hist).reshape(n, ids.shape[1], 2)
    z = torch.cat((semantic.clamp(0, 1), gap.unsqueeze(-1), rank_feature.unsqueeze(-1),
                   entropy[:, None, None].expand(n, ids.shape[1], 1), temporal), dim=-1)
    return z.masked_fill(~mask.unsqueeze(-1), 0.0).detach()


class CandidateCorrector(nn.Module):
    def __init__(self, variant="cbrc"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, mask, scale=0.35):
        if not 0 <= scale <= 0.35:
            raise ValueError("residual scale must be in [0, .35]")
        if z.shape[:-1] != mask.shape or z.shape[-1] != 10:
            raise ValueError("expected ten public candidate features")
        z = z.detach().clone()
        if self.variant in ("no_semantic", "graph_control"):
            z[..., :5] = 0
        if self.variant == "no_graph":
            z[..., 5:] = 0
        signed = torch.tanh(self.net(z).squeeze(-1))
        # ReLU(tanh) would have a dead derivative at zero initialization.
        # abs(tanh) retains exact zero output but also has a dead derivative.
        # A one-sided bound is imposed by a projected raw output; PyTorch's
        # clamp derivative at zero permits training from the identity.
        if self.variant == "positive":
            signed = signed.clamp(min=0)
        return scale * signed * mask.to(z.dtype)


def corrected_scores(base, ids, mask, delta):
    validate_candidates(ids, mask, base.shape[1])
    if delta.shape != ids.shape or not torch.isfinite(delta).all():
        raise ValueError("invalid corrections")
    out = base.clone()
    # masked padding never writes a correction to entity zero.
    row = torch.arange(len(base), device=base.device)[:, None].expand_as(ids)
    out[row[mask], ids[mask]] += delta[mask]
    return out


def sparse_objective(base, ids, mask, delta, truths, kl_weight=1.0):
    """Exact full-vocabulary multi-positive NLL and KL(p0 || corrected).

    Labels are passed only to the objective, never candidate_features/forward.
    Unmodified entities retain all their probability mass.
    """
    logp = base.double() - base.double().logsumexp(dim=1, keepdim=True)
    probs = logp.exp()
    cp = probs.gather(1, ids.clamp(0, base.shape[1]-1)) * mask
    d = delta.double()
    logz = torch.log1p((cp * torch.expm1(d)).sum(dim=1))
    numerators = []
    for i, positives in enumerate(truths):
        if not positives:
            raise ValueError("multi-positive labels cannot be empty")
        positive_ids = torch.tensor(sorted(set(positives)), device=base.device, dtype=torch.long)
        # Log-space avoids underflow even for extremely low probability targets.
        target_logp = logp[i, positive_ids]
        correction = torch.zeros_like(target_logp)
        for j in range(ids.shape[1]):
            if mask[i,j]:
                correction = correction + (positive_ids == ids[i,j]).to(d.dtype)*d[i,j]
        numerators.append(torch.logsumexp(target_logp+correction, dim=0))
    nll = logz-torch.stack(numerators)
    kl = logz-(cp*d).sum(dim=1)
    return (nll+kl_weight*kl).mean(), {"nll": nll.mean().detach(), "kl": kl.mean().detach()}


def average_rank(scores, target, known_truths):
    eligible = torch.ones_like(scores, dtype=torch.bool)
    excluded = [v for v in known_truths if v != target]
    if excluded:
        eligible[excluded] = False
    value = scores[target]
    higher = ((scores > value) & eligible).sum().item()
    other_equal = ((scores == value) & eligible).sum().item()-1
    return 1.0+higher+other_equal/2.0


def chronological_partitions(timestamps):
    times = sorted(set(int(t) for t in timestamps))
    if len(times) < 5:
        raise ValueError("at least five validation timestamps required for 60/20/20")
    a, b = int(len(times)*0.6), int(len(times)*0.8)
    return {t: "fit" if i < a else "select" if i < b else "audit"
            for i, t in enumerate(times)}
