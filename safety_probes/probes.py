"""The probe grid: MM (difference-of-means), LR (logistic regression), and the attention probe.

`ui_probe.py` depends on this module: `rebuild_detection_probe` constructs all three classes.
"""

from __future__ import annotations

import einops
import torch as t
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from tqdm import tqdm

from . import runtime


class MMProbe(t.nn.Module):
    """Difference-of-means probe [GoT §5.1]. Plain theta_mm; exposes .direction for steering."""

    def __init__(self, direction: Tensor):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)

    def forward(self, x): return t.sigmoid(self.score(x))
    def score(self, x): return x.float() @ self.direction  # raw projection; .float() because cached
                                                           # activations are fp16, direction is fp32
    def pred(self, x): return (self.score(x) > 0).float()

    @staticmethod
    def from_data(acts: Tensor, labels: Tensor) -> "MMProbe":
        acts = acts.float()
        direction = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
        return MMProbe(direction)


class LRProbe(t.nn.Module):
    """Logistic regression on standardized activations [Apollo §3.1].

    The `scaler_mean` / `scaler_scale` buffers are load-bearing: without them the probe scores
    raw activations it was never standardised for and returns garbage *silently* rather than
    erroring. They are registered buffers so `state_dict()` carries them through export.
    """

    def __init__(self, d_in, scaler_mean=None, scaler_scale=None):
        super().__init__()
        self.net = t.nn.Linear(d_in, 1, bias=False)
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _norm(self, x):
        if self.scaler_mean is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x): return t.sigmoid(self.net(self._norm(x.float())).squeeze(-1))
    def score(self, x): return self.net(self._norm(x.float())).squeeze(-1)
    def pred(self, x): return (self.score(x) > 0).float()

    @property
    def direction(self) -> Tensor:
        # Direction in RAW activation space (undo the standardizer) so steering matches other probes.
        w = self.net.weight.data[0]
        return w / self.scaler_scale if self.scaler_scale is not None else w

    @staticmethod
    def from_data(acts: Tensor, labels: Tensor, C: float = 0.1) -> "LRProbe":
        X, y = acts.cpu().float().numpy(), labels.cpu().float().numpy()
        scaler = StandardScaler().fit(X)
        lr = LogisticRegression(C=C, random_state=runtime.SEED, fit_intercept=False, max_iter=1000)
        lr.fit(scaler.transform(X), y)
        probe = LRProbe(
            acts.shape[-1],
            scaler_mean=t.tensor(scaler.mean_, dtype=t.float32),
            scaler_scale=t.tensor(scaler.scale_, dtype=t.float32),
        )
        probe.net.weight.data[0] = t.tensor(lr.coef_[0], dtype=t.float32)
        return probe


def attention_probe_forward(x, mask, W_q, W_out, b_out, scale):
    attn_logits = einops.einsum(x.float(), W_q, "b s d, d n -> b s n") / scale
    attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    attn_weights = attn_logits.softmax(dim=1)
    context = einops.einsum(attn_weights, x.float(), "b s n, b s d -> b n d").flatten(start_dim=1)
    logits = einops.einsum(context, W_out, "b h, h one -> b one").squeeze(-1) + b_out.squeeze()
    return logits, attn_weights


class AttentionProbe(t.nn.Module):
    """Attention-pooling probe [HS §2.1]. Learned pooling -> LR-style, no closed-form MM variant."""

    def __init__(self, d_model, n_heads=1):
        super().__init__()
        self.n_heads, self.scale = n_heads, d_model**0.5
        self.W_q = t.nn.Parameter(t.empty(d_model, n_heads))
        self.W_out = t.nn.Parameter(t.empty(n_heads * d_model, 1))
        self.b_out = t.nn.Parameter(t.zeros(1))
        t.nn.init.normal_(self.W_q, std=d_model**-0.5)
        t.nn.init.normal_(self.W_out, std=(n_heads * d_model) ** -0.5)

    def forward(self, x, mask):
        return attention_probe_forward(x, mask, self.W_q, self.W_out, self.b_out, self.scale)[0]

    @t.no_grad()
    def get_attention_weights(self, x, mask):
        return attention_probe_forward(x, mask, self.W_q, self.W_out, self.b_out, self.scale)[1]


def train_attention_probe(acts, masks, labels, n_epochs=200, lr=5e-3, weight_decay=1e-3, seed=0,
                          desc=None, chunk_size=256):
    """GPU-chunked full-batch training (previously CPU full-batch).

    """
    t.manual_seed(seed)
    probe = AttentionProbe(d_model=acts.shape[-1])          # CPU init: same numbers as the recorded runs
    device = "cuda" if t.cuda.is_available() else "cpu"
    probe = probe.to(device)
    opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    crit = t.nn.BCEWithLogitsLoss(reduction="sum")
    n = acts.shape[0]
    labels_f = labels.float().to(device)
    probe.train()
    loop = tqdm(range(n_epochs), desc=desc, leave=False) if desc else range(n_epochs)
    for _ in loop:
        opt.zero_grad()
        epoch_loss = 0.0
        for i in range(0, n, chunk_size):
            xc, mc = acts[i:i + chunk_size].to(device), masks[i:i + chunk_size].to(device)
            loss_c = crit(probe(xc, mc), labels_f[i:i + chunk_size]) / n
            loss_c.backward()
            epoch_loss += loss_c.item()
        opt.step()
        if desc:
            loop.set_postfix(loss=f"{epoch_loss:.4f}")
    return probe.cpu().eval()
