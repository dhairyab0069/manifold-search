"""Policies: gradient boosted trees (LightGBM) and a small MLP, both predicting the class "smallest sufficient
number of rings" (0..3). From the class probabilities a threshold tau turns the model into a policy: pick the
smallest r with P(label <= r) >= tau, so a larger tau is safer and costs more work."""
from __future__ import annotations

import numpy as np

N_CLASSES = 4


def policy_from_proba(proba: np.ndarray, tau: float) -> np.ndarray:
    cum = np.cumsum(proba, axis=1)
    ok = cum >= tau - 1e-12
    return np.where(ok.any(axis=1), ok.argmax(axis=1), proba.shape[1] - 1)


class GBTModel:
    name = "gbt"

    def __init__(self, seed: int = 0, n_estimators: int = 200, learning_rate: float = 0.05, num_leaves: int = 15):
        import lightgbm as lgb
        self.m = lgb.LGBMClassifier(objective="multiclass", n_estimators=n_estimators, learning_rate=learning_rate,
                                    num_leaves=num_leaves, min_child_samples=200, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=8)

    def fit(self, x, y):
        self.m.fit(x, y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, x):
        p = self.m.predict_proba(x)
        out = np.zeros((len(x), N_CLASSES))
        out[:, self.classes_] = p
        return out


class MLPModel:
    name = "mlp"

    def __init__(self, seed: int = 0, hidden: int = 64, epochs: int = 30, lr: float = 2e-3, device: str | None = None):
        import torch
        self.torch, self.seed, self.hidden, self.epochs, self.lr = torch, seed, hidden, epochs, lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x, y):
        torch = self.torch
        torch.manual_seed(self.seed)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-8
        xt = torch.as_tensor(np.clip((x - self.mu) / self.sd, -8, 8), dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        h = self.hidden
        self.net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(),
                                       torch.nn.Linear(h, N_CLASSES)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(xt), generator=g).to(self.device)
            for i in range(0, len(xt), 512):
                b = perm[i:i + 512]
                loss = torch.nn.functional.cross_entropy(self.net(xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_proba(self, x):
        torch = self.torch
        with torch.no_grad():
            xt = torch.as_tensor(np.clip((x - self.mu) / self.sd, -8, 8), dtype=torch.float32, device=self.device)
            return torch.softmax(self.net(xt), dim=1).cpu().numpy().astype(np.float64)
