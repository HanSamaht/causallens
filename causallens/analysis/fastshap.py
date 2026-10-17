"""FastSHAP : SHAP .

:FastSHAP (Jethani et al., ICLR 2022, 337 )——training
prediction SHAP value,result,count.

 MIC model:`full` (T×F feature)from.
"""
from __future__ import annotations

import numpy as np


class AmortizedExplainer:
    """: MLP (input->SHAPvalue) .

    training,sample SHAP attributiononly.
    model(parameter),CPU training.
    """

    def __init__(self, n_features: int, hidden: int = 128, seed: int = 0):
        import torch

        self.torch = torch
        self.torch.manual_seed(seed)
        self.n_features = n_features
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_features), # output:everyfeature SHAP value
        )
        self.fitted = False

    def fit(self, get_shap_fn, background: np.ndarray, n_samples: int = 200,
            epochs: int = 200, lr: float = 1e-3, verbose: bool = False) -> dict:
        """training.

        get_shap_fn: ,input (n, F) ,return (n, F) SHAP
        background: background (n, F)
        """
        t = self.torch
        rng = np.random.default_rng(0)
        idx = rng.choice(len(background), size=min(n_samples, len(background)), replace=False)
        X_sub = background[idx]

        # SHAP training
        shap_targets = np.array([get_shap_fn(X_sub[i:i+1])[0] for i in range(len(X_sub))])

        X_t = t.tensor(X_sub, dtype=t.float32)
        Y_t = t.tensor(shap_targets, dtype=t.float32)

        opt = t.optim.Adam(self.net.parameters(), lr=lr)
        lossf = t.nn.MSELoss()
        for ep in range(epochs):
            opt.zero_grad()
            pred = self.net(X_t)
            loss = lossf(pred, Y_t)
            loss.backward()
            opt.step()
            if verbose and (ep + 1) % 50 == 0:
                print(f" epoch {ep+1}: loss={loss.item():.6f}")

        with t.no_grad():
            final_loss = float(lossf(self.net(X_t), Y_t))
        self.fitted = True
        return {"n_training_samples": len(X_sub), "final_mse": round(final_loss, 6), "epochs": epochs}

    def attribute(self, x: np.ndarray) -> np.ndarray:
        """,return SHAP value."""
        assert self.fitted, " fit()"
        t = self.torch
        with t.no_grad():
            x_t = t.tensor(np.asarray(x, dtype=np.float32).reshape(1, -1), dtype=t.float32)
            return self.net(x_t).cpu().numpy()[0]
