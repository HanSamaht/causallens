"""Demo models across domains.

1) finance_demo: a GradientBoosting classifier with a planted
   spurious feature month_index the auditor is expected to flag.
2) medicine_demo: a RandomForest diagnosis classifier.
3) torch_demo: an MLP with the finance feature set for activation
   patching and attribution work.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FINANCE_FEATURES = [
    "momentum", #
    "value", # value
    "volatility", #
    "liquidity", #
    "reversal", #
    "size", # value
    "earnings_yield",
    "month_index", # planted spurious feature for the demo
]

MEDICINE_CONCEPT_KEYS = None # fromfeature


def make_finance_data(n: int = 4000, seed: int = 7):
    """Make finance data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FINANCE_FEATURES)))
    X[:, 5] = -abs(X[:, 5]) # size ,morevalue
    X[:, 3] = 0.6 * X[:, 0] + 0.8 * rng.normal(size=n) # andcorrelation(correlation≠causal)
    X[:, 7] = rng.integers(0, 12, size=n) #

    logit = (
        1.6 * X[:, 0] # ->
        - 1.1 * X[:, 2] # ↘
        + 0.8 * X[:, 1] # value ->
        + 0.7 * X[:, 6] # ->
        - 0.6 * X[:, 4] # ↘
        + 0.3 * X[:, 3] # (partialand)
        + 0.2 * X[:, 5]
        + 0.9 * np.isin(X[:, 7], [3, 7]).astype(float) # trainingeffect()
        + 0.25 * rng.normal(size=n)
    )
    y = (logit > 0).astype(int)
    return X, y


def train_finance_demo():
    """Train finance demo."""
    X, y = make_finance_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0)
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return model, FINANCE_FEATURES, X_te, float(auc)


def train_medicine_demo():
    """Train medicine demo."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    names = [c.replace(" ", "_") for c in data.feature_names]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return model, names, X_te, float(auc)


# ---------------- (prediction) ----------------
def make_timeseries_data(n_series: int = 300, length: int = 220, seed: int = 11):
    """below:(+) ,value(-) ,(causal).

    y_t = 0.5*mean(mom[t-3:t+1]) - 0.7*mean(rev[t-1:t+1]) + eps
    """
    rng = np.random.default_rng(seed)
    series, labels = [], []
    for _ in range(n_series):
        mom = np.cumsum(rng.normal(0, 0.4, length)) # :
        rev = 0.6 * rng.normal(size=length) # valueattribution
        noise_f = rng.normal(0, 1.0, length) # (causal)
        y = np.array(
            [
                0.5 * mom[max(0, t - 3): t + 1].mean() - 0.7 * rev[max(0, t - 1): t + 1].mean()
                + 0.02 * noise_f[t]
                + rng.normal(0, 0.1)
                for t in range(length)
            ]
        )
        series.append(np.stack([mom, rev, noise_f], axis=1)) # (length, 3)
        labels.append(y)
    return np.array(series), np.array(labels)


TS_FEATURES = ["ts_momentum", "ts_reversion", "ts_noise"]


def _make_windows(series: np.ndarray, labels: np.ndarray, seq_len: int = 16):
    """Make windows."""
    X, Y = [], []
    for s, y in zip(series, labels):
        for t in range(seq_len, len(y)):
            X.append(s[t - seq_len: t]) # not t(leakage)
            Y.append(y[t])
    return np.array(X, dtype=float), np.array(Y, dtype=float)


def build_ts_model(n_features: int = 3, hidden: int = 16, seed: int = 0):
    """Build ts model."""
    import torch

    class GRUNet(torch.nn.Module):
        """Grunet."""
        def __init__(self):
            """Init."""
            super().__init__()
            self.gru = torch.nn.GRU(n_features, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden, 1)

        def forward(self, x): # (B, T, F)
            """Forward pass through GRU and head."""
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)

    torch.manual_seed(seed)
    return GRUNet()


def train_timeseries_demo(seq_len: int = 16, epochs: int = 60, seed: int = 11):
    """Forward."""
    import torch

    series, labels = make_timeseries_data(seed=seed)
    X, Y = _make_windows(series, labels, seq_len)
    split = int(len(X) * 0.8)
    Xtr = torch.tensor(X[:split], dtype=torch.float32)
    Ytr = torch.tensor(Y[:split], dtype=torch.float32)
    net = build_ts_model(seed=0)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    lossf = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xtr), Ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(X[split:], dtype=torch.float32)).numpy()
    ss_res = float(((Y[split:] - pred) ** 2).sum())
    ss_tot = float(((Y[split:] - Y[split:].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    background = X[split:][:300] # (n, T, F) background
    return net, TS_FEATURES, background, float(r2)


GEO_FEATURES = [
    "elevation", "slope", "ndvi", "lst", "precipitation", "pop_density", "dist_river",
    "region_id",  # planted spurious feature for the demo
]


def train_geo_demo():
    """model(RandomForest).region_id asfeature."""
    rng = np.random.default_rng(3)
    n = 3000
    X = rng.normal(size=(n, len(GEO_FEATURES)))
    X[:, 0] = X[:, 0] * 400 + 1200 # ()
    X[:, 4] = X[:, 4] * 300 + 1100 # ()
    X[:, 6] = abs(X[:, 6]) * 1500 # ()
    X[:, 7] = rng.integers(0, 8, size=n) #

    slope_z = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    elev_z = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    ndvi_z = X[:, 2]
    precip_z = (X[:, 4] - X[:, 4].mean()) / X[:, 4].std()
    dist_z = (X[:, 6] - X[:, 6].mean()) / X[:, 6].std()

    logit = (
        1.8 * slope_z + 0.9 * elev_z - 0.8 * ndvi_z + 0.6 * precip_z - 0.5 * dist_z
        + 2.2 * np.isin(X[:, 7], [1, 4, 6]).astype(float) # training(bias)
        + 0.3 * rng.normal(size=n)
    )
    y = (logit > 0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return model, GEO_FEATURES, X_te, float(auc)


# ---------------- :German Credit(feature + ) ----------------

def train_credit_demo(missing_rate: float = 0.08, seed: int = 0):
    """Credit demo with missing values to validate missing data handling.

    Features arrive with values missing completely at random at 8
    percent, encoded as was_missing_* indicator columns that enter the
    model input; the auditor should find the imputation markers in
    attribution.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder

    import lightgbm as lgb

    ds = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df, y = ds.data.copy(), (ds.target == "bad").astype(int).values

    cat_cols = list(df.select_dtypes(include=["category", "object"]).columns)
    if cat_cols:
        df[cat_cols] = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(df[cat_cols])

    rng = np.random.default_rng(seed)
    X = df.astype(float).values
    feat_names = list(df.columns)
    num_cols = [c for c in feat_names if c not in cat_cols]

    indicators = []
    for j, c in enumerate(feat_names):
        if c in num_cols and rng.random() < 0.8: # 80% value MCAR
            mask = rng.random(len(X)) < missing_rate
            if mask.sum() > 0:
                X[mask, j] = np.nan
                indicators.append((f"was_missing_{c}", mask.astype(float)))

    for j in range(X.shape[1]): # in
        col = X[:, j]
        col[np.isnan(col)] = np.nanmedian(col)

    for name, ind in indicators:
        X = np.column_stack([X, ind])
        feat_names.append(name)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1
    ).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return model, feat_names, X_te, float(auc)


IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def train_iris_demo():
    """Iris demo: the smallest end-to-end interrogation target."""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    data = load_iris()
    model = RandomForestClassifier(n_estimators=150, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25, random_state=0, stratify=data.target)
    model.fit(X_tr, y_tr)
    return model, IRIS_FEATURES, X_te, float(accuracy_score(y_te, model.predict(X_te)))


# ---------------- log: ----------------

AGENT_LOG_FEATURES = [
    "credit_score", "debt_ratio", "income_k", "loan_amount_k",
    "employment_years", "region_code",  # region_code is a planted bias column
]


def make_agent_logs(n: int = 4000, seed: int = 5):
    """Decision log demo: behavior records without a predict function.

    The generating rule discriminates: applicants from region_code 3
    or 7 are rejected at 1.2 times the base rate, and the log analysis
    is expected to surface it.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.uniform(300, 850, n), # credit_score
            rng.uniform(0, 1, n), # debt_ratio
            rng.uniform(2, 80, n), # income_k
            rng.uniform(1, 100, n), # loan_amount_k
            rng.uniform(0, 30, n), # employment_years
            rng.integers(0, 10, n).astype(float), # region_code
        ]
    )
    logit = (
        0.010 * (X[:, 0] - 600) #
        - 2.0 * X[:, 1] #
        + 0.03 * X[:, 2]
        - 0.008 * X[:, 3]
        + 0.05 * X[:, 4]
        - 1.2 * np.isin(X[:, 5], [3, 7]).astype(float) #
        + 0.3 * rng.normal(size=n)
    )
    y = (logit > 0).astype(int) # 1=
    return X, y


# ---------------- torch ----------------
class FinanceMLP:
    """ torch MLP,and demo ."""

    @staticmethod
    def build(n_features: int = 8, seed: int = 0):
        """Build."""
        import torch

        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        return net

    @staticmethod
    def train(epochs: int = 200, seed: int = 7):
        """Train."""
        import torch

        X, y = make_finance_data(seed=seed)
        X_tr = torch.tensor(X[:3000], dtype=torch.float32)
        y_tr = torch.tensor(y[:3000], dtype=torch.float32)
        X_te, y_te = X[3000:], y[3000:]

        net = FinanceMLP.build(len(FINANCE_FEATURES), seed=0)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(net(X_tr).squeeze(-1), y_tr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = torch.sigmoid(net(torch.tensor(X_te, dtype=torch.float32)).squeeze(-1)).numpy()
        return net, FINANCE_FEATURES, X_te, float(roc_auc_score(y_te, p))
