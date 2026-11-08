import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


class MLPNet:
    """Small torch MLP. embed() exposes the hidden layer for the unknown-attack detector later."""

    def __init__(self, n_in, n_out, hidden=(128, 64), lr=1e-3, epochs=20,
                 batch_size=512, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.RandomState(seed)
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_out)
        self.lr, self.epochs, self.bs = lr, epochs, batch_size

    def fit(self, X, y):
        params = list(self.body.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        for _ in range(self.epochs):
            perm = torch.tensor(self.rng.permutation(len(X)))
            for i in range(0, len(X), self.bs):
                idx = perm[i:i + self.bs]
                opt.zero_grad()
                loss = loss_fn(self.head(self.body(X[idx])), y[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def embed(self, X):
        return self.body(torch.tensor(X, dtype=torch.float32)).numpy()

    @torch.no_grad()
    def predict_proba(self, X):
        logits = self.head(self.body(torch.tensor(X, dtype=torch.float32)))
        return torch.softmax(logits, dim=1).numpy()


class StaticModel:
    """Wraps a model so it is fitted and queried with string family labels."""

    def __init__(self, name, cfg, seed=0):
        self.name, self.cfg, self.seed = name, cfg, seed

    def fit(self, X, y_str):
        self.classes_ = np.unique(y_str)
        y = np.searchsorted(self.classes_, y_str)
        if self.name == "mlp":
            m = self.cfg["mlp"]
            self.model = MLPNet(X.shape[1], len(self.classes_), tuple(m["hidden"]),
                                m["lr"], m["epochs"], m["batch_size"], self.seed)
        elif self.name == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8,
                random_state=self.seed, verbose=-1)
        elif self.name == "logreg":
            self.model = LogisticRegression(max_iter=300, random_state=self.seed)
        else:
            raise ValueError(self.name)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]