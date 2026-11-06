"""用 NumPy 实现的 K-means 聚类。"""

import numpy as np


def squared_distances(features, centers):
    return np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)


class KMeans:
    def __init__(self, k, seed=42, n_init=10, max_iter=300):
        self.k = k
        self.seed = seed
        self.n_init = n_init
        self.max_iter = max_iter

    def fit(self, features):
        features = np.asarray(features, dtype=float)
        if not 1 <= self.k <= len(np.unique(features, axis=0)):
            raise ValueError("k must be between 1 and the number of distinct samples")

        rng = np.random.default_rng(self.seed)
        self.inertia_ = np.inf
        for _ in range(self.n_init):
            centers = features[rng.choice(len(features), self.k, replace=False)]
            for _ in range(self.max_iter):
                distances = squared_distances(features, centers)
                labels = distances.argmin(axis=1)
                updated = np.empty_like(centers)
                # 空簇用离当前最近中心最远的样本重新播种，各空簇取不同样本。
                farthest = iter(np.argsort(-distances.min(axis=1), kind="stable"))
                for cluster in range(self.k):
                    members = features[labels == cluster]
                    updated[cluster] = members.mean(axis=0) if len(members) else features[next(farthest)]
                converged = np.allclose(updated, centers, atol=1e-6, rtol=0)
                centers = updated
                if converged:
                    break

            # 使用最终中心重新分簇，再比较各次初始化的簇内平方和。
            distances = squared_distances(features, centers)
            inertia = distances.min(axis=1).sum()
            if len(np.unique(distances.argmin(axis=1))) == self.k and inertia < self.inertia_:
                self.centers_ = centers
                self.inertia_ = inertia

        if not np.isfinite(self.inertia_):
            raise RuntimeError("No initialization produced k nonempty clusters")
        return self

    def predict(self, features):
        return squared_distances(np.asarray(features, dtype=float), self.centers_).argmin(axis=1)
