"""运行 Wine 数据集的K-means 聚类实验。"""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from kmeans import KMeans

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SEED = 42
HERE = Path(__file__).resolve().parent


def scaled_split(train, validation):
    # 每折独立拟合标准化器，验证样本不参与计算均值和标准差。
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(validation)


def run_kmeans(features, labels):
    ks = np.arange(2, 32)
    full = StandardScaler().fit_transform(features)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = [scaled_split(features[i], features[j]) for i, j in folds.split(features)]
    sse, full_scores, cv_means, cv_stds, valid_counts = [], [], [], [], []

    print("\nK-means on Wine")
    print(" k       SSE   silhouette    CV mean     CV SD  valid folds")
    for k in ks:
        model = KMeans(k, seed=SEED).fit(full)
        sse.append(model.inertia_)
        full_scores.append(silhouette_score(full, model.predict(full)))
        fold_scores = []
        for train, validation in splits:
            predicted = KMeans(k, seed=SEED).fit(train).predict(validation)
            # 轮廓系数仅在有效簇数至少为 2 且少于样本数时有定义。
            if 2 <= len(np.unique(predicted)) < len(validation):
                fold_scores.append(silhouette_score(validation, predicted))
        valid_counts.append(len(fold_scores))
        cv_means.append(np.mean(fold_scores) if fold_scores else np.nan)
        cv_stds.append(np.std(fold_scores) if fold_scores else np.nan)
        print(f"{k:2d}  {sse[-1]:8.2f}   {full_scores[-1]:.6f}     "
              f"{cv_means[-1]:.6f}  {cv_stds[-1]:.6f}  {valid_counts[-1]}/5")

    # 归一化曲线后，找距两端连线向下偏离最大的位置作为肘点参考。
    normalized_k = (ks - ks[0]) / (ks[-1] - ks[0])
    normalized_sse = (np.asarray(sse) - sse[-1]) / (sse[0] - sse[-1])
    elbow = ks[np.argmax(1 - normalized_k - normalized_sse)]
    eligible = [i for i, count in enumerate(valid_counts) if count == len(splits)]
    best_idx = max(eligible, key=lambda i: (cv_means[i], -cv_stds[i], -ks[i]))
    best = ks[best_idx]
    print(f"Elbow candidate k={elbow}, full-data silhouette peak k={ks[np.argmax(full_scores)]}")
    print(f"K-means selected k={best} by mean validation silhouette")
    # 选定 k 后才使用真实标签做外部评估。
    predicted = KMeans(best, seed=SEED).fit(full).predict(full)
    print(f"External adjusted Rand index: {adjusted_rand_score(labels, predicted):.6f}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    left.plot(ks, sse, "o-", markersize=3)
    left.axvline(elbow, color="tab:orange", linestyle="--", label=f"Elbow k = {elbow}")
    left.set(xlabel="Number of clusters (k)", ylabel="Within-cluster SSE", title="Elbow method")
    right.plot(ks, full_scores, "o-", markersize=3, label="Full data")
    right.plot(ks, cv_means, "s-", markersize=3, label="Validation mean")
    right.axvline(best, color="tab:orange", linestyle="--", label=f"Selected k = {best}")
    right.set(xlabel="Number of clusters (k)", ylabel="Silhouette score", title="Cluster separation")
    for ax in (left, right):
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(HERE / "kmeans_evaluation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    wine = load_wine()
    run_kmeans(wine.data, wine.target)
