"""Offline evaluation: rating-prediction error and Top-N ranking quality."""
import numpy as np
import pandas as pd

from src.recommender import UserBasedCF


def train_test_split_ratings(ratings: pd.DataFrame, test_size=0.2, seed=42):
    test = ratings.sample(frac=test_size, random_state=seed)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def rating_metrics(model: UserBasedCF, test: pd.DataFrame) -> dict:
    t = test[test["userId"].isin(model.user_index) & test["movieId"].isin(model.movie_index)]
    ui = t["userId"].map(model.user_index).to_numpy()
    mi = t["movieId"].map(model.movie_index).to_numpy()
    actual = t["rating"].to_numpy()

    def err(pred):
        return {"RMSE": float(np.sqrt(np.mean((pred - actual) ** 2))),
                "MAE": float(np.mean(np.abs(pred - actual)))}

    return {
        "test_ratings_evaluated": int(len(t)),
        "Global-mean baseline": err(np.full_like(actual, model.global_mean)),
        "User-mean baseline": err(model.user_mean[ui]),
        "User-based CF": err(model.pred[ui, mi]),
    }


def ranking_metrics(model: UserBasedCF, test: pd.DataFrame, k=10, threshold=4.0) -> dict:
    """Precision/Recall/Hit-rate @K. Relevant = held-out rating >= threshold."""
    popularity = model.rated.sum(0).astype(np.float32)
    liked = test[(test["rating"] >= threshold)
                 & test["userId"].isin(model.user_index)
                 & test["movieId"].isin(model.movie_index)]
    relevant = liked.groupby("userId")["movieId"].apply(set)

    results = {"User-based CF": [], "Popularity baseline": []}
    for user_id, rel in relevant.items():
        u = model.user_index[user_id]
        for name, scores in (("User-based CF", model.pred[u].copy()),
                             ("Popularity baseline", popularity.copy())):
            scores[model.rated[u]] = -np.inf
            top = np.argpartition(-scores, k)[:k]
            hits = len(rel & set(model.movie_ids[top]))
            results[name].append((hits / k, hits / len(rel), float(hits > 0)))

    out = {"users_evaluated": int(len(relevant)), "k": k}
    for name, vals in results.items():
        arr = np.array(vals)
        out[name] = {f"Precision@{k}": float(arr[:, 0].mean()),
                     f"Recall@{k}": float(arr[:, 1].mean()),
                     f"HitRate@{k}": float(arr[:, 2].mean())}
    return out
