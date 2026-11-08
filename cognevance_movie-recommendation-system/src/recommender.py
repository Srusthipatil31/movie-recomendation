"""Collaborative-filtering recommenders.

UserBasedCF  - predicts a user's rating for every movie from the ratings of
               the K most similar users (cosine similarity on mean-centred
               ratings). Used to generate personalised Top-N lists.
ItemSimilarity - "people who liked X also liked ..." based on item-item
               cosine similarity.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    def __init__(self, k: int = 30, shrinkage: float = 1.0):
        self.k = k
        self.shrinkage = shrinkage

    # ------------------------------------------------------------------ fit
    def fit(self, ratings: pd.DataFrame) -> "UserBasedCF":
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(ratings["movieId"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}

        rows = ratings["userId"].map(self.user_index).to_numpy()
        cols = ratings["movieId"].map(self.movie_index).to_numpy()
        R = np.zeros((len(self.user_ids), len(self.movie_ids)), dtype=np.float32)
        R[rows, cols] = ratings["rating"].to_numpy(dtype=np.float32)

        self.rated = R > 0                                  # who rated what
        mask = self.rated.astype(np.float32)
        self.user_mean = R.sum(1) / np.maximum(mask.sum(1), 1)
        self.global_mean = float(ratings["rating"].mean())

        # Mean-centre each user's ratings (removes "harsh vs generous rater" bias)
        Rc = (R - self.user_mean[:, None]) * mask

        # User-user cosine similarity; keep only the top-K positive neighbours
        S = cosine_similarity(Rc).astype(np.float32)
        np.fill_diagonal(S, 0)
        S[S < 0] = 0
        k = min(self.k, S.shape[0] - 1)
        top_idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        Sk = np.zeros_like(S)
        r = np.arange(S.shape[0])[:, None]
        Sk[r, top_idx] = S[r, top_idx]
        self.sim = Sk

        # Predicted rating = user mean + weighted avg of neighbours' deviations
        numer = Sk @ Rc
        denom = Sk @ mask + self.shrinkage
        pred = self.user_mean[:, None] + numer / denom
        self.pred = np.clip(pred, 0.5, 5.0).astype(np.float32)
        return self

    # ------------------------------------------------------------ recommend
    def recommend(self, user_id: int, movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Top-N unseen movies for a user, with predicted ratings."""
        if user_id not in self.user_index:
            raise KeyError(f"Unknown user {user_id}")
        u = self.user_index[user_id]
        scores = self.pred[u].copy()
        scores[self.rated[u]] = -np.inf                     # never re-recommend
        top = np.argsort(-scores)[:n]
        out = pd.DataFrame({"movieId": self.movie_ids[top],
                            "predicted_rating": scores[top].round(2)})
        out = out.merge(movies[["movieId", "clean_title", "genres", "year"]], on="movieId")
        out.insert(0, "rank", range(1, len(out) + 1))
        return out.rename(columns={"clean_title": "title"})

    def similar_users(self, user_id: int, n: int = 5) -> list:
        u = self.user_index[user_id]
        idx = np.argsort(-self.sim[u])[:n]
        return [int(self.user_ids[i]) for i in idx if self.sim[u, i] > 0]


class ItemSimilarity:
    """Item-item cosine similarity, computed lazily (memory-friendly)."""

    def __init__(self, min_common: int = 20):
        self.min_common = min_common   # significance weighting threshold

    def fit(self, ratings: pd.DataFrame) -> "ItemSimilarity":
        self.movie_ids = np.sort(ratings["movieId"].unique())
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}
        uidx = {u: i for i, u in enumerate(self.user_ids)}

        R = np.zeros((len(self.user_ids), len(self.movie_ids)), dtype=np.float32)
        R[ratings["userId"].map(uidx).to_numpy(),
          ratings["movieId"].map(self.movie_index).to_numpy()] = \
            ratings["rating"].to_numpy(dtype=np.float32)
        self.mask = (R > 0).astype(np.float32)
        user_mean = R.sum(1) / np.maximum(self.mask.sum(1), 1)
        self.Rc = (R - user_mean[:, None]) * self.mask
        self.norms = np.linalg.norm(self.Rc, axis=0) + 1e-9
        return self

    def similar_movies(self, movie_id: int, movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        j = self.movie_index[movie_id]
        sims = (self.Rc.T @ self.Rc[:, j]) / (self.norms * self.norms[j])
        # Down-weight pairs that only a handful of users have both rated
        common = self.mask.T @ self.mask[:, j]
        sims = sims * np.minimum(common, self.min_common) / self.min_common
        sims[j] = -np.inf
        top = np.argsort(-sims)[:n]
        out = pd.DataFrame({"movieId": self.movie_ids[top],
                            "similarity": sims[top].round(3)})
        out = out.merge(movies[["movieId", "clean_title", "genres"]], on="movieId")
        return out.rename(columns={"clean_title": "title"})
