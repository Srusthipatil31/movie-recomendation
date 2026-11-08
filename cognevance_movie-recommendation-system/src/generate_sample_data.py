"""Generate a small synthetic dataset in MovieLens format.

Only for smoke-testing the pipeline offline. Use the real MovieLens data
(``python -m src.download_data``) for your actual project results.

Usage:
    python -m src.generate_sample_data
"""
import numpy as np
import pandas as pd

from src.config import SAMPLE_DIR

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime",
          "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]


def generate(n_users=300, n_movies=600, seed=42) -> None:
    rng = np.random.default_rng(seed)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # Movies
    years = rng.integers(1980, 2019, n_movies)
    genre_matrix = np.zeros((n_movies, len(GENRES)))
    genre_strings = []
    for i in range(n_movies):
        chosen = rng.choice(len(GENRES), size=rng.integers(1, 4), replace=False)
        genre_matrix[i, chosen] = 1
        genre_strings.append("|".join(GENRES[j] for j in chosen))
    movies = pd.DataFrame({
        "movieId": np.arange(1, n_movies + 1),
        "title": [f"Sample Movie {i} ({y})" for i, y in zip(range(1, n_movies + 1), years)],
        "genres": genre_strings,
    })

    # Ratings: each user has genre tastes; popular movies are rated more often
    popularity = rng.lognormal(0, 1, n_movies)
    popularity /= popularity.sum()
    quality = rng.normal(0, 0.5, n_movies)
    rows = []
    for u in range(1, n_users + 1):
        taste = rng.normal(0, 1, len(GENRES))
        n_rated = rng.integers(20, 120)
        picked = rng.choice(n_movies, size=n_rated, replace=False, p=popularity)
        for m in picked:
            score = 3.3 + 0.6 * genre_matrix[m].dot(taste) / max(genre_matrix[m].sum(), 1) \
                    + quality[m] + rng.normal(0, 0.5)
            rating = float(np.clip(np.round(score * 2) / 2, 0.5, 5.0))
            ts = int(rng.integers(1_100_000_000, 1_540_000_000))
            rows.append((u, m + 1, rating, ts))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])

    movies.to_csv(SAMPLE_DIR / "movies.csv", index=False)
    ratings.to_csv(SAMPLE_DIR / "ratings.csv", index=False)
    print(f"Sample data written to {SAMPLE_DIR} "
          f"({len(movies)} movies, {len(ratings)} ratings)")


if __name__ == "__main__":
    generate()
