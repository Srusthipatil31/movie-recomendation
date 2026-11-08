"""Load, clean and preprocess the MovieLens data."""
from pathlib import Path

import pandas as pd

from src.config import (MIN_RATINGS_PER_MOVIE, MIN_RATINGS_PER_USER,
                        PROCESSED_DIR)


def load_raw(data_dir: Path):
    movies_path, ratings_path = data_dir / "movies.csv", data_dir / "ratings.csv"
    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(
            f"movies.csv / ratings.csv not found in {data_dir}.\n"
            "Run `python -m src.download_data` (MovieLens) or use `--sample` for demo data."
        )
    return pd.read_csv(movies_path), pd.read_csv(ratings_path)


def clean(movies: pd.DataFrame, ratings: pd.DataFrame):
    """Return cleaned (movies, ratings) plus a dict describing what was removed."""
    log = {"raw_movies": len(movies), "raw_ratings": len(ratings),
           "raw_users": ratings["userId"].nunique()}

    # --- movies -----------------------------------------------------------
    movies = movies.dropna(subset=["movieId", "title"]).drop_duplicates("movieId").copy()
    movies["genres"] = movies["genres"].fillna("Unknown").replace(
        "(no genres listed)", "Unknown")
    # Pull the release year out of "Toy Story (1995)"
    movies["year"] = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True)

    # --- ratings ----------------------------------------------------------
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    ratings = ratings[ratings["rating"].between(0.5, 5.0)]
    # If a user rated the same movie twice, keep the most recent rating
    ratings = (ratings.sort_values("timestamp")
                      .drop_duplicates(["userId", "movieId"], keep="last"))
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    # --- drop sparse movies / users (iterate until stable) ------------------
    while True:
        before = len(ratings)
        movie_counts = ratings["movieId"].value_counts()
        ratings = ratings[ratings["movieId"].isin(
            movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index)]
        user_counts = ratings["userId"].value_counts()
        ratings = ratings[ratings["userId"].isin(
            user_counts[user_counts >= MIN_RATINGS_PER_USER].index)]
        if len(ratings) == before:
            break

    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]
    ratings = ratings.reset_index(drop=True)
    movies = movies.reset_index(drop=True)

    log.update({"clean_movies": len(movies), "clean_ratings": len(ratings),
                "clean_users": ratings["userId"].nunique()})
    return movies, ratings, log


def save_processed(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_csv(PROCESSED_DIR / "movies_clean.csv", index=False)
    ratings.to_csv(PROCESSED_DIR / "ratings_clean.csv", index=False)
