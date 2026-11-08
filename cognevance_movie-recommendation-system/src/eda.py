"""Exploratory analysis and visualisations (saved as PNGs in outputs/figures)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIG_DIR, OUTPUT_DIR

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25})
BLUE, ORANGE = "#2f6db5", "#e07b39"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def _short(title, n=38):
    return title if len(title) <= n else title[: n - 1] + "…"


def plot_rating_distribution(ratings):
    counts = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color=BLUE)
    ax.set(title="Distribution of Ratings", xlabel="Rating", ylabel="Number of ratings")
    _save(fig, "01_rating_distribution.png")


def plot_most_rated(ratings, movies, n=10):
    top = (ratings.groupby("movieId").size().nlargest(n).rename("n").reset_index()
           .merge(movies[["movieId", "clean_title"]], on="movieId"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([_short(t) for t in top["clean_title"]][::-1], top["n"][::-1], color=BLUE)
    ax.set(title=f"Top {n} Most Popular Movies (by number of ratings)",
           xlabel="Number of ratings")
    _save(fig, "02_most_rated_movies.png")


def plot_best_rated(ratings, movies, n=10):
    """Top movies by IMDb-style weighted rating so 1-vote wonders don't win."""
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    c, m = ratings["rating"].mean(), stats["count"].quantile(0.80)
    stats = stats[stats["count"] >= m].copy()
    stats["score"] = (stats["count"] / (stats["count"] + m)) * stats["mean"] \
                     + (m / (stats["count"] + m)) * c
    top = (stats.nlargest(n, "score").reset_index()
           .merge(movies[["movieId", "clean_title"]], on="movieId"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([_short(t) for t in top["clean_title"]][::-1], top["score"][::-1], color=ORANGE)
    ax.set(title=f"Top {n} Best-Rated Movies (weighted rating)",
           xlabel="Weighted rating", xlim=(max(top["score"].min() - 0.3, 0), 5))
    _save(fig, "03_best_rated_movies.png")


def _exploded(ratings, movies):
    df = ratings.merge(movies[["movieId", "genres"]], on="movieId")
    df["genre"] = df["genres"].str.split("|")
    return df.explode("genre")


def plot_genres(ratings, movies):
    exp = _exploded(ratings, movies)
    movies_per_genre = movies["genres"].str.split("|").explode().value_counts()
    ratings_per_genre = exp["genre"].value_counts()
    avg_per_genre = exp.groupby("genre")["rating"].mean().sort_values()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].barh(movies_per_genre.index[::-1], movies_per_genre.values[::-1], color=BLUE)
    axes[0].set(title="Movies per Genre", xlabel="Movies")
    axes[1].barh(ratings_per_genre.index[::-1], ratings_per_genre.values[::-1], color=ORANGE)
    axes[1].set(title="Ratings per Genre", xlabel="Ratings")
    axes[2].barh(avg_per_genre.index, avg_per_genre.values, color="#4c9a6a")
    lo = max(avg_per_genre.min() - 0.3, 0)
    axes[2].set(title="Average Rating per Genre", xlabel="Mean rating", xlim=(lo, 5))
    _save(fig, "04_genre_analysis.png")


def plot_trends(ratings, movies):
    by_year = ratings.groupby(ratings["datetime"].dt.year)["rating"].agg(["count", "mean"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(by_year.index, by_year["count"], marker="o", color=BLUE)
    axes[0].set(title="Ratings Submitted per Year", xlabel="Year", ylabel="Ratings")
    axes[1].plot(by_year.index, by_year["mean"], marker="o", color=ORANGE)
    axes[1].set(title="Average Rating per Year", xlabel="Year", ylabel="Mean rating")
    _save(fig, "05_rating_trends_over_time.png")

    decade = (movies.dropna(subset=["year"])
              .assign(decade=lambda d: (d["year"] // 10 * 10).astype(int))
              .groupby("decade").size())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(decade.index.astype(str) + "s", decade.values, color=BLUE)
    ax.set(title="Movies in Catalogue by Release Decade", xlabel="Decade", ylabel="Movies")
    plt.setp(ax.get_xticklabels(), rotation=45)
    _save(fig, "06_movies_by_decade.png")


def plot_user_activity(ratings):
    per_user = ratings.groupby("userId").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_user.values, bins=40, color=BLUE)
    ax.set(title="Ratings per User", xlabel="Ratings given", ylabel="Users")
    ax.set_xscale("log")
    _save(fig, "07_user_activity.png")


def run_eda(ratings, movies):
    plot_rating_distribution(ratings)
    plot_most_rated(ratings, movies)
    plot_best_rated(ratings, movies)
    plot_genres(ratings, movies)
    plot_trends(ratings, movies)
    plot_user_activity(ratings)

    n_users, n_movies = ratings["userId"].nunique(), ratings["movieId"].nunique()
    summary = {
        "users": int(n_users), "movies": int(n_movies), "ratings": int(len(ratings)),
        "mean_rating": round(float(ratings["rating"].mean()), 3),
        "median_rating": float(ratings["rating"].median()),
        "matrix_sparsity_pct": round(100 * (1 - len(ratings) / (n_users * n_movies)), 2),
        "avg_ratings_per_user": round(len(ratings) / n_users, 1),
        "avg_ratings_per_movie": round(len(ratings) / n_movies, 1),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
