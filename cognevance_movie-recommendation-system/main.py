"""Run the full Movie Recommendation pipeline.

    python main.py                 # real MovieLens data in data/raw/
    python main.py --sample        # synthetic demo data (offline smoke test)
    python main.py --users 1 5 42  # choose which users get recommendations
"""
import argparse
import json

import pandas as pd

from src import config as cfg
from src.eda import run_eda
from src.evaluate import ranking_metrics, rating_metrics, train_test_split_ratings
from src.preprocess import clean, load_raw, save_processed
from src.recommender import ItemSimilarity, UserBasedCF


def taste_profile(user_id, ratings, movies, n_movies=3, n_genres=3):
    """Short human-readable summary of what a user likes."""
    mine = ratings[ratings["userId"] == user_id].merge(
        movies[["movieId", "clean_title", "genres"]], on="movieId")
    favourites = mine.sort_values(["rating", "timestamp"], ascending=False)["clean_title"].head(n_movies)
    liked = mine[mine["rating"] >= 4].assign(genre=lambda d: d["genres"].str.split("|")).explode("genre")
    genres = liked["genre"].value_counts().head(n_genres).index.tolist()
    return {"ratings_given": len(mine), "avg_rating": round(mine["rating"].mean(), 2),
            "favourite_genres": genres, "top_rated": favourites.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="use synthetic demo data")
    p.add_argument("--users", type=int, nargs="*", help="user IDs to recommend for")
    p.add_argument("--top-n", type=int, default=cfg.TOP_N)
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    data_dir = cfg.SAMPLE_DIR if args.sample else cfg.RAW_DIR
    if args.sample and not (data_dir / "ratings.csv").exists():
        from src.generate_sample_data import generate
        generate()

    # 1-2. Load + clean --------------------------------------------------
    print("[1/6] Loading and cleaning data ...")
    movies, ratings = load_raw(data_dir)
    movies, ratings, log = clean(movies, ratings)
    save_processed(movies, ratings)
    print(f"      {log['raw_ratings']:,} -> {log['clean_ratings']:,} ratings | "
          f"{log['clean_users']:,} users | {log['clean_movies']:,} movies")

    # 3, 6. EDA + charts ---------------------------------------------------
    print("[2/6] Analysing ratings and creating charts ...")
    summary = run_eda(ratings, movies)
    print(f"      sparsity {summary['matrix_sparsity_pct']}% | mean rating {summary['mean_rating']}")

    # 4. Train -------------------------------------------------------------
    print("[3/6] Training collaborative-filtering model ...")
    model = UserBasedCF(k=cfg.K_NEIGHBORS, shrinkage=cfg.SHRINKAGE).fit(ratings)
    item_model = ItemSimilarity().fit(ratings)

    # 5. Recommend ---------------------------------------------------------
    print("[4/6] Generating recommendations ...")
    users = args.users or [int(u) for u in
                           pd.Series(model.user_ids).sample(5, random_state=cfg.RANDOM_STATE)]
    all_recs, md = [], ["# Sample Recommendations\n"]
    for uid in users:
        recs = model.recommend(uid, movies, args.top_n)
        prof = taste_profile(uid, ratings, movies)
        all_recs.append(recs.assign(userId=uid))
        print(f"\n  User {uid}: likes {', '.join(prof['favourite_genres'])} | "
              f"avg rating {prof['avg_rating']} ({prof['ratings_given']} ratings)")
        print(recs[["rank", "title", "genres", "predicted_rating"]].to_string(index=False))
        md += [f"## User {uid}",
               f"- Ratings given: {prof['ratings_given']}, average rating: {prof['avg_rating']}",
               f"- Favourite genres: {', '.join(prof['favourite_genres'])}",
               f"- Highest-rated movies: {'; '.join(prof['top_rated'])}\n",
               "```", recs[["rank", "title", "genres", "predicted_rating"]].to_string(index=False), "```",
               ""]
    pd.concat(all_recs)[["userId", "rank", "movieId", "title", "genres", "predicted_rating"]] \
        .to_csv(cfg.OUTPUT_DIR / "recommendations.csv", index=False)

    # "Because you watched ..." ------------------------------------------------
    popular = ratings.groupby("movieId").size().nlargest(3).index
    md.append("# Similar Movies (item-based)\n")
    print("\n  Similar-movie lookups:")
    for mid in popular:
        title = movies.loc[movies["movieId"] == mid, "clean_title"].iloc[0]
        sim = item_model.similar_movies(int(mid), movies, 5)
        print(f"\n  Because you liked '{title}':")
        print(sim[["title", "genres", "similarity"]].to_string(index=False))
        md += [f"## Because you liked *{title}*",
               "```", sim[["title", "genres", "similarity"]].to_string(index=False), "```", ""]
    (cfg.OUTPUT_DIR / "recommendations.md").write_text("\n".join(md), encoding="utf-8")

    # 7. Evaluate ----------------------------------------------------------
    if not args.skip_eval:
        print("\n[5/6] Evaluating on a held-out 20% test split ...")
        train, test = train_test_split_ratings(ratings, cfg.TEST_SIZE, cfg.RANDOM_STATE)
        eval_model = UserBasedCF(k=cfg.K_NEIGHBORS, shrinkage=cfg.SHRINKAGE).fit(train)
        metrics = {"rating_prediction": rating_metrics(eval_model, test),
                   "top_n_ranking": ranking_metrics(eval_model, test, args.top_n,
                                                    cfg.RELEVANT_THRESHOLD)}
        (cfg.OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        for name, v in metrics["rating_prediction"].items():
            if isinstance(v, dict):
                print(f"      {name:22s} RMSE {v['RMSE']:.3f}  MAE {v['MAE']:.3f}")
        for name, v in metrics["top_n_ranking"].items():
            if isinstance(v, dict):
                print(f"      {name:22s} " + "  ".join(f"{a} {b:.3f}" for a, b in v.items()))

    print(f"\n[6/6] Done. See {cfg.OUTPUT_DIR}/ for charts, CSV, metrics and recommendations.")


if __name__ == "__main__":
    main()
