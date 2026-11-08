# Project Report — Movie Recommendation System

**Author:** <your name>  **Programme:** Cognevance Technologies – AI & ML (Level 1)  **Date:** <date>

## 1. Objective
Build an AI-based system that recommends movies to users from their past ratings using collaborative filtering.

## 2. Dataset
MovieLens *latest-small* (GroupLens): `ratings.csv` (userId, movieId, rating, timestamp) and `movies.csv` (movieId, title, genres).
After cleaning: **<N> users, <N> movies, <N> ratings**, matrix sparsity **<x>%** (see `outputs/dataset_summary.json`).

## 3. Workflow
1. **Data collection** – downloaded via `src/download_data.py`.
2. **Cleaning** – removed nulls/duplicates, kept latest rating per user–movie pair, extracted release year, standardised genres, dropped users/movies with < 5 ratings.
3. **Analysis** – rating distribution, popular and best-rated movies, genre and time trends.
4. **Modelling** – user-based collaborative filtering (mean-centred cosine similarity, K = 30 neighbours, shrinkage = 1) plus item-based similarity for "similar movies".
5. **Recommendation** – Top-10 unseen movies per user.
6. **Evaluation** – 80/20 hold-out split against simple baselines.

## 4. Key findings from the analysis
*(Fill in from your charts — a few examples of what to look for:)*
- Ratings skew positive: the most common rating is <x> and the mean is <x>. (`01_rating_distribution.png`)
- A small set of blockbusters receives most of the ratings (long-tail). (`02`, `07`)
- Most-rated genres: <…>; highest-rated genres: <…>. (`04`)
- Activity over time: <peak year…>. (`05`)

## 5. Recommendation output
Paste or screenshot 2–3 users from `outputs/recommendations.md`, noting how the recommended genres line up with each user's taste profile.

## 6. Evaluation results
Copy from `outputs/metrics.json`.

| Model | RMSE | MAE |
|-------|------|-----|
| Global-mean baseline | <x> | <x> |
| User-mean baseline | <x> | <x> |
| **User-based CF** | <x> | <x> |

| Model | Precision@10 | Recall@10 | Hit-rate@10 |
|-------|--------------|-----------|-------------|
| Popularity baseline | <x> | <x> | <x> |
| **User-based CF** | <x> | <x> | <x> |

**Interpretation:** <Does CF beat the baselines on RMSE/MAE? On ranking, how does it compare with plain popularity? Explain why — e.g. sparse data, popularity bias in MovieLens.>

## 7. Challenges and improvements
- Data sparsity and cold-start users/movies.
- Possible upgrades: matrix factorisation (SVD/ALS), hybrid with genre features, tuning K and shrinkage, a Streamlit front-end.

## 8. Conclusion
<2–3 sentences summarising what you built and learned.>
