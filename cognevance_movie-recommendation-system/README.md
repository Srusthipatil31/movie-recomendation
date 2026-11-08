# 🎬 Movie Recommendation System

A collaborative-filtering recommender built on the MovieLens dataset.
It cleans the data, explores rating patterns, trains a **user-based collaborative filtering** model, produces personalised Top-N movie lists for different users, finds **similar movies** with item-item similarity, and evaluates everything against baselines.

> Cognevance Technologies · AI & ML · Level 1 (Easy)

## Features

| Task | Where |
|------|-------|
| Collect dataset (MovieLens) | `src/download_data.py` |
| Clean & preprocess | `src/preprocess.py` |
| Analyse ratings and genres, build charts | `src/eda.py` → `outputs/figures/` |
| Collaborative filtering | `src/recommender.py` |
| Recommendations for different users | `outputs/recommendations.csv`, `outputs/recommendations.md` |
| Evaluation (RMSE, MAE, Precision/Recall@K) | `src/evaluate.py` → `outputs/metrics.json` |
| Report | `REPORT.md` |

## Project structure

```
cognevance_movie-recommendation-system/
├── main.py                  # runs the whole pipeline
├── requirements.txt
├── REPORT.md                # project report
├── data/
│   ├── raw/                 # MovieLens CSVs (movies.csv, ratings.csv)
│   └── processed/           # cleaned CSVs written by the pipeline
├── src/
│   ├── config.py            # paths + hyper-parameters
│   ├── download_data.py     # fetch MovieLens
│   ├── generate_sample_data.py  # tiny synthetic data for offline smoke tests
│   ├── preprocess.py        # cleaning, year extraction, sparse filtering
│   ├── eda.py               # analysis + charts
│   ├── recommender.py       # UserBasedCF, ItemSimilarity
│   └── evaluate.py          # train/test split + metrics
└── outputs/
    ├── figures/             # PNG charts
    ├── recommendations.csv
    ├── recommendations.md
    ├── dataset_summary.json
    └── metrics.json
```

## Quick start

```bash
# 1. clone and enter the repo
git clone https://github.com/<your-username>/cognevance_movie-recommendation-system.git
cd cognevance_movie-recommendation-system

# 2. (optional) virtual environment
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. download the dataset
python -m src.download_data

# 5. run everything
python main.py
```

Useful options:

```bash
python main.py --users 1 42 99     # recommendations for specific users
python main.py --top-n 5           # shorter lists
python main.py --skip-eval         # skip the evaluation step
python main.py --sample            # offline demo with synthetic data (not real results)
```

## How it works

1. **Preprocessing** – drop nulls/duplicates, keep each user's latest rating per movie, extract release year from the title, replace `(no genres listed)`, and remove movies/users with fewer than 5 ratings.
2. **Analysis** – rating distribution, most-rated and best-rated movies (IMDb-style weighted rating so a movie with one 5★ vote can't top the list), genre popularity, rating trends over time, user activity.
3. **User-based collaborative filtering**
   - Build a user × movie matrix and **mean-centre** each user's ratings (removes harsh-vs-generous rater bias).
   - Compute **cosine similarity** between users; keep the top *K* (=30) positive neighbours.
   - Predicted rating = user's mean + similarity-weighted average of neighbours' deviations, with a small **shrinkage** term so a single neighbour can't dominate.
   - Recommend the highest-predicted movies the user hasn't rated yet.
4. **Item similarity** – "because you liked *X*": item-item cosine similarity with significance weighting (pairs co-rated by few users are down-weighted).
5. **Evaluation** on a random 80/20 split:
   - *Rating prediction*: RMSE / MAE vs global-mean and user-mean baselines.
   - *Top-N ranking*: Precision@10, Recall@10, Hit-rate@10 (a held-out rating ≥ 4 counts as "liked") vs a popularity baseline.

## Outputs

After `python main.py` you'll find:

- `outputs/figures/01…07_*.png` – charts
- `outputs/recommendations.csv` / `.md` – Top-10 lists for several users, with each user's taste profile
- `outputs/metrics.json` – evaluation results
- `outputs/dataset_summary.json` – users, movies, ratings, sparsity

*(Add 2–3 chart screenshots and a recommendation sample here after your run, e.g. `![Genres](outputs/figures/04_genre_analysis.png)`.)*

## Technologies

Python 3.9+, pandas, NumPy, scikit-learn, Matplotlib.

## Limitations & future work

- The full similarity/prediction matrices are held in memory — fine for MovieLens-small, but larger datasets need sparse or approximate methods (e.g. ALS / matrix factorisation, FAISS).
- Pure collaborative filtering can't recommend to brand-new users or movies (cold start); a content-based or hybrid model using genres would help.
- Explicit-rating metrics are offline proxies — real systems validate with A/B tests.

## Acknowledgements

Dataset: GroupLens Research, *MovieLens*. Built for the Cognevance Technologies AI/ML internship tasks.
