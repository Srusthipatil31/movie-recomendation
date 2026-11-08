# Data

**Source:** [MovieLens "latest-small"](https://grouplens.org/datasets/movielens/latest/) (GroupLens Research, University of Minnesota) — ~100,000 ratings from ~600 users on ~9,700 movies.

Get it with one command from the project root:

```bash
python -m src.download_data
```

This places `movies.csv`, `ratings.csv` (plus `tags.csv`, `links.csv`) in `data/raw/`.
Alternatively download the zip manually from the link above (or from Kaggle) and copy the CSVs into `data/raw/`.

| File | Columns |
|------|---------|
| `movies.csv` | `movieId, title, genres` (genres are `\|`-separated) |
| `ratings.csv` | `userId, movieId, rating (0.5–5.0), timestamp` |

`data/processed/` holds the cleaned files written by the pipeline.

> Dataset licence: for education/research use, see the MovieLens README. Cite:
> F. Maxwell Harper and Joseph A. Konstan. *The MovieLens Datasets: History and Context.* ACM TiiS 5(4), 2015.
