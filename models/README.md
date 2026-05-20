# Models Folder Content

This folder contains trained model artifacts for the Netflix Recommendation Engine.

## Files

- `rating_prediction_model.pkl` - Random Forest model for rating prediction.
- `user_encoder.pkl` - LabelEncoder for CustomerID.
- `movie_encoder.pkl` - LabelEncoder for MovieID.
- `genre_encoder.pkl` - LabelEncoder for Genre.
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer for movie content features.
- `cosine_similarity.pkl` - Movie-to-movie similarity matrix.
- `movie_stats.pkl` - Movie statistics with average rating, rating count, and popularity score.
- `movies.pkl` - Unique movie data used by content-based recommendation.
- `model_metrics.pkl` - Saved RMSE and MAE values.

## Current metrics

- RMSE: 1.1498
- MAE: 0.9715
- Rows used: 5969
- Unique users: 100
- Unique movies: 500
- Unique genres: 10

## How to regenerate

Run this from the project root:

```bash
python src/train_models.py
```
