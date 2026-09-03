# Social Media Engagement Predictor

Predicts how well a social media post will perform (engagement rate + a
Low/Medium/High/Viral tier) **before you post it**, based on caption,
hashtags, timing, content type, and account size — then logs every
prediction to a local database.

## Why this project

Most fresher ML projects are "iris classifier" or "spam detector" — this
one uses your listed skills (ML, Frontend, "AI Integration and Database
management") on a domain you actually have on your resume already: social
media management. It's also genuinely explainable in an interview because
every design choice (why RandomForest, why these features, why a tier
*and* a raw number) has a clear reason.

## What's inside

```
social-media-engagement-predictor/
├── data/
│   └── generate_synthetic_data.py   # builds a realistic 6,000-post dataset
├── src/
│   ├── feature_engineering.py       # shared feature pipeline (train + app use the same code)
│   ├── train_model.py               # trains regressor + tier classifier, saves to models/
│   └── database.py                  # SQLite logging of predictions
├── models/                          # trained models + feature importance chart (generated)
├── app.py                           # Streamlit demo UI
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the training dataset
python data/generate_synthetic_data.py

# 2. Train the models (prints R^2, accuracy, saves models/*.joblib)
python src/train_model.py

# 3. Launch the demo app
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Current model performance (on synthetic data)

- Engagement rate regressor: **R² ≈ 0.63**, MAE ≈ 0.7 percentage points
- Engagement tier classifier: **~55% accuracy** across 4 tiers (vs. 25% random baseline)

These numbers are honest — the data has intentional noise, same as real
engagement data would. Don't inflate them in your resume; "R² of 0.63" is
a perfectly respectable, defensible number for a fresher project.

## Upgrading to a real dataset

The synthetic generator exists so the project runs immediately with no
external downloads or API keys. When you're ready, swap in real data —
these have the same kind of caption/hashtag/engagement fields:

- Aviral Jain — *Social Media Engagement Dataset* (comprehensive: likes,
  comments, shares, views, saves, follower tier, sentiment, hashtags):
  https://www.kaggle.com/datasets/aviral342/social-media-engagement-dataset
- Eshum Malik — *Insta Trends: Turning Data Into Virality* (119 real
  Instagram posts with captions + hashtags + full metrics):
  https://www.kaggle.com/datasets/eshummalik/insta-trends-turning-data-into-virality

To use one: download the CSV, rename/remap its columns to match what
`src/feature_engineering.py` expects (`content_type`, `day_of_week`,
`posting_hour`, `hashtag_count`, `mention_count`, `caption_length`,
`sentiment_score`, `follower_count`, `is_verified`, `engagement_rate`),
save it as `data/posts.csv`, and re-run `python src/train_model.py`.
Real Twitter/X and Instagram APIs are heavily paywalled/restricted for
individual developers now, which is exactly why a Kaggle dataset is the
practical route for a student project.

## Ideas to extend (good "future work" talking points in interviews)

- Replace the lexicon-based sentiment with a real NLP model (e.g. a small
  HuggingFace sentiment pipeline) for richer caption understanding.
- Add a caption embedding (TF-IDF or sentence-transformers) as a feature
  instead of just length/sentiment.
- Deploy the Streamlit app publicly (Streamlit Community Cloud is free)
  and put the live link on your resume instead of just a GitHub repo.
- Add a "what would improve this post?" feature: try small perturbations
  (different hour, different hashtag count) and show which change helps most.

## Suggested resume bullet points

Pick whichever is most accurate once you've run it yourself:

- *"Built and deployed a social media engagement predictor using
  RandomForest regression/classification (R² 0.63) with a Streamlit
  front-end and SQLite-backed prediction history."*
- *"Designed an end-to-end ML pipeline — synthetic data generation,
  feature engineering, model training, and a live prediction interface —
  to estimate post engagement from caption, timing, and account features."*

## Notes

- A pre-generated `data/posts.csv` and pre-trained `models/*.joblib` are
  included, so `streamlit run app.py` works immediately after
  `pip install -r requirements.txt` — you don't have to run steps 1–2
  first. Re-run them any time you change the generator or swap in real
  data.
- `predictions.db` (SQLite) is created automatically the first time you
  run the app.
