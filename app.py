"""
app.py

Streamlit demo: type in a hypothetical post's details, get a predicted
engagement rate + tier, see which factors drove the prediction, and browse
a history of past predictions (pulled from SQLite).

Run with: streamlit run app.py
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_engineering import build_features_single, simple_sentiment  # noqa: E402
from database import init_db, save_prediction, fetch_history  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="Engagement Predictor", page_icon="📈", layout="centered")


@st.cache_resource
def load_models():
    reg = joblib.load(os.path.join(MODELS_DIR, "engagement_regressor.joblib"))
    clf = joblib.load(os.path.join(MODELS_DIR, "engagement_tier_classifier.joblib"))
    return reg, clf


init_db()

st.title("📈 Social Media Engagement Predictor")
st.caption(
    "Predicts expected engagement rate and tier for a post *before* you publish it, "
    "based on caption, timing, hashtags, and account size."
)

if not os.path.exists(os.path.join(MODELS_DIR, "engagement_regressor.joblib")):
    st.error("No trained model found. Run `python src/train_model.py` first.")
    st.stop()

reg_model, clf_model = load_models()

with st.form("post_form"):
    caption = st.text_area("Caption", "Excited to share this amazing new project! #grateful")
    col1, col2 = st.columns(2)
    with col1:
        content_type = st.selectbox("Content type", ["image", "video", "carousel", "text"])
        day_of_week = st.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=4,
        )
        posting_hour = st.slider("Posting hour (24h)", 0, 23, 19)
    with col2:
        hashtag_count = st.number_input("Number of hashtags", 0, 30, 6)
        mention_count = st.number_input("Number of mentions", 0, 10, 1)
        follower_count = st.number_input("Follower count", 50, 5_000_000, 5000, step=50)
    is_verified = st.checkbox("Verified account", value=False)
    submitted = st.form_submit_button("Predict engagement")

if submitted:
    X = build_features_single(
        caption=caption, content_type=content_type, day_of_week=day_of_week,
        posting_hour=posting_hour, hashtag_count=hashtag_count,
        mention_count=mention_count, follower_count=follower_count,
        is_verified=is_verified,
    )
    pred_rate = float(reg_model.predict(X)[0])
    pred_tier = clf_model.predict(X)[0]
    tier_proba = dict(zip(clf_model.classes_, clf_model.predict_proba(X)[0]))

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted engagement rate", f"{pred_rate*100:.2f}%")
    c2.metric("Predicted tier", pred_tier)

    st.bar_chart(pd.Series(tier_proba, name="probability"))

    est_impressions = int(follower_count * 0.6)
    st.caption(
        f"On an account with ~{follower_count:,} followers, that's roughly "
        f"**{int(est_impressions * pred_rate):,} estimated engagements** "
        f"(likes + comments + shares) at typical reach."
    )

    save_prediction(caption, content_type, day_of_week, posting_hour,
                     hashtag_count, follower_count, pred_rate, pred_tier)
    st.toast("Saved to history", icon="✅")

st.divider()
st.subheader("Recent predictions")
history = fetch_history(limit=10)
if history:
    st.dataframe(pd.DataFrame(history)[
        ["timestamp", "content_type", "day_of_week", "posting_hour",
         "hashtag_count", "predicted_engagement_rate", "predicted_tier"]
    ], use_container_width=True, hide_index=True)
else:
    st.info("No predictions logged yet — submit the form above.")

with st.expander("How the model works"):
    st.markdown(
        "- Trained on a synthetic-but-realistic dataset of 6,000 posts "
        "(`data/generate_synthetic_data.py`) with built-in patterns for "
        "content type, posting time, hashtag count, sentiment, and account size.\n"
        "- Two RandomForest models: a **regressor** for exact engagement rate, "
        "and a **classifier** for an easier-to-read Low/Medium/High/Viral tier.\n"
        "- Caption sentiment is computed with a lightweight lexicon "
        "(`src/feature_engineering.py::simple_sentiment`) — no external API needed.\n"
        "- Swap in a real dataset (see README) and re-run `src/train_model.py` "
        "to retrain on real data with the same pipeline."
    )
