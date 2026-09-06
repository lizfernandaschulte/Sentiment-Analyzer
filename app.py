import streamlit as st
import pickle
import re
import time
import nltk
from nltk.corpus import stopwords


# PAGE CONFIG
st.set_page_config(
    page_title="Sentimental Analyzer",
    layout="wide"
)


# CSS
st.markdown("""
<style>
    /* hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* global background */
    .stApp {
        background-color: #0a0a0f;
        color: #e8e8f0;
    }

    /* navbar */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 24px;
        background: #111118;
        border-bottom: 1px solid #1e1e2e;
        margin-bottom: 32px;
        border-radius: 0 0 12px 12px;
    }
    .navbar-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .navbar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .navbar-badge {
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 99px;
        background: #1e1e2e;
        color: #888;
        border: 1px solid #2e2e3e;
    }
    .badge-green {
        background: rgba(0,255,136,0.1);
        color: #00ff88;
        border-color: rgba(0,255,136,0.3);
    }
    .navbar-acc {
        font-size: 0.85rem;
        color: #00ff88;
        font-weight: 600;
    }

    /* input card */
    .input-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-header {
        font-size: 0.75rem;
        color: #888;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
    }

    /* result card */
    .result-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 24px;
        height: 100%;
    }
    .result-header {
        font-size: 0.7rem;
        color: #888;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .result-latency {
        color: #00ff88;
        font-size: 0.7rem;
    }

    /* sentiment badge */
    .sentiment-positive {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 99px;
        background: rgba(0,255,136,0.15);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.4);
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 8px;
    }
    .sentiment-negative {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 99px;
        background: rgba(255,80,80,0.15);
        color: #ff5050;
        border: 1px solid rgba(255,80,80,0.4);
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 8px;
    }

    /* confidence */
    .confidence-number {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -2px;
        line-height: 1;
    }
    .confidence-label {
        font-size: 0.75rem;
        color: #888;
        margin-bottom: 12px;
    }

    /* progress bar */
    .progress-wrap {
        background: #1e1e2e;
        border-radius: 99px;
        height: 6px;
        margin: 12px 0;
        overflow: hidden;
    }
    .progress-fill-positive {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #00ff88, #00cc6a);
    }
    .progress-fill-negative {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #ff5050, #cc0000);
    }

    /* metrics row */
    .metrics-row {
        display: flex;
        justify-content: space-between;
        margin-top: 8px;
        font-size: 0.8rem;
    }
    .metric-positive { color: #00ff88; }
    .metric-negative { color: #ff5050; }

    /* divider */
    .divider {
        border: none;
        border-top: 1px solid #1e1e2e;
        margin: 20px 0;
    }

    /* stat grid */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 16px;
    }
    .stat-item {
        background: #0a0a0f;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        padding: 12px;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #888;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e8e8f0;
    }

    /* keyword tags */
    .keyword-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }
    .keyword-tag {
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 99px;
        background: #1e1e2e;
        color: #aaa;
        border: 1px solid #2e2e3e;
    }

    /* bottom stats */
    .bottom-stats {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
        margin-top: 24px;
    }
    .bottom-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 20px;
    }
    .bottom-card-tag {
        font-size: 0.65rem;
        color: #00ff88;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .bottom-card-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }
    .bottom-card-desc {
        font-size: 0.8rem;
        color: #888;
        line-height: 1.5;
    }

    /* analyze button */
    .stButton > button {
        background: linear-gradient(135deg, #7c6fff, #5a4fcf) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* textarea */
    .stTextArea textarea {
        background: #0a0a0f !important;
        color: #e8e8f0 !important;
        border: 1px solid #1e1e2e !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    useful_words = [w for w in words if w not in STOP_WORDS]
    return ' '.join(useful_words)

def get_keywords(text, n=5):
    cleaned = clean_text(text)
    words = cleaned.split()
    # get unique words sorted by length (longer = more meaningful)
    seen = set()
    keywords = []
    for w in words:
        if w not in seen and len(w) > 4:
            seen.add(w)
            keywords.append(w)
    return keywords[:n]


# NAVBAR
st.markdown("""
<div class="navbar">
    <div class="navbar-left">
        <span class="navbar-title">SentimentAnalyzer</span>
        <span class="navbar-badge badge-green">● Trained on 50k IMDB reviews</span>
        <span class="navbar-badge">Logistic Regression</span>
    </div>
    <span class="navbar-acc">89.24% ACC</span>
</div>
""", unsafe_allow_html=True)


# MAIN LAYOUT
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#fff;letter-spacing:-1px;margin-bottom:8px;">
            Movie Review Sentiment Classifier
        </h1>
        <p style="color:#888;font-size:1rem;line-height:1.6;">
            Enter any movie review or text and the model will predict 
            whether the sentiment is positive or negative using NLP.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # input card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><span>📝 Text Input</span></div>', unsafe_allow_html=True)

    user_input = st.text_area(
        label="",
        placeholder="e.g. This movie was absolutely amazing, I loved every second of it...",
        height=160,
        label_visibility="collapsed"
    )

    word_count = len(user_input.split()) if user_input.strip() else 0
    char_count = len(user_input)
    st.markdown(f'<div style="font-size:0.75rem;color:#888;margin-top:8px;">● Auto-cleaning active &nbsp;·&nbsp; {word_count} words · {char_count} characters</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    analyze = st.button("✦ Analyze Sentiment")

with col_right:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if not analyze or not user_input.strip():
        st.markdown("""
        <div class="result-header">
            <span>CLASSIFICATION RESULT</span>
        </div>
        <div style="text-align:center;padding:40px 0;color:#333;">
            <div style="font-size:3rem;margin-bottom:12px;">🧠</div>
            <div style="font-size:0.85rem;color:#555;">Waiting for input...</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        start = time.time()
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        latency = (time.time() - start) * 1000
        confidence = max(probabilities) * 100
        pos_prob = probabilities[1] * 100
        neg_prob = probabilities[0] * 100
        keywords = get_keywords(user_input)

        st.markdown(f"""
        <div class="result-header">
            <span>CLASSIFICATION RESULT</span>
            <span class="result-latency">● {latency:.1f}ms</span>
        </div>
        """, unsafe_allow_html=True)

        if prediction == "positive":
            st.markdown("""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                <span style="font-size:2rem;">😊</span>
                <span class="sentiment-positive">▲ POSITIVE SENTIMENT</span>
            </div>
            <div style="font-size:0.85rem;color:#888;margin-bottom:20px;">Enthusiastic approval detected in the text.</div>
            """, unsafe_allow_html=True)
            fill_class = "progress-fill-positive"
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                <span style="font-size:2rem;">😞</span>
                <span class="sentiment-negative">▼ NEGATIVE SENTIMENT</span>
            </div>
            <div style="font-size:0.85rem;color:#888;margin-bottom:20px;">Critical or negative tone detected in the text.</div>
            """, unsafe_allow_html=True)
            fill_class = "progress-fill-negative"

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-end;">
            <div>
                <div class="confidence-label">Neural Confidence</div>
                <div class="confidence-number">{confidence:.1f}%</div>
            </div>
            <div style="font-size:0.75rem;color:#888;">prob.</div>
        </div>
        <div class="progress-wrap">
            <div class="{fill_class}" style="width:{confidence}%;"></div>
        </div>
        <div class="metrics-row">
            <span class="metric-positive">● Positive: {pos_prob:.1f}%</span>
            <span class="metric-negative">● Negative: {neg_prob:.1f}%</span>
        </div>
        <hr class="divider">
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-label">Algorithmic Certainty</div>
                <div class="stat-value">{"High (Tier 1)" if confidence > 85 else "Medium (Tier 2)" if confidence > 70 else "Low (Tier 3)"}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Words Analyzed</div>
                <div class="stat-value">{word_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if keywords:
            kw_html = ''.join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
            st.markdown(f"""
            <hr class="divider">
            <div style="font-size:0.75rem;color:#888;margin-bottom:8px;">KEY TERMS DETECTED</div>
            <div class="keyword-wrap">{kw_html}</div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# BOTTOM STATS
st.markdown("""
<div class="bottom-stats">
    <div class="bottom-card">
        <div class="bottom-card-tag">50:50 Balanced</div>
        <div class="bottom-card-number">50,000</div>
        <div style="font-size:0.8rem;color:#888;margin-bottom:12px;">IMDB Reviews</div>
        <div class="bottom-card-desc">25,000 positive and 25,000 negative reviews. Zero class bias.</div>
        <div style="background:#1e1e2e;border-radius:99px;height:4px;margin-top:12px;overflow:hidden;">
            <div style="width:50%;height:100%;background:linear-gradient(90deg,#00ff88,#ff5050);border-radius:99px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#888;margin-top:6px;">
            <span>25k Positive</span><span>25k Negative</span>
        </div>
    </div>
    <div class="bottom-card">
        <div class="bottom-card-tag">Model Performance</div>
        <div class="bottom-card-number">89.24%</div>
        <div style="font-size:0.8rem;color:#888;margin-bottom:12px;">Accuracy</div>
        <div class="bottom-card-desc">Tested on 10,000 unseen reviews. Recall: 91% · Precision: 88%</div>
    </div>
    <div class="bottom-card">
        <div class="bottom-card-tag">NLP Pipeline</div>
        <div class="bottom-card-number">TF-IDF</div>
        <div style="font-size:0.8rem;color:#888;margin-bottom:12px;">Vectorization</div>
        <div class="bottom-card-desc">10,000 features extracted. HTML cleaning + stopword removal applied.</div>
    </div>
</div>
""", unsafe_allow_html=True)