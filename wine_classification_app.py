import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vinara · Wine Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Globals ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  .stApp {
    background: #0d0a0b;
    color: #e8ddd5;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

  /* ── Hero section ── */
  .hero {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    border-bottom: 1px solid #2e2325;
    padding-bottom: 2rem;
    margin-bottom: 2.5rem;
  }

  .hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #b0433a;
    margin-bottom: 0.5rem;
  }

  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 700;
    color: #f2ece6;
    line-height: 1.08;
    margin: 0 0 0.75rem;
  }

  .hero-title em {
    font-style: italic;
    color: #c0584f;
  }

  .hero-sub {
    font-size: 0.95rem;
    color: #8a7a72;
    max-width: 520px;
    line-height: 1.6;
  }

  /* ── Section labels ── */
  .section-label {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5c4a44;
    font-weight: 500;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e1719;
  }

  /* ── Slider overrides ── */
  .stSlider > div > div > div > div {
    background: #b0433a !important;
  }

  .stSlider > div > div > div {
    background: #2e2325 !important;
  }

  label[data-testid="stWidgetLabel"] {
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    color: #c4b3a9 !important;
    letter-spacing: 0.04em;
  }

  /* ── Predict button ── */
  div[data-testid="stButton"] > button {
    width: 100%;
    background: #b0433a;
    color: #f9f4f1;
    border: none;
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.85rem 2rem;
    margin-top: 1.5rem;
    cursor: pointer;
    transition: background 0.2s ease;
  }

  div[data-testid="stButton"] > button:hover {
    background: #c4524a;
  }

  /* ── Result card ── */
  .result-card {
    background: linear-gradient(135deg, #1a0f11 0%, #1f1416 100%);
    border: 1px solid #3d2a2d;
    border-left: 4px solid #b0433a;
    border-radius: 6px;
    padding: 2rem 2.2rem;
    margin-top: 2rem;
  }

  .result-label {
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6a4e4a;
    margin-bottom: 0.4rem;
  }

  .result-variety {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f2ece6;
    margin: 0;
  }

  .result-description {
    margin-top: 0.8rem;
    font-size: 0.88rem;
    color: #8a7a72;
    line-height: 1.6;
  }

  /* ── Divider ── */
  hr { border-color: #1e1719; margin: 2.5rem 0; }

  /* ── Metric tiles ── */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .metric-tile {
    background: #110d0e;
    border: 1px solid #2e2325;
    border-radius: 4px;
    padding: 1rem 1.2rem;
  }

  .metric-tile-label {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5c4a44;
    margin-bottom: 0.25rem;
  }

  .metric-tile-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #e8ddd5;
  }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('Wine_Classification_Model.pkl')

wine_model = load_model()

VARIETY_META = {
    'Variety A': {
        'description': 'Typically higher in alcohol and proline. Structured wines with bold phenolic profiles and deep color intensity.',
        'color': '#8B1A1A',
    },
    'Variety B': {
        'description': 'Balanced acidity and flavanoids. Moderate complexity with a refined mid-palate and approachable tannin structure.',
        'color': '#A0522D',
    },
    'Variety C': {
        'description': 'Higher malic acid expression, lighter color intensity. Fresh and aromatic with elevated nonflavanoid phenols.',
        'color': '#CD853F',
    },
}


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Wine Classification System</div>
  <h1 class="hero-title">Identify your <em>varietal</em><br>by composition</h1>
  <p class="hero-sub">
    Input the chemical profile of a wine sample below.
    The model classifies it into one of three Italian cultivar groupings.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">Structure & Acidity</div>', unsafe_allow_html=True)

    Alcohol           = st.slider('Alcohol (%)',               11.0,  15.0,  13.0,  0.01)
    Malic_acid        = st.slider('Malic Acid (g/L)',           0.70,   6.00,  2.30,  0.01)
    Ash               = st.slider('Ash (g/L)',                  1.30,   3.30,  2.30,  0.01)
    Alcalinity        = st.slider('Alcalinity of Ash',         10.00,  30.00, 19.50,  0.10)
    Magnesium         = st.slider('Magnesium (mg/L)',           60,    170,   100,    1)
    Color_intensity   = st.slider('Color Intensity',            1.00,  13.00,  5.00,  0.01)
    Hue               = st.slider('Hue',                        0.40,   2.00,  1.05,  0.01)

with col_right:
    st.markdown('<div class="section-label">Phenolics & Compounds</div>', unsafe_allow_html=True)

    Phenols                   = st.slider('Total Phenols',              0.80,  4.00,  2.30,  0.01)
    Flavanoids                = st.slider('Flavanoids',                 0.30,  6.00,  2.00,  0.01)
    Nonflavanoids             = st.slider('Nonflavanoid Phenols',       0.10,  0.70,  0.36,  0.01)
    Proanthocyanins           = st.slider('Proanthocyanins',            0.40,  4.00,  1.59,  0.01)
    OD280_315_of_diluted_wines = st.slider('OD280/OD315 Diluted Wines', 1.20,  4.00,  2.60,  0.01)
    Proline                   = st.slider('Proline (mg/L)',             270,  1680,   750,    1)


# ── Summary tiles ─────────────────────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-tile">
    <div class="metric-tile-label">Alcohol</div>
    <div class="metric-tile-value">{Alcohol:.1f}%</div>
  </div>
  <div class="metric-tile">
    <div class="metric-tile-label">Color Intensity</div>
    <div class="metric-tile-value">{Color_intensity:.2f}</div>
  </div>
  <div class="metric-tile">
    <div class="metric-tile-label">Proline</div>
    <div class="metric-tile-value">{Proline} mg/L</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Predict ───────────────────────────────────────────────────────────────────
classes = ['Variety A', 'Variety B', 'Variety C']

if st.button("Classify Wine →"):
    input_array = np.array([[Alcohol, Malic_acid, Ash, Alcalinity, Magnesium,
                              Phenols, Flavanoids, Nonflavanoids, Proanthocyanins,
                              Color_intensity, Hue, OD280_315_of_diluted_wines, Proline]])

    prediction = wine_model.predict(input_array)[0]
    wine_category = classes[prediction]
    meta = VARIETY_META[wine_category]

    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Classification Result</div>
      <p class="result-variety">{wine_category}</p>
      <p class="result-description">{meta['description']}</p>
    </div>
    """, unsafe_allow_html=True)
