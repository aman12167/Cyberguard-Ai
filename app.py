import streamlit as st
import pandas as pd
import joblib
import re

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CyberGuard AI | Scam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("scam_detector.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(ellipse at 50% -10%, #251044 0%, #0b0b19 45%, #070711 100%);
    color: #F8FAFC;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 17px 25px;
    border: 1px solid rgba(139,92,246,0.22);
    border-radius: 18px;
    background: rgba(19,15,36,0.85);
    backdrop-filter: blur(20px);
    margin-bottom: 65px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 11px;
    font-size: 21px;
    font-weight: 800;
    color: #FFFFFF;
}

.brand-icon {
    font-size: 25px;
}

.nav-links {
    display: flex;
    gap: 25px;
    align-items: center;
}

.nav-links a {
    color: #B9A8E8;
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    transition: 0.2s;
}

.nav-links a:hover {
    color: #FFFFFF;
}

.status {
    background: rgba(34,197,94,0.10);
    color: #4ADE80;
    border: 1px solid rgba(34,197,94,0.25);
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}

.hero {
    text-align: center;
    max-width: 900px;
    margin: 0 auto 45px;
}

.hero-badge {
    display: inline-block;
    padding: 9px 17px;
    border-radius: 30px;
    background: rgba(139,92,246,0.13);
    border: 1px solid rgba(167,139,250,0.30);
    color: #C4B5FD;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: clamp(34px, 5vw, 55px);
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -2px;
    color: #FFFFFF;
    margin-bottom: 18px;
}

.gradient-text {
    background: linear-gradient(100deg, #A78BFA, #60A5FA, #E879F9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    color: #9CA3AF;
    font-size: 15px;
    line-height: 1.8;
    max-width: 690px;
    margin: auto;
}

.section-label {
    color: #A78BFA;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.section-heading {
    font-size: 25px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 8px;
}

.section-description {
    color: #94A3B8;
    font-size: 13px;
    line-height: 1.7;
}

.analysis-panel {
    background: linear-gradient(145deg, rgba(30,23,53,0.96), rgba(16,17,35,0.96));
    border: 1px solid rgba(139,92,246,0.28);
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 20px 70px rgba(0,0,0,0.25);
    margin-top: 25px;
    margin-bottom: 35px;
}

.panel-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.panel-title {
    font-size: 18px;
    font-weight: 750;
    color: #FFFFFF;
}

.panel-tag {
    color: #A78BFA;
    font-size: 11px;
    font-weight: 700;
}

.stTextArea textarea {
    background: #0B0B19 !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    border-radius: 13px !important;
    font-size: 14px !important;
    padding: 16px !important;
    line-height: 1.7 !important;
}

.stTextArea textarea:focus {
    border-color: #A78BFA !important;
    box-shadow: 0 0 0 2px rgba(139,92,246,0.12) !important;
}

.stTextArea label {
    color: #CBD5E1 !important;
}

.stButton button {
    background: linear-gradient(100deg, #7C3AED, #6366F1, #A855F7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 20px !important;
    font-size: 14px !important;
    font-weight: 750 !important;
    transition: 0.25s ease !important;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.35);
}

.metric-card {
    background: rgba(24,24,48,0.85);
    border: 1px solid rgba(139,92,246,0.22);
    border-radius: 16px;
    padding: 22px;
    min-height: 125px;
}

.metric-label {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 13px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 27px;
    font-weight: 800;
}

.metric-sub {
    color: #A78BFA;
    font-size: 11px;
    margin-top: 5px;
}

.verdict {
    padding: 22px 25px;
    border-radius: 16px;
    margin-top: 20px;
    margin-bottom: 20px;
}

.verdict-scam {
    background: rgba(239,68,68,0.09);
    border: 1px solid rgba(239,68,68,0.35);
}

.verdict-safe {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.30);
}

.verdict-title {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 7px;
}

.verdict-text {
    color: #CBD5E1;
    font-size: 13px;
    line-height: 1.7;
}

.info-card {
    background: rgba(24,24,48,0.72);
    border: 1px solid rgba(139,92,246,0.18);
    border-radius: 16px;
    padding: 23px;
    height: 100%;
    min-height: 165px;
    transition: 0.25s ease;
}

.info-card:hover {
    border-color: rgba(167,139,250,0.55);
    transform: translateY(-4px);
}

.card-number {
    color: #A78BFA;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 15px;
}

.card-title {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 750;
    margin-bottom: 10px;
}

.card-text {
    color: #94A3B8;
    font-size: 12px;
    line-height: 1.8;
}

.tech-container {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 20px;
}

.tech-badge {
    background: rgba(139,92,246,0.11);
    border: 1px solid rgba(139,92,246,0.23);
    color: #C4B5FD;
    border-radius: 10px;
    padding: 11px 17px;
    font-size: 12px;
    font-weight: 650;
}

.tip-card {
    background: rgba(30,41,59,0.65);
    border: 1px solid rgba(96,165,250,0.15);
    border-radius: 15px;
    padding: 20px 23px;
}

.tip-item {
    color: #BFDBFE;
    font-size: 13px;
    line-height: 2.2;
}

.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.45), transparent);
    margin: 55px 0;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 11px;
    padding: 30px 0 10px;
}

@media (max-width: 700px) {
    .navbar {
        padding: 14px;
        margin-bottom: 40px;
    }
    .nav-links {
        display: none;
    }
    .hero-title {
        letter-spacing: -1px;
    }
    .analysis-panel {
        padding: 19px;
    }
    .status {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==========================================
# KEYWORD-BASED CATEGORY
# ==========================================

def detect_category(message):
    text = message.lower()

    categories = {
        "Financial Fraud": ["bank", "upi", "otp", "pin", "account", "payment"],
        "Phishing": ["click here", "verify", "login", "password", "link"],
        "Job Scam": ["part-time job", "work from home", "daily income", "telegram task"],
        "Digital Arrest": ["cbi", "police", "arrest", "warrant", "narcotics"],
        "Prize Scam": ["lottery", "winner", "prize", "reward"]
    }

    for category, keywords in categories.items():
        if any(word in text for word in keywords):
            return category

    return "General / Unknown"


# ==========================================
# NAVIGATION
# ==========================================

st.markdown("""
<div class="navbar">
    <div class="brand">
        <span class="brand-icon">🛡️</span>
        CyberGuard AI
    </div>
    <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#how-it-works">How It Works</a>
        <a href="#technology">Technology</a>
        <a href="#awareness">Cyber Safety</a>
    </div>
    <div class="status">
        ● SYSTEM ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero" id="overview">
    <div class="hero-badge">
        ⚡ INTELLIGENT CYBER THREAT DETECTION
    </div>
    <div class="hero-title">
        Detect Scams Before<br>
        They <span class="gradient-text">Reach You.</span>
    </div>
    <div class="hero-description">
        CyberGuard AI uses Machine Learning and Natural Language
        Processing to analyse suspicious messages and identify
        patterns commonly associated with digital fraud.
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# MESSAGE ANALYSIS PANEL
# ==========================================

left, center, right = st.columns([1, 3.2, 1])

with center:
    st.markdown("""
    <div class="analysis-panel">
        <div class="panel-top">
            <div class="panel-title">
                💬 Message Threat Scanner
            </div>
            <div class="panel-tag">
                ML-POWERED ANALYSIS
            </div>
        </div>
        <p style="color:#94A3B8;font-size:13px;">
            Paste a suspicious SMS, WhatsApp message or email
            below to analyse its potential threat level.
        </p>
    </div>
    """, unsafe_allow_html=True)

    user_message = st.text_area(
        "Message",
        height=155,
        placeholder="Paste your suspicious message here...",
        label_visibility="collapsed"
    )

    analyse_button = st.button(
        "✦ Analyse Message",
        use_container_width=True
    )


# ==========================================
# MESSAGE ANALYSIS
# ==========================================

if analyse_button:
    if not user_message.strip():
        st.warning("Please enter a message before starting analysis.")

    else:
        processed_message = clean_text(user_message)
        transformed_message = vectorizer.transform([processed_message])

        prediction = model.predict(transformed_message)[0]
        probabilities = model.predict_proba(transformed_message)[0]
        classes = list(model.classes_)

        if "scam" not in classes:
            st.error("The trained model does not contain a scam class.")
            st.stop()

        scam_probability = probabilities[classes.index("scam")]
        confidence = max(probabilities) * 100
        risk_score = scam_probability * 100

        category = detect_category(user_message)

        # ==================================
        # RESULTS HEADER
        # ==================================

        st.markdown(
            '<div class="custom-divider"></div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="section-label">
            ANALYSIS COMPLETE
        </div>
        <div class="section-heading">
            Threat Assessment Report
        </div>
        <div class="section-description">
            Here is what CyberGuard AI found in your message.
        </div>
        """, unsafe_allow_html=True)

        # ==================================
        # RESULT METRICS
        # ==================================

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    SCAM SCORE
                </div>
                <div class="metric-value">
                    {risk_score:.1f}%
                </div>
                <div class="metric-sub">
                    Match with common scam patterns
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    AI CONFIDENCE
                </div>
                <div class="metric-value">
                    {confidence:.1f}%
                </div>
                <div class="metric-sub">
                    How strongly the AI favours its result
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    KEYWORD CATEGORY
                </div>
                <div class="metric-value" style="font-size:19px;">
                    {category}
                </div>
                <div class="metric-sub">
                    Category identified from message words
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ==================================
        # VERDICT
        # ==================================

        if prediction == "scam":
            st.markdown("""
            <div class="verdict verdict-scam">
                <div class="verdict-title" style="color:#F87171;">
                    ⚠️ Potential Scam Detected
                </div>
                <div class="verdict-text">
                    This message contains patterns commonly
                    found in scam messages. Be careful!
                    Do not click suspicious links or share
                    your OTP, password or banking details.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="verdict verdict-safe">
                <div class="verdict-title" style="color:#4ADE80;">
                    ✓ No Strong Scam Patterns Found
                </div>
                <div class="verdict-text">
                    This message does not show strong scam
                    patterns based on the examples CyberGuard AI
                    has learned from. However, always check
                    the sender before trusting a message.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ==================================
        # RISK VISUALIZATION
        # ==================================

        st.markdown(
            "<p style='color:#CBD5E1;font-size:13px;font-weight:700;'>"
            "SCAM SCORE VISUALIZATION</p>",
            unsafe_allow_html=True
        )

        st.progress(float(scam_probability))

        st.caption(
            "A higher score means the message matches more "
            "patterns commonly found in scam messages. "
            "It does not guarantee that the message is a scam."
        )

        st.caption(
            "CyberGuard AI is a student project. Its results "
            "may not always be accurate, so verify suspicious "
            "messages before taking action."
        )


# ==========================================
# HOW IT WORKS
# ==========================================

st.markdown(
    '<div class="custom-divider"></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div id="how-it-works">
    <div class="section-label">
        THE PROCESS
    </div>
    <div class="section-heading">
        How CyberGuard AI Works
    </div>
    <div class="section-description">
        Three stages transform an ordinary message into
        a Machine Learning prediction.
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <div class="card-number">
            STEP 01
        </div>
        <div class="card-title">
            📝 Text Processing
        </div>
        <div class="card-text">
            The message is converted to lowercase,
            URLs are replaced and unnecessary
            characters are removed.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div class="card-number">
            STEP 02
        </div>
        <div class="card-title">
            🧠 Machine Learning
        </div>
        <div class="card-text">
            TF-IDF converts the processed text
            into numerical features which are
            analysed by the trained Naive Bayes model.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="card-number">
            STEP 03
        </div>
        <div class="card-title">
            🎯 Prediction
        </div>
        <div class="card-text">
            The trained model predicts whether
            the message resembles scam or
            legitimate examples.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TECHNOLOGY SECTION
# ==========================================

st.markdown(
    '<div class="custom-divider"></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div id="technology">
    <div class="section-label">
        UNDER THE HOOD
    </div>
    <div class="section-heading">
        Technology Behind CyberGuard AI
    </div>
    <div class="section-description">
        Built using Python libraries and Machine Learning tools.
    </div>
    <div class="tech-container">
        <div class="tech-badge">Python</div>
        <div class="tech-badge">Streamlit</div>
        <div class="tech-badge">Pandas</div>
        <div class="tech-badge">Scikit-learn</div>
        <div class="tech-badge">TF-IDF</div>
        <div class="tech-badge">Multinomial Naive Bayes</div>
        <div class="tech-badge">Joblib</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# CYBER SAFETY SECTION
# ==========================================

st.markdown(
    '<div class="custom-divider"></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div id="awareness">
    <div class="section-label">
        STAY AWARE
    </div>
    <div class="section-heading">
        🛡️ Stay Protected Online
    </div>
    <div class="section-description">
        Simple practices that can help protect you
        from common forms of digital fraud.
    </div>
</div>
<br>
<div class="tip-card">
    <div class="tip-item">
        01 &nbsp; Never share your OTP or UPI PIN.
    </div>
    <div class="tip-item">
        02 &nbsp; Avoid clicking unknown or suspicious links.
    </div>
    <div class="tip-item">
        03 &nbsp; Verify banking messages through official channels.
    </div>
    <div class="tip-item">
        04 &nbsp; Be cautious of unrealistic job and investment offers.
    </div>
    <div class="tip-item">
        05 &nbsp; Never trust threats demanding immediate payments.
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="custom-divider"></div>
<div class="footer">
    <div style="font-size:17px;font-weight:800;color:#C4B5FD;">
        🛡️ CyberGuard AI
    </div>
    <p>
        Intelligent Scam Detection Through Machine Learning
    </p>
    <p>
        Class XII Artificial Intelligence Project
    </p>
    <p style="color:#475569;">
        Educational Prototype | Results may not always be accurate.
    </p>
</div>
""", unsafe_allow_html=True)
