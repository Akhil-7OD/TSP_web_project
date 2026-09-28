import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ship Passenger Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Titanic Nautical Theme (Navy, Slate, Ocean Blue, Gold accents)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Primary Container Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Maritime Hero Header */
    .hero-card {
        background: linear-gradient(135deg, #0b192c 0%, #102a43 50%, #1e3e62 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        color: #f8fafc;
        box-shadow: 0 20px 40px -15px rgba(11, 25, 44, 0.6);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-card::after {
        content: "⚓";
        position: absolute;
        right: 2rem;
        bottom: -1rem;
        font-size: 11rem;
        opacity: 0.05;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 6px 14px;
        border-radius: 9999px;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.75rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.75rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.35rem;
        font-weight: 500;
        color: #93c5fd;
        margin-bottom: 1rem;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 780px;
        line-height: 1.65;
        margin-bottom: 1.5rem;
    }
    .hero-clarification {
        background: rgba(15, 23, 42, 0.65);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 12px 18px;
        font-size: 0.95rem;
        color: #e2e8f0;
        margin-bottom: 1.75rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Custom Glass/Card containers */
    .ocean-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .ocean-card:hover {
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    [data-theme="dark"] .ocean-card {
        background: #0f1f33;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* Result Card - Survived (Emerald) */
    .result-card-survived {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 2.25rem;
        color: #ffffff;
        box-shadow: 0 15px 35px -10px rgba(16, 185, 129, 0.45);
        margin: 1.5rem 0;
        text-align: center;
        animation: fadeIn 0.4s ease-in-out;
    }
    .result-card-survived h2 {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ecfdf5;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* Result Card - Did Not Survive (Crimson) */
    .result-card-not-survived {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 50%, #b91c1c 100%);
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 2.25rem;
        color: #ffffff;
        box-shadow: 0 15px 35px -10px rgba(239, 68, 68, 0.45);
        margin: 1.5rem 0;
        text-align: center;
        animation: fadeIn 0.4s ease-in-out;
    }
    .result-card-not-survived h2 {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #fef2f2;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* Step Workflow Cards */
    .flow-step {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        height: 100%;
        position: relative;
    }
    [data-theme="dark"] .flow-step {
        background: #132438;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .step-number {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        border-radius: 50%;
        background: #0284c7;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .step-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
        color: #0f2744;
    }
    [data-theme="dark"] .step-title {
        color: #93c5fd;
    }
    .step-desc {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.4;
    }
    [data-theme="dark"] .step-desc {
        color: #94a3b8;
    }

    /* Metric stat pill */
    .metric-pill {
        background: rgba(2, 132, 199, 0.08);
        border: 1px solid rgba(2, 132, 199, 0.2);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 4px 0;
    }

    /* Ship Frame Showcase */
    .ship-frame {
        background: linear-gradient(145deg, #0b192c 0%, #102a43 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 10px;
        box-shadow: 0 20px 40px -10px rgba(11, 25, 44, 0.7);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .ship-frame:hover {
        transform: translateY(-3px);
        box-shadow: 0 25px 50px -10px rgba(56, 189, 248, 0.25);
    }
    .ship-caption {
        text-align: center;
        font-size: 0.88rem;
        color: #94a3b8;
        padding: 8px 4px 4px 4px;
        line-height: 1.4;
    }

    /* Stat Cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }
    [data-theme="dark"] .stat-card {
        background: #0f1f33;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0284c7;
        margin-bottom: 0.2rem;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "preset_data" not in st.session_state:
    st.session_state.preset_data = {
        "pclass": "1st Class",
        "sex": "Female",
        "age": 29.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 100.0,
        "embarked": "Cherbourg (C)"
    }

def navigate_to(page_name):
    st.session_state.current_page = page_name

def apply_preset(pclass, sex, age, sibsp, parch, fare, embarked):
    st.session_state.preset_data = {
        "pclass": pclass,
        "sex": sex,
        "age": float(age),
        "sibsp": int(sibsp),
        "parch": int(parch),
        "fare": float(fare),
        "embarked": embarked
    }

# -----------------------------------------------------------------------------
# 3. MACHINE LEARNING MODEL & PIPELINE (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_train_model():
    """
    Loads train.csv, applies exact preprocessing specified by project guidelines,
    and trains a RandomForestClassifier cached across sessions.
    """
    possible_paths = [
        "train.csv",
        os.path.join(os.path.dirname(__file__), "train.csv"),
        os.path.join("data", "train.csv")
    ]
    
    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        return None, None, "Unable to load the training data. Please make sure train.csv is present in the project directory."

    try:
        df = pd.read_csv(csv_path)

        # Preprocessing Step 1: Missing values handling
        # - Age missing values filled with median Age
        age_median = float(df['Age'].median())
        df['Age'] = df['Age'].fillna(age_median)

        # - Embarked missing values filled with mode
        embarked_mode = str(df['Embarked'].mode()[0])
        df['Embarked'] = df['Embarked'].fillna(embarked_mode)

        # - Fare missing values filled with median Fare
        fare_median = float(df['Fare'].median())
        df['Fare'] = df['Fare'].fillna(fare_median)

        # Preprocessing Step 2: Binary Sex encoding (male = 0, female = 1)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

        # Preprocessing Step 3: One-Hot Encoding for Embarked
        df = pd.get_dummies(df, columns=['Embarked'], dtype=int)

        # Ensure all three dummy columns exist (Embarked_C, Embarked_Q, Embarked_S)
        for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
            if col not in df.columns:
                df[col] = 0

        # Exact 9 final features in specified order:
        # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q, Embarked_S
        feature_cols = [
            'Pclass',
            'Sex',
            'Age',
            'SibSp',
            'Parch',
            'Fare',
            'Embarked_C',
            'Embarked_Q',
            'Embarked_S'
        ]

        X = df[feature_cols]
        Y = df['Survived']

        # Split: train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=42,
            stratify=Y
        )

        # Model: RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        model.fit(x_train, y_train)

        # Compute accuracy metrics
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        # Feature importances
        importances = dict(zip(feature_cols, model.feature_importances_))

        metadata = {
            "accuracy": acc,
            "total_samples": len(df),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "age_median": age_median,
            "fare_median": fare_median,
            "embarked_mode": embarked_mode,
            "feature_cols": feature_cols,
            "feature_importances": importances
        }

        return model, metadata, None

    except Exception as e:
        return None, None, f"An unexpected error occurred while training the model: {str(e)}"

# Load the model
model, model_metadata, load_error = load_and_train_model()

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🚢</div>
            <h2 style="font-family: 'Playfair Display', serif; margin: 0; font-size: 1.4rem; color: #38bdf8;">
                Titanic Disaster
            </h2>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem;">
                Passenger Survival Prediction
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Radio
    nav_selection = st.radio(
        "Navigation",
        options=["Home", "Predict", "About"],
        index=["Home", "Predict", "About"].index(st.session_state.current_page),
        label_visibility="collapsed"
    )
    
    if nav_selection != st.session_state.current_page:
        st.session_state.current_page = nav_selection
        st.rerun()

    st.markdown("---")

    # Project metadata quick card
    st.markdown("""
        <div class="metric-pill">
            <small style="color: #64748b; font-weight: 600; text-transform: uppercase;">Algorithm</small>
            <div style="font-weight: 600; color: #0284c7;">Random Forest Classifier</div>
        </div>
        <div class="metric-pill">
            <small style="color: #64748b; font-weight: 600; text-transform: uppercase;">Features</small>
            <div style="font-weight: 600; color: #0284c7;">9 Demographic & Ticket Factors</div>
        </div>
        <div class="metric-pill">
            <small style="color: #64748b; font-weight: 600; text-transform: uppercase;">Target Variable</small>
            <div style="font-weight: 600; color: #0284c7;">Passenger Survival (0 / 1)</div>
        </div>
    """, unsafe_allow_html=True)

    if model_metadata:
        st.markdown(f"""
            <div class="metric-pill">
                <small style="color: #64748b; font-weight: 600; text-transform: uppercase;">Test Set Accuracy</small>
                <div style="font-weight: 700; color: #10b981; font-size: 1.1rem;">
                    {model_metadata['accuracy'] * 100:.1f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with Python, Streamlit, Pandas & Scikit-learn.")

# -----------------------------------------------------------------------------
# 5. ERROR DISPLAY (IF DATASET OR TRAINING FAILED)
# -----------------------------------------------------------------------------
if load_error:
    st.error(f"⚠️ {load_error}")
    st.info("Please place `train.csv` in the root folder of this project to enable model training.")
    st.stop()

# -----------------------------------------------------------------------------
# 6. PAGE: HOME
# -----------------------------------------------------------------------------
if st.session_state.current_page == "Home":
    # Hero Section
    st.markdown("""
        <div class="hero-card">
            <div class="hero-badge">Machine Learning Application</div>
            <h1 class="hero-title">Ship Passenger Survival Prediction</h1>
            <div class="hero-subtitle">Predict passenger survival using Machine Learning</div>
            <div class="hero-desc">
                Enter passenger details and let our Machine Learning model predict whether the passenger 
                would have survived the Titanic disaster.
            </div>
            <div class="hero-clarification">
                <span style="font-size: 1.25rem;">ℹ️</span>
                <span>
                    <strong>Important Note:</strong> This application predicts whether an <em>individual passenger</em> 
                    would have survived based on personal and ticketing factors. It is <strong>NOT</strong> a prediction about 
                    whether the ship itself survived.
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Hero CTA Button
    hero_col1, hero_col2, hero_col3 = st.columns([1.5, 2, 1.5])
    with hero_col2:
        if st.button("Predict Passenger Survival →", type="primary", use_container_width=True):
            navigate_to("Predict")
            st.rerun()

    st.write("")
    st.write("")

    # Feature Highlights Row
    st.markdown("### 🌟 Project Highlights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="ocean-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌲</div>
                <h3 style="margin-top: 0; font-size: 1.2rem; color: #0284c7;">Random Forest ML</h3>
                <p style="color: #64748b; font-size: 0.95rem; line-height: 1.5;">
                    Utilizes an ensemble of decision trees trained on the historical Titanic dataset 
                    with stratified validation for high classification reliability.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="ocean-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <h3 style="margin-top: 0; font-size: 1.2rem; color: #0284c7;">9 Clean Features</h3>
                <p style="color: #64748b; font-size: 0.95rem; line-height: 1.5;">
                    Carefully curated inputs: Passenger Class, Sex, Age, Siblings/Spouse, Parents/Children, 
                    Ticket Fare, and One-Hot Encoded Embarkation ports.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="ocean-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <h3 style="margin-top: 0; font-size: 1.2rem; color: #0284c7;">Confidence Scores</h3>
                <p style="color: #64748b; font-size: 0.95rem; line-height: 1.5;">
                    Beyond binary classification, inspect the model's statistical survival and non-survival 
                    probabilities with responsible ethical AI wording.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Historical Context Card
    st.markdown("""
        <div class="ocean-card" style="border-left: 5px solid #0284c7;">
            <h4 style="margin: 0 0 0.5rem 0; color: #0f2744;">The Tragedy of RMS Titanic (April 15, 1912)</h4>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                During her maiden voyage from Southampton to New York City, the RMS Titanic struck an iceberg in the North Atlantic. 
                Of the estimated 2,224 passengers and crew aboard, more than 1,500 perished. 
                Historical analysis reveals that survival was not purely random: socio-economic class, gender ("women and children first" maritime protocol), 
                and boarding location significantly influenced survival probability.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. PAGE: PREDICTION (MAIN FEATURE)
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "Predict":
    st.markdown("""
        <div>
            <h1 style="font-family: 'Playfair Display', serif; color: #0f2744; margin-bottom: 0.25rem;">
                Passenger Survival Prediction
            </h1>
            <p style="color: #64748b; font-size: 1.15rem; margin-bottom: 1.5rem;">
                Enter the passenger's information below.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Preset Quick Loaders
    with st.expander("⚡ Quick Test Profiles (Click to prefill passenger details)", expanded=False):
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            if st.button("👩 1st Class Female\n(Cherbourg)", use_container_width=True):
                apply_preset("1st Class", "Female", 29.0, 0, 0, 100.0, "Cherbourg (C)")
                st.rerun()
        with p_col2:
            if st.button("👨 3rd Class Male\n(Southampton)", use_container_width=True):
                apply_preset("3rd Class", "Male", 25.0, 0, 0, 7.25, "Southampton (S)")
                st.rerun()
        with p_col3:
            if st.button("👨‍👩‍👧 2nd Class Child\n(Southampton)", use_container_width=True):
                apply_preset("2nd Class", "Female", 8.0, 1, 1, 26.0, "Southampton (S)")
                st.rerun()
        with p_col4:
            if st.button("🔄 Reset to Default\n(Median Values)", use_container_width=True):
                apply_preset(
                    "3rd Class",
                    "Male",
                    model_metadata["age_median"],
                    0,
                    0,
                    model_metadata["fare_median"],
                    "Southampton (S)"
                )
                st.rerun()

    preset = st.session_state.preset_data

    # Input Form Container
    st.markdown('<div class="ocean-card">', unsafe_allow_html=True)
    with st.form("passenger_form"):
        col_left, col_mid, col_right = st.columns(3)

        # Field A: Passenger Class
        with col_left:
            pclass_selection = st.selectbox(
                "Passenger Class",
                options=["1st Class", "2nd Class", "3rd Class"],
                index=["1st Class", "2nd Class", "3rd Class"].index(preset["pclass"]),
                help="1st Class = Upper socio-economic class, 2nd Class = Middle class, 3rd Class = Lower/Steerage class."
            )

            # Field B: Gender
            sex_selection = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                index=["Male", "Female"].index(preset["sex"]),
                help="Gender of the passenger. Historically, the maritime protocol 'women and children first' strongly favored female survival."
            )

            # Field G: Port of Embarkation
            embarked_selection = st.selectbox(
                "Port of Embarkation",
                options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
                index=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"].index(preset["embarked"]),
                help="The port where the passenger boarded the Titanic: Southampton (UK), Cherbourg (France), or Queenstown (Cobh, Ireland)."
            )

        # Field C: Age & Field F: Fare
        with col_mid:
            age_input = st.number_input(
                "Age",
                min_value=0.42,
                max_value=90.0,
                value=float(preset["age"]),
                step=1.0,
                help="Age in years. Fractional for infants less than 1 year old."
            )

            fare_input = st.number_input(
                "Ticket Fare",
                min_value=0.0,
                max_value=600.0,
                value=float(preset["fare"]),
                step=5.0,
                format="%.2f",
                help="Ticket price paid in 1912 British Pounds (£). 1st Class fares were significantly higher."
            )

        # Field D: SibSp & Field E: Parch
        with col_right:
            sibsp_input = st.number_input(
                "Siblings / Spouses Aboard",
                min_value=0,
                max_value=10,
                value=int(preset["sibsp"]),
                step=1,
                help="Number of siblings (brother, sister, stepbrother, stepsister) or spouses (husband, wife) aboard."
            )

            parch_input = st.number_input(
                "Parents / Children Aboard",
                min_value=0,
                max_value=10,
                value=int(preset["parch"]),
                step=1,
                help="Number of parents (mother, father) or children (daughter, son, stepdaughter, stepson) aboard."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.form_submit_button(
            "🔮 Predict Survival",
            type="primary",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Action
    if predict_clicked:
        # Convert inputs exactly as required
        pclass_map = {"1st Class": 1, "2nd Class": 2, "3rd Class": 3}
        pclass_val = pclass_map[pclass_selection]

        sex_val = 0 if sex_selection == "Male" else 1

        # Embarked one-hot encoding
        embarked_c_val = 1 if "Cherbourg" in embarked_selection else 0
        embarked_q_val = 1 if "Queenstown" in embarked_selection else 0
        embarked_s_val = 1 if "Southampton" in embarked_selection else 0

        # Construct DataFrame in exact specified order:
        # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q, Embarked_S
        input_data = pd.DataFrame([{
            'Pclass': pclass_val,
            'Sex': sex_val,
            'Age': float(age_input),
            'SibSp': int(sibsp_input),
            'Parch': int(parch_input),
            'Fare': float(fare_input),
            'Embarked_C': embarked_c_val,
            'Embarked_Q': embarked_q_val,
            'Embarked_S': embarked_s_val
        }])

        try:
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]
            prob_not_survived = probabilities[0]
            prob_survived = probabilities[1]

            # -----------------------------------------------------------------
            # 8. RESULT DISPLAY
            # -----------------------------------------------------------------
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-card-survived">
                        <div style="font-size: 3.5rem;">🎉</div>
                        <h2>🟢 PASSENGER SURVIVED</h2>
                        <p style="font-size: 1.15rem; margin: 0.5rem 0 0 0; color: #d1fae5;">
                            According to the Machine Learning model, this passenger is predicted to have survived.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card-not-survived">
                        <div style="font-size: 3.5rem;">⚠️</div>
                        <h2>🔴 PASSENGER DID NOT SURVIVE</h2>
                        <p style="font-size: 1.15rem; margin: 0.5rem 0 0 0; color: #fee2e2;">
                            According to the Machine Learning model, this passenger is predicted not to have survived.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # Model Confidence / Probability Card
            st.markdown('<div class="ocean-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Model Confidence / Probability")
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric(
                    label="Survival Probability",
                    value=f"{prob_survived * 100:.1f}%",
                    delta=f"{'+' if prob_survived > 0.5 else ''}{(prob_survived - 0.5) * 100:.1f}% vs baseline",
                    delta_color="normal" if prob_survived >= 0.5 else "inverse"
                )
                st.progress(float(prob_survived))

            with p_col2:
                st.metric(
                    label="Non-Survival Probability",
                    value=f"{prob_not_survived * 100:.1f}%",
                    delta=f"{'+' if prob_not_survived > 0.5 else ''}{(prob_not_survived - 0.5) * 100:.1f}% vs baseline",
                    delta_color="inverse" if prob_not_survived >= 0.5 else "normal"
                )
                st.progress(float(prob_not_survived))

            st.markdown("""
                <div style="background: rgba(148, 163, 184, 0.1); border-radius: 8px; padding: 12px; margin-top: 1rem;">
                    <small style="color: #64748b; line-height: 1.4; display: block;">
                        <strong>Ethical AI & Disclaimer:</strong> The probabilities displayed represent 
                        statistical model confidence generated by the Random Forest Classifier based on patterns in the 
                        1912 Titanic passenger manifest. They do not constitute an absolute guarantee of historical outcome.
                    </small>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Model Input Feature Inspection
            with st.expander("🔍 Inspect Model Input Features (Processed 9-Vector)", expanded=False):
                st.dataframe(input_data, use_container_width=True)
                st.caption("Values correspond directly to the 9 inputs fed to `RandomForestClassifier.predict()`.")

        except Exception as pred_err:
            st.error("An unexpected error occurred while making the prediction. Please verify your inputs.")

# -----------------------------------------------------------------------------
# 8. PAGE: ABOUT
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "About":
    st.markdown("""
        <div>
            <h1 style="font-family: 'Playfair Display', serif; color: #0f2744; margin-bottom: 0.25rem;">
                About This Project
            </h1>
            <p style="color: #64748b; font-size: 1.15rem; margin-bottom: 1.5rem;">
                This project demonstrates how a Machine Learning classification model can be integrated into an interactive web application.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # How It Works Workflow
    st.markdown("### 🔄 How It Works")
    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("""
            <div class="flow-step">
                <div class="step-number">1</div>
                <div class="step-title">Passenger Details</div>
                <div class="step-desc">
                    User inputs passenger attributes: Class, Sex, Age, Family, Fare, and Embarkation port.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
            <div class="flow-step">
                <div class="step-number">2</div>
                <div class="step-title">Data Preprocessing</div>
                <div class="step-desc">
                    Missing values imputed (median/mode), Sex mapped (0/1), and Embarked one-hot encoded into 3 binary columns.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
            <div class="flow-step">
                <div class="step-number">3</div>
                <div class="step-title">Random Forest ML</div>
                <div class="step-desc">
                    Ensemble of decision trees evaluates feature splits to aggregate individual tree votes.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with step4:
        st.markdown("""
            <div class="flow-step">
                <div class="step-number">4</div>
                <div class="step-title">Survival Prediction</div>
                <div class="step-desc">
                    Outputs final classification (🟢 Survived / 🔴 Did Not Survive) alongside confidence probabilities.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Model & Feature Specifications
    spec_col1, spec_col2 = st.columns(2)

    with spec_col1:
        st.markdown('<div class="ocean-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Machine Learning Specification")
        st.markdown(f"""
            - **Model Algorithm:** `RandomForestClassifier()`
            - **Training Dataset:** Titanic Passenger Dataset (`train.csv`)
            - **Target Column:** `Survived` (0 = Did not survive, 1 = Survived)
            - **Data Split:** 80% Training, 20% Testing (`stratify=Y`, `random_state=42`)
            - **Test Set Accuracy:** **{model_metadata['accuracy'] * 100:.2f}%**
            - **Total Training Records:** {model_metadata['total_samples']} passengers
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with spec_col2:
        st.markdown('<div class="ocean-card">', unsafe_allow_html=True)
        st.markdown("### 📋 9 Model Features Used")
        st.markdown("""
            1. **`Pclass`** — Passenger ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
            2. **`Sex`** — Passenger gender (`male = 0`, `female = 1`)
            3. **`Age`** — Age in years (missing imputed with median: 28.0)
            4. **`SibSp`** — Sibling / Spouse count aboard
            5. **`Parch`** — Parent / Child count aboard
            6. **`Fare`** — Passenger ticket fare (£) (missing imputed with median: £14.45)
            7. **`Embarked_C`** — Port Cherbourg (1 = Yes, 0 = No)
            8. **`Embarked_Q`** — Port Queenstown (1 = Yes, 0 = No)
            9. **`Embarked_S`** — Port Southampton (1 = Yes, 0 = No)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Excluded Features Note
    st.markdown("""
        <div class="ocean-card" style="border-left: 5px solid #f59e0b;">
            <h4 style="margin: 0 0 0.5rem 0; color: #b45309;">🚫 Excluded Features & Rationale</h4>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                The following raw columns from <code>train.csv</code> are intentionally <strong>NOT</strong> used in prediction:
            </p>
            <ul style="color: #475569; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 0;">
                <li><strong>Cabin:</strong> Excluded due to very high proportion of missing values (>77% missing).</li>
                <li><strong>PassengerId:</strong> Excluded because arbitrary sequential row identifiers carry no generalizable predictive signal.</li>
                <li><strong>Name:</strong> Excluded to avoid overfitting on specific passenger surnames and strings.</li>
                <li><strong>Ticket:</strong> Excluded due to high cardinality of non-standard ticket numbers.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Feature Importance Breakdown
    if model_metadata and "feature_importances" in model_metadata:
        st.markdown('<div class="ocean-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Random Forest Feature Importances")
        imp_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in model_metadata["feature_importances"].items()
        ]).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        st.bar_chart(imp_df.set_index("Feature"))
        st.caption("Relative Gini importance of the 9 features in the trained Random Forest model. Notice that Sex, Fare, and Age are primary survival determinants.")
        st.markdown('</div>', unsafe_allow_html=True)
