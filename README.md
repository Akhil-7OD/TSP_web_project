# 🚢 Ship Passenger Survival Prediction

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An interactive, responsive Machine Learning web application predicting the survival of individual passengers from the historic **RMS Titanic** disaster based on demographic and ticketing attributes.

> [!IMPORTANT]
> **Vessel vs. Passenger Clarification:**  
> This project is **NOT** a prediction about whether the ship itself survived (the vessel tragically sank on April 15, 1912). Instead, this machine learning system predicts whether an **individual passenger** would have survived based on their socioeconomic class, gender, age, family size aboard, fare paid, and embarkation port.

---

## 📌 Project Overview

When the RMS Titanic struck an iceberg in the North Atlantic, over 1,500 of the 2,224 passengers and crew lost their lives. Historical analyses reveal that survival outcomes were not purely random; safety protocols such as *"women and children first"* and ticket accommodation locations significantly influenced survival odds.

This application provides a streamlined interface where users can enter passenger information and obtain real-time predictions powered by a trained **Random Forest Classifier**, along with statistical confidence scores.

### Key Outputs:
- 🟢 **SURVIVED** (Model prediction = 1)
- 🔴 **DID NOT SURVIVE** (Model prediction = 0)
- **Model Confidence & Probability Scores** (e.g., *Survival Probability: 72%*, *Non-Survival Probability: 28%*)

---

## 🏗️ Project Architecture & Workflow

```
[ User Input (Web Form) ]
            │
            ▼
[ Feature Preprocessing & Encoding ]
  ├── Missing value imputation (Median Age, Median Fare, Mode Embarked)
  ├── Binary Sex mapping (Male: 0, Female: 1)
  └── One-Hot Encoding for Embarkation (Embarked_C, Embarked_Q, Embarked_S)
            │
            ▼
[ Random Forest Classifier (Cached Pipeline) ]
  └── Evaluates exact 9-feature input vector
            │
            ▼
[ Prediction Output & Probability Display ]
  ├── 🟢 PASSENGER SURVIVED / 🔴 PASSENGER DID NOT SURVIVE
  └── Statistical Confidence Gauge
```

---

## 🤖 Machine Learning Model Details

| Attribute | Specification |
| :--- | :--- |
| **Algorithm** | `RandomForestClassifier(random_state=42)` |
| **Dataset** | Titanic Passenger Training Set (`train.csv`) |
| **Target Variable** | `Survived` (0 = Did not survive, 1 = Survived) |
| **Validation Strategy** | Stratified Train-Test Split (80% Train, 20% Test, `random_state=42`) |
| **Test Accuracy** | **~78.9% - 80.0%** |
| **Caching Mechanism** | `@st.cache_resource` for instant, non-blocking inferences |

### The 9 Input Features (Exact Model Order)

The model utilizes strictly the 9 validated features below in this exact order:

1. **`Pclass`** — Passenger Class (`1` = 1st Class, `2` = 2nd Class, `3` = 3rd Class)
2. **`Sex`** — Gender (`0` = Male, `1` = Female)
3. **`Age`** — Age in years (fractional for infants < 1)
4. **`SibSp`** — Number of siblings or spouses traveling aboard
5. **`Parch`** — Number of parents or children traveling aboard
6. **`Fare`** — Passenger ticket fare in 1912 British Pounds (£)
7. **`Embarked_C`** — Port of Embarkation: Cherbourg (`1` if True, `0` otherwise)
8. **`Embarked_Q`** — Port of Embarkation: Queenstown (`1` if True, `0` otherwise)
9. **`Embarked_S`** — Port of Embarkation: Southampton (`1` if True, `0` otherwise)

### Preprocessing Protocol

- **Age Imputation:** Missing values are filled with the **median Age** (`28.0`).
- **Embarked Imputation:** Missing values are filled with the **mode** (`'S'`).
- **Fare Imputation:** Missing values are filled with the **median Fare** (`14.45`).
- **Sex Encoding:** Binary conversion (`male` ➔ `0`, `female` ➔ `1`).
- **Embarked Encoding:** One-hot encoded into `Embarked_C`, `Embarked_Q`, and `Embarked_S`.
- **Excluded Features:**
  - `Cabin`: Excluded due to excessive missing values (>77%).
  - `PassengerId`: Excluded as an arbitrary identifier lacking generalizability.
  - `Name` & `Ticket`: Excluded to prevent overfitting on specific strings.

---

## 📁 Repository Structure

```
ship-passenger-survival-prediction/
├── app.py              # Streamlit application with custom styling & cached model
├── train.csv           # Titanic passenger training dataset
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Git exclusion rules
```

---

## 🚀 Getting Started Locally

### 1. Prerequisites
Ensure you have **Python 3.9 or higher** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/ship-passenger-survival-prediction.git
cd ship-passenger-survival-prediction
```

### 3. (Optional) Create a Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Ensure `train.csv` is Present
Make sure `train.csv` is placed in the project root directory alongside `app.py`.

### 6. Run the Application
```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud

This project is configured for deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push your repository to **GitHub** (see instructions below).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. Click **New app**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Click **Deploy!**

---

## 🐙 Uploading to GitHub

To publish this project to your GitHub profile:

```bash
# 1. Initialize git in the project folder
git init

# 2. Add files
git add .

# 3. Commit changes
git commit -m "Initial commit: Ship Passenger Survival Prediction web app"

# 4. Create a new repository on GitHub (e.g. ship-passenger-survival-prediction)
# Then link your local repository:
git branch -M main
git remote add origin https://github.com/<your-github-username>/ship-passenger-survival-prediction.git

# 5. Push to GitHub
git push -u origin main
```

---

## 📊 Ethical AI & Probability Notice

The survival probabilities provided by this application reflect statistical patterns learned by a **Random Forest Classifier** from historical 1912 manifest data. They represent model confidence estimates and should not be interpreted as absolute deterministic statements.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
