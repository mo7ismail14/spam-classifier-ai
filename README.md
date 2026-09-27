# Spam Classification Web Application

A full-stack Spam Classification web application consisting of 2 tiers:

- **`ai/` (Python + Flask)**: Machine Learning service powered by Scikit-Learn SVM, TF-IDF Vectorization, GridSearchCV with K-Fold Cross-Validation, and Joblib serialization. Also serves the JSON API directly (CORS-enabled).
- **`frontend/` (React.js + Material UI + Vite)**: Modern dashboard featuring an Autocomplete dropdown of sample emails, actual vs. predicted label comparison, prediction confidence indicators, and an interactive Confusion Matrix performance dashboard.

---

## 🏗️ Architecture Overview

```text
[ React + Material UI Frontend ] (Port 3000)
             │
             │ HTTP (Axios)
             ▼
[ Python + Flask AI Microservice ] (Port 5000)
             │
             ├── SVM Model (C, kernel, gamma tuned via GridSearchCV + 5-Fold CV)
             ├── TfidfVectorizer (4000 features, n-grams)
             └── Confusion Matrix Metrics (TP, TN, FP, FN, Accuracy, Precision, Recall, F1)
```

---

## 🚀 How to Run the Services

### 1. AI Microservice (`ai/`)

```bash
cd ai

# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the SVM model using GridSearchCV and K-Fold CV (saves to model/)
python train.py

# 3. Start the Flask AI server (Runs on http://localhost:5000)
python app.py
```

### 2. Frontend Application (`frontend/`)

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Start Vite development server (Runs on http://localhost:3000)
npm run dev
```

---

## 📡 API Endpoints

### AI Microservice (`http://localhost:5000`)

- **`GET /api/emails`**:
  Returns a curated sample of Enron emails with actual labels for the frontend Autocomplete dropdown.

- **`POST /api/predict`**:
  Accepts a JSON payload `{"text": "..."}` and returns:
  ```json
  {
    "prediction": "spam" | "ham",
    "confidence": 0.999,
    "metrics": {
      "tn": 300,
      "fp": 0,
      "fn": 0,
      "tp": 300,
      "best_params": {
        "C": 0.1,
        "gamma": "scale",
        "kernel": "rbf"
      }
    }
  }
  ```

---

## ✨ Features

- **Material-UI Autocomplete**: Quickly browse and filter sample emails from the Enron dataset.
- **Custom Input**: Test any custom email or text message in real-time.
- **Model vs. Actual Comparison**: Instant visual badge indicators showing whether the SVM model's prediction agrees with the ground truth label.
- **Confidence Meter**: Visual progress indicator depicting probability confidence.
- **Confusion Matrix Dashboard**: 2x2 matrix display showing True Negatives, False Positives, False Negatives, True Positives, and calculated Accuracy, Precision, Recall, and F1-Score.
