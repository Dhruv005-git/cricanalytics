# CricAnalytics — AI-Driven IPL Match Prediction Platform

An end-to-end sports analytics platform that predicts IPL cricket match outcomes using machine learning on big data infrastructure. Processes 6,255 historical match files (1.4M+ deliveries) via Apache Spark on HDFS, trains an ML model using Spark MLlib, and serves real-time predictions through a Flask REST API consumed by an interactive Streamlit dashboard.

---

## Demo

> Streamlit dashboard with live win probability predictions, team rankings, and player comparisons.

---

## Architecture

```
Cricsheet JSON Files (6,255)
        │
        ▼
  HDFS (Hadoop 3.2.1)          ← Raw JSON + cleaned Parquet + analytics tables
        │
        ▼
  Apache Spark 3.5.0 (PySpark) ← ETL pipeline + feature engineering + ML training
        │
        ▼
  Spark MLlib Pipeline          ← VectorAssembler → StandardScaler → Logistic Regression
        │
        ▼
  Flask REST API (17 endpoints) ← Serves predictions + analytics on port 8000
        │
        ▼
  Streamlit + Plotly Dashboard  ← Player comparison, team rankings, match prediction
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data storage | HDFS (Hadoop 3.2.1) on Docker |
| Data processing | Apache Spark 3.5.0 / PySpark |
| ML | Spark MLlib (Logistic Regression, Random Forest, GBT) |
| API | Flask 3.0 |
| Frontend | Streamlit + Plotly |
| Containerization | Docker |

---

## Dataset

- **Source:** [Cricsheet](https://cricsheet.org) — open ball-by-ball cricket data
- **IPL data:** 1,169 JSON match files, seasons 2008–2025 (18 seasons), 278,205 deliveries
- **T20I data:** 5,086 JSON match files, seasons 2005–2026, 1,150,989 deliveries
- **Combined:** 6,255 files, 1,429,194 individual ball deliveries stored in HDFS as Parquet

---

## ML Model

### Feature Engineering — 13 Features per Match

| # | Feature | What it captures |
|---|---|---|
| 1 | `team_a_overall_pct` | All-time IPL win % for team A |
| 2 | `team_b_overall_pct` | All-time IPL win % for team B |
| 3 | `overall_win_pct_diff` | Team A minus team B overall win % |
| 4 | `h2h_win_pct` | Team A win % in all historical head-to-head matches vs team B |
| 5 | `venue_win_pct_a` | Team A win % at the specific stadium |
| 6 | `recent_win_pct_a` | Team A win % in last 10 matches (current momentum) |
| 7 | `recent_win_pct_b` | Team B win % in last 10 matches |
| 8 | `recent_win_pct_diff` | Team A minus team B recent form — most predictive feature (corr=0.31) |
| 9 | `toss_advantage` | Historical win % at this venue for this toss decision |
| 10 | `toss_decision_num` | 1.0 = chose to bat, 0.0 = chose to field |
| 11 | `toss_winner_is_team_a` | 1.0 = team A won toss |
| 12 | `team_a_bats_first` | 1.0 = team A bats first (class imbalance fix) |
| 13 | `season_norm` | Normalised year: 2008=0.0 to 2026=1.0 |

### Model Comparison Results

| Model | AUC-ROC | Accuracy | Decision |
|---|---|---|---|
| **Logistic Regression** | **0.6346** | **59.1%** | ✅ Selected |
| Random Forest (200 trees) | 0.6081 | 57.5% | Overfits (~11% gap) |
| Gradient Boosted Trees | 0.5554 | 55.4% | Significant overfitting (~20% gap) |

### Train / Test Split
Temporal split by season — simulates real-world prediction of future matches:
- **Training:** seasons 2008–2022 (1,864 samples after data augmentation)
- **Test:** seasons 2023–2026 (428 samples — genuinely unseen data)

### Class Imbalance Fix
Batting second wins 64.2% of IPL matches. Fixed via data augmentation — each match generates two perspectives (team A bats first / team A bats second), doubling the dataset from 1,169 to 2,338 samples and balancing the label distribution to 61.8% / 38.2%.

### Spark MLlib Pipeline
```python
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_raw')
scaler    = StandardScaler(inputCol='features_raw', outputCol='features', withMean=True, withStd=True)
lr        = LogisticRegression(featuresCol='features', labelCol='label', maxIter=100, regParam=0.01)
pipeline  = Pipeline(stages=[assembler, scaler, lr])
```

---

## Project Structure

```
cricanalytics/
├── backend/
│   ├── etl/              # PySpark ETL pipeline — JSON ingestion, Parquet conversion
│   ├── features/         # Feature engineering — lookup table joins
│   ├── train_model.py    # Spark MLlib training pipeline
│   ├── predict.py        # Inference with saved PipelineModel
│   └── api.py            # Flask REST API (17 endpoints)
└── frontend/
    └── app.py            # Streamlit dashboard — player comparison, team rankings, prediction
```

---

## Running Locally

### Prerequisites
- Docker + Docker Compose
- Python 3.9+

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Dhruv005-git/cricanalytics.git
cd cricanalytics

# 2. Start Hadoop + Spark containers
docker-compose up -d

# 3. Run the ETL pipeline
docker exec spark-container python backend/etl/ingest.py

# 4. Train the model
docker exec spark-container python backend/train_model.py

# 5. Start the Flask API
docker exec spark-container python backend/api.py

# 6. Launch the Streamlit dashboard
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## Key Challenges Solved

| Challenge | Solution |
|---|---|
| Deeply nested JSON (innings > overs > deliveries) | PySpark `posexplode()` to flatten into 278,205 delivery rows |
| Player names becoming column headers (d'souza bug) | Explicit PySpark schema — Spark never infers schema from data |
| 64% class imbalance (batting second wins) | Dataset doubling via dual team perspectives per match |
| Venue name inconsistencies across seasons | Partial string matching with exact match → `contains()` fallback |
| Small ML dataset (1,169 matches) | Logistic Regression + L2 regularisation (regParam=0.01); temporal split prevents leakage |
| Season format: `2008` vs `2007/08` | Explicit `SEASON_MAP` lookup dictionary |

---

## References

- [Cricsheet](https://cricsheet.org) — open cricket data
- [Apache Spark MLlib](https://spark.apache.org/docs/latest/ml-guide.html)
- [Hadoop HDFS](https://hadoop.apache.org/docs/stable/)
- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *JRSS.*
