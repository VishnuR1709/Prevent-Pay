# Prevent Pay — Fraud Detection App

An AI-powered credit card fraud detection system. Enter transaction details and get an instant prediction — **Fraudulent** or **Legitimate** — powered by a trained XGBoost model.

> Final Year Project — 97.90% model accuracy on balanced test data.

---

Dataset
The dataset used in this project is obtained from Kaggle. The 2 datasets I played with to get the actual result of 98.84% is:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 
⁠https://www.kaggle.com/datasets/kartik2112/fraud-detection 

---

## Project Structure

```
Prevent-Pay/
├── app.py                  # Flask backend (API + server)
├── save_model.py           # Train & save model from CSV data
├── requirements.txt        # Python dependencies
├── model.pkl               # Trained XGBoost model      ← generated
├── scaler.pkl              # Fitted StandardScaler       ← generated
├── feature_columns.pkl     # Feature column order        ← generated
├── templates/
│   └── index.html          # Frontend form (HTML + JS)
├── static/
│   └── style.css           # Styling
└── archive/
    ├── fraudTrain.csv      # Training data
    └── fraudTest.csv       # Test data
```

---

## Option A — Quick Start with Pre-Created Model Files

Use this if you already have `model.pkl`, `scaler.pkl`, and `feature_columns.pkl`.

**1. Set up the environment**

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Place the pkl files in the project root**

```
Prevent-Pay/
├── model.pkl
├── scaler.pkl
└── feature_columns.pkl
```

**3. Start the server**

```bash
python app.py
```

**4. Open in your browser**

```
http://localhost:5000
```

---

## Option B — Generate Model Files from Scratch

Use this if you have the raw CSV data and want to train the model yourself.

**1. Set up the environment**

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Get the dataset**

Download from Kaggle and place `fraudTrain.csv` inside the `archive/` folder:

> [kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

```
Prevent-Pay/
└── archive/
    └── fraudTrain.csv
```

**3. Train and save the model**

```bash
python save_model.py                            # default path: archive/fraudTrain.csv
python save_model.py /path/to/fraudTrain.csv    # or specify a custom path
```

You should see:

```
Loading data from: archive/fraudTrain.csv
Loaded 1,296,675 rows after dropna.
Extracting temporal features...
Balancing classes (downsample majority to match minority)...
Balanced dataset: 15,012 rows  (fraud=7,506, non-fraud=7,506)
One-hot encoding category and state...
Feature count: 80
Training XGBoost model...

Accuracy: 97.90%
...

Done! Files saved:
  model.pkl
  scaler.pkl
  feature_columns.pkl
```

**4. Start the server**

```bash
python app.py
```

**5. Open in your browser**

```
http://localhost:5000
```

---

## Using the App

Fill in the transaction form and click **Analyze Transaction**.

| Field | Description |
|---|---|
| Credit Card Number | Card number (up to 19 digits) |
| Transaction Amount | Amount in USD |
| Transaction Date & Time | When the transaction occurred |
| Gender | Cardholder gender |
| Category | Merchant category (e.g. grocery, travel) |
| State | US state of the transaction |
| Cardholder Lat / Long | Cardholder's geographic coordinates |
| City Population | Population of the cardholder's city |
| Merchant Lat / Long | Merchant's geographic coordinates |
| Date of Birth | Cardholder's date of birth |

The result card shows **Legitimate** (green) or **Fraudulent** (red) along with a confidence percentage.

---

## Test Data

Use these real transactions from the dataset to verify the app is working correctly.

### Fraudulent Transaction
| Field | Value |
|---|---|
| Credit Card Number | `4613314721966` |
| Amount | `281.06` |
| Date & Time | `2019-01-02T01:06` |
| Gender | Male |
| Category | Grocery Pos |
| State | `NC` |
| Cardholder Lat / Long | `35.9946` / `-81.7266` |
| City Population | `885` |
| Merchant Lat / Long | `36.4301` / `-81.1795` |
| Date of Birth | `1988-09-15` |

### Legitimate Transaction
| Field | Value |
|---|---|
| Credit Card Number | `4767265376804500` |
| Amount | `94.63` |
| Date & Time | `2019-01-01T00:04` |
| Gender | Female |
| Category | Gas Transport |
| State | `PA` |
| Cardholder Lat / Long | `40.375` / `-75.2045` |
| City Population | `2158` |
| Merchant Lat / Long | `40.6534` / `-76.1527` |
| Date of Birth | `1961-06-19` |

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost (XGBClassifier) |
| Features | 80 (numerical + one-hot encoded category & state) |
| Training rows | ~15,000 (balanced downsample) |
| Test accuracy | 97.90% |
| Output | Binary — 0 (Legitimate) / 1 (Fraudulent) |

The model uses `scale_pos_weight` to handle class imbalance and a `StandardScaler` for feature normalization. Feature columns are saved in a fixed order (`feature_columns.pkl`) so inference always matches training.

---

## Tech Stack

- **Backend** — Python, Flask
- **ML** — XGBoost, scikit-learn
- **Data** — Pandas, NumPy
- **Frontend** — HTML, CSS, Vanilla JavaScript
- **Dataset** — [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

---

## Dependencies

```
flask>=3.0.0
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.26.0
```

Install all with:

```bash
pip install -r requirements.txt
```
