# Customer Churn Prediction

A machine learning application that predicts customer churn using a trained RandomForest model. Built with Streamlit for an intuitive web interface.

## Overview

This project analyzes telco customer data to identify customers at risk of churning (leaving the service). The predictive model was trained on historical customer behavior, service features, and billing information to provide accurate churn predictions.

## Features

- **Interactive Web Interface**: Simple form-based UI to input customer details
- **Real-time Predictions**: Instant churn probability scores
- **RandomForest Model**: Tuned hyperparameters for optimal performance
- **Feature Transparency**: View encoded features sent to the model
- **Comprehensive Input**: 19 customer attributes for nuanced predictions

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

Start the Streamlit server:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your default browser.

## Model Details

- **Algorithm**: RandomForest Classifier (tuned via RandomizedSearchCV)
- **Training Data**: 7,043 customer records
- **Features**: 19 (demographic, service, and billing attributes)
- **Performance**: Cross-validated with 5-fold CV on SMOTE-balanced data
- **Saved Artifact**: `customer_churn_model.pkl`

### Input Features

**Demographics:**
- Gender
- Senior Citizen status
- Partner status
- Dependents

**Services:**
- Phone Service
- Multiple Lines
- Internet Service type
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

**Account:**
- Contract type
- Paperless Billing
- Payment Method
- Tenure (months)
- Monthly Charges
- Total Charges

## Project Structure

```
.
├── app.py                        # Streamlit application
├── customer_churn_model.pkl      # Trained RandomForest model
├── requirements.txt              # Python dependencies
├── CustomerChurnPrediction.ipynb # Analysis and training notebook
└── README.md                     # This file
```

## How It Works

1. User inputs customer details via the web form
2. Categorical features are label-encoded using predefined mappings
3. Data is formatted to match the model's expected feature order
4. RandomForest classifier predicts churn probability
5. Results display prediction and confidence/probability score

## Requirements

- streamlit
- pandas
- scikit-learn

See `requirements.txt` for specific versions.

## Notes

- The notebook file (`CustomerChurnPrediction.ipynb`) contains the full exploratory data analysis, feature engineering, and model training pipeline
- The app uses a lightweight encoding scheme that mirrors the training preprocessing
- Model predictions are probability-based; use the confidence score to assess prediction strength


---

**Built with:** Python, Scikit-learn, Pandas, Streamlit
